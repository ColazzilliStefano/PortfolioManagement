"""Hierarchical Risk Parity (Lopez de Prado 2016).

Implementation of the HRP algorithm:
1. Tree clustering on the correlation matrix
2. Quasi-diagonalization
3. Recursive bisection for weights

HRP is robust because it avoids the inversion of the covariance matrix,
which is the main source of instability in Markowitz.

Fixes applied (v1.3):
- Return tuple[pd.Series, pd.DataFrame] annotation (Fix 4)
- max_weight support with post-processing (Fix 5)
- Fallback that respects max_weight (Fix 5)
- Correction of the recursive bisection

Reference: Lopez de Prado (2016), "Building Diversified Portfolios
that Outperform Out-of-Sample", Journal of Portfolio Management.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from loguru import logger


# Distances and clustering


def _correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """
    Distance from correlation: d_ij = sqrt(0.5 * (1 - rho_ij)).
    
    Clips the input to [0, 2] to avoid NaN when the correlation
    sample is slightly > 1 or < -1 due to numerical error.
    """
    c = np.clip(1.0 - corr.values, 0.0, 2.0)
    return pd.DataFrame(np.sqrt(0.5 * c),
                        index=corr.index, columns=corr.columns)


def _quasi_diagonalize(link: np.ndarray) -> list:
    """
    Reorganizes the order of the assets using the dendrogram.
    Returns a list of indices in the quasi-diagonal order.
    """
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()


# Recursive bisection

def _cluster_variance(cov: pd.DataFrame, cluster_items: list) -> float:
    """Variance of a cluster with inverse-variance weights."""
    c = cov.loc[cluster_items, cluster_items]
    ivp = 1.0 / np.diag(c.values)
    ivp = ivp / ivp.sum()
    w = ivp.reshape(-1, 1)
    return float((w.T @ c.values @ w).item())


def _recursive_bisection(cov: pd.DataFrame, sort_ix: list) -> pd.Series:
    """
    Assigns weights recursively by bisecting the dendrogram.
    Standard implementation by Lopez de Prado (2016).
    """
    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]

    while len(clusters) > 0:
        # bisect each cluster > 1
        clusters = [
            c[i:j]
            for c in clusters
            for i, j in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]

        # assign weights for each pair of sub-clusters
        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                break
            c0 = clusters[i]
            c1 = clusters[i + 1]
            v0 = _cluster_variance(cov, c0)
            v1 = _cluster_variance(cov, c1)
            alpha = 1 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1 - alpha

    return w


# Cap and normalization

def _apply_max_weight(w: pd.Series, max_weight: float) -> pd.Series:
    """
    Applies cap and renormalizes.
    If after the cap the sum is 0, raises ValueError.
    """
    w = w.clip(upper=max_weight)
    if w.sum() <= 0:
        raise ValueError("Null weight sum after cap")
    return w / w.sum()


# HRP weights


def hrp_weights(cov: pd.DataFrame,
                corr: pd.DataFrame | None = None) -> pd.Series:
    """
    HRP weights from covariance (and optionally correlation).

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance of the assets.
    corr : pd.DataFrame | None
        Correlation. If None, calculated from cov.

    Returns
    -------
    pd.Series of HRP weights.
    """
    n = len(cov)
    if n == 0:
        raise ValueError("Empty covariance matrix")

    if corr is None:
        d = np.sqrt(np.diag(cov.values))
        corr = cov / np.outer(d, d)

    dist = _correlation_distance(corr)

    try:
        link = linkage(squareform(dist.values, checks=False), method="single")
    except Exception as e:
        logger.warning(f"HRP linkage failed: {e}. Using equal weight.")
        return pd.Series(1.0 / n, index=cov.columns, name="hrp")

    sort_ix = _quasi_diagonalize(link)
    sorted_assets = [cov.index[i] for i in sort_ix]
    w = _recursive_bisection(cov, sorted_assets)

    # reorder using the original order
    w = w.reindex(cov.index)
    w = w / w.sum()
    w.name = "hrp"
    return w


# Walk-forward backtest

def hrp_backtest(returns: pd.DataFrame,
                 window: int = 60,
                 rebalance: int = 6,
                 max_weight: float = 1.0,
                 return_type: str = "log"
                 ) -> tuple[pd.Series, pd.DataFrame]:
    """
    Walk-forward HRP: at each rebalancing re-estimate cov/corr and weights.

    Parameters
    ----------
    returns : pd.DataFrame
        Monthly returns (log or simple).
    window : int
        Months of training.
    rebalance : int
        Months between rebalancings.
    max_weight : float
        Cap for single asset (post-processing).
    return_type : str
        "log" | "simple".

    Returns
    -------
    (port_returns, weights_history) : tuple[pd.Series, pd.DataFrame]
    """
    from src.analytics.returns import log_to_simple

    dates = returns.index
    n = returns.shape[1]

    if max_weight < 1.0 / n:
        raise ValueError(
            f"max_weight={max_weight:.4f} infeasible with {n} assets. "
            f"Minimum required: {1.0/n:.4f}"
        )

    port_returns = []
    weights_history = {}

    for i in range(window, len(dates), rebalance):
        train = returns.iloc[i - window:i]
        train_simple = log_to_simple(train) if return_type == "log" else train

        try:
            cov = train_simple.cov() * 12
            corr = train_simple.corr()
            w = hrp_weights(cov, corr)
            if max_weight < 1.0:
                w = _apply_max_weight(w, max_weight)
        except Exception as e:
            logger.warning(f"HRP failed at {dates[i].date()}: {e}. "
                           f"Using fallback.")
            w = _equal_weight_fallback(train.columns, max_weight)

        weights_history[dates[i]] = w

        end = min(i + rebalance, len(dates))
        test = returns.iloc[i:end]
        test_simple = log_to_simple(test) if return_type == "log" else test
        port_returns.append((test_simple * w).sum(axis=1))

    if not port_returns:
        raise RuntimeError("No OOS period calculated")

    port = pd.concat(port_returns).sort_index()
    port.name = "HRP"

    weights_df = pd.DataFrame(weights_history).T
    weights_df.index = pd.to_datetime(weights_df.index)

    return port, weights_df


def _equal_weight_fallback(columns, max_weight: float) -> pd.Series:
    """
    Equal-weight fallback that respects max_weight.
    """
    n = len(columns)
    w = np.full(n, 1.0 / n)

    if 1.0 / n > max_weight + 1e-12:
        w = np.full(n, max_weight)
        w = w / w.sum()

    return pd.Series(w, index=columns, name="hrp")