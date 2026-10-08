"""Risk Parity benchmark (Equal Risk Contribution).

Implementation of Maillard-Roncalli-Teiletche (2010):
weights are chosen so that each asset contributes equally
to the total volatility of the portfolio.

    w_i * (Sigma w)_i = w_j * (Sigma w)_j   for each i, j

Solved numerically via SLSQP.

Fixes applied (v1.3):
- Return tuple[pd.Series, pd.DataFrame] annotation (Fix 4)
- max_weight >= 1/n_assets validation (Fix 5)
- Fallback that respects max_weight (Fix 5)

Reference: Maillard, Roncalli, Teiletche (2010),
"The Properties of Equally Weighted Risk Contribution Portfolios",
Journal of Portfolio Management.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from loguru import logger


# Risk contribution

def _risk_contribution(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Returns the vector of risk contributions."""
    sigma_p = float(np.sqrt(w @ cov @ w))
    if sigma_p <= 0:
        return np.zeros_like(w)
    mcr = cov @ w / sigma_p          # marginal contribution
    return w * mcr                    # total contribution


def _erc_objective(w: np.ndarray, cov: np.ndarray) -> float:
    """Distance from equal contribution."""
    rc = _risk_contribution(w, cov)
    target = rc.mean()
    return float(((rc - target) ** 2).sum())


# Validation

def _validate_max_weight(n: int, max_weight: float) -> None:
    """Raises ValueError if max_weight is infeasible with n assets."""
    if max_weight < 1.0 / n:
        raise ValueError(
            f"max_weight={max_weight:.4f} infeasible with {n} assets. "
            f"Minimum required: {1.0/n:.4f}"
        )
    if max_weight > 1.0:
        raise ValueError(f"max_weight={max_weight} > 1.0 not valid")


# ERC weights

def risk_parity_weights(cov: pd.DataFrame,
                        long_only: bool = True,
                        max_weight: float = 1.0) -> pd.Series:
    n = len(cov)
    if n == 0:
        raise ValueError("Empty covariance matrix")

    _validate_max_weight(n, max_weight)

    C = cov.values
    x0 = np.ones(n) / n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight)] * n if long_only else [(-1.0, 1.0)] * n

    res = minimize(
        fun=lambda w: _erc_objective(w, C),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 2000},
    )

    if not res.success:
        logger.warning(f"Risk parity did not converge: {res.message}. "
                       f"Using equal weight (max_weight respected).")
        return _equal_weight_fallback(cov.index, max_weight)

    return pd.Series(res.x, index=cov.columns, name="risk_parity")


def _equal_weight_fallback(columns, max_weight: float) -> pd.Series:
    """
    Equal-weight fallback that respects max_weight.
    If equal-weight violates max_weight, redistributes the excess weights.
    """
    n = len(columns)
    w = np.full(n, 1.0 / n)

    # if 1/n > max_weight, cap and redistribute iteratively
    if 1.0 / n > max_weight + 1e-12:
        w = np.full(n, max_weight)
        residual = 1.0 - w.sum()
        # distribute the residual (which could be negative if n*max_weight < 1)
        if residual > 1e-12:
            # should not happen if max_weight >= 1/n, but for safety
            w = w / w.sum()
        else:
            w = w / w.sum()

    return pd.Series(w, index=columns, name="risk_parity")


# Walk-forward backtest

def risk_parity_backtest(returns: pd.DataFrame,
                         window: int = 60,
                         rebalance: int = 6,
                         long_only: bool = True,
                         max_weight: float = 1.0,
                         return_type: str = "log"
                         ) -> tuple[pd.Series, pd.DataFrame]:

    from src.analytics.returns import log_to_simple
    from src.estimators.shrinkage_cov import shrinkage_cov

    dates = returns.index
    n = returns.shape[1]
    _validate_max_weight(n, max_weight)

    port_returns = []
    weights_history = {}

    for i in range(window, len(dates), rebalance):
        train = returns.iloc[i - window:i]
        train_simple = log_to_simple(train) if return_type == "log" else train

        try:
            cov = shrinkage_cov(train_simple, method="ledoit_wolf")
        except Exception as e:
            logger.warning(f"Shrinkage failed at {dates[i].date()}: {e}. "
                           f"Using sample covariance.")
            cov = train_simple.cov() * 12

        try:
            w = risk_parity_weights(cov, long_only=long_only,
                                    max_weight=max_weight)
        except Exception as e:
            logger.warning(f"Risk parity failed at {dates[i].date()}: {e}. "
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
    port.name = "Risk Parity"

    weights_df = pd.DataFrame(weights_history).T
    weights_df.index = pd.to_datetime(weights_df.index)

    return port, weights_df