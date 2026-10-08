from __future__ import annotations

import numpy as np
import pandas as pd

from src.markowitz.mean_variance import (
    min_variance,
    target_return_portfolio,
    portfolio_stats,
)


def compute_frontier(mu: pd.Series, cov: pd.DataFrame, n_points: int = 50,
                     long_only: bool = True, max_weight: float = 1.0,
                     rf: float = 0.0):
    w_min = min_variance(mu, cov, long_only, max_weight)
    r_min = float(mu.values @ w_min.values)
    r_max = float(mu.max())

    targets = np.linspace(r_min, r_max, n_points)
    rows, weights = [], {}

    for t in targets:
        w = target_return_portfolio(mu, cov, t, long_only, max_weight)
        if w is None:
            continue
        s = portfolio_stats(w, mu, cov, rf)
        rows.append({"target_return": t, **s})
        weights[t] = w

    frontier = pd.DataFrame(rows).set_index("target_return")
    return frontier, weights


def random_portfolios(mu: pd.Series, cov: pd.DataFrame, n: int = 5000,
                      long_only: bool = True, seed: int = 42,
                      rf: float = 0.0):
    rng = np.random.default_rng(seed)
    n_assets = len(mu)
    W = rng.random((n, n_assets))
    W /= W.sum(axis=1, keepdims=True)

    rets = W @ mu.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov.values, W))
    sharpes = np.where(vols > 0, (rets - rf) / vols, np.nan)

    df = pd.DataFrame({"return": rets, "vol": vols, "sharpe": sharpes})
    return df, W