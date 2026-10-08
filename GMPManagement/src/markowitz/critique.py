from __future__ import annotations

import numpy as np
import pandas as pd

from src.markowitz.mean_variance import max_sharpe
from src.analytics.stats import mean_returns, cov_matrix


def estimation_error_experiment(returns: pd.DataFrame, n_sims: int = 300,
                                seed: int = 42, rf: float = 0.0,
                                min_train_frac: float = 0.5,
                                max_train_frac: float = 0.8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(returns)
    lo = int(min_train_frac * n)
    hi = int(max_train_frac * n)

    is_sharpes, oos_sharpes = [], []
    for _ in range(n_sims):
        split = int(rng.integers(lo, hi))
        train = returns.iloc[:split]
        test = returns.iloc[split:]

        mu = mean_returns(train)
        cov = cov_matrix(train)
        try:
            w = max_sharpe(mu, cov, rf=rf)
        except Exception:
            continue

        r_is = (train * w).sum(axis=1)
        r_oos = (test * w).sum(axis=1)

        rf_monthly = rf / 12
        if r_is.std() > 0:
            is_sharpes.append((r_is.mean() - rf_monthly) /
                              r_is.std() * np.sqrt(12))
        if r_oos.std() > 0:
            oos_sharpes.append((r_oos.mean() - rf_monthly) /
                               r_oos.std() * np.sqrt(12))

    m = min(len(is_sharpes), len(oos_sharpes))
    return pd.DataFrame({
        "in_sample_sharpe": is_sharpes[:m],
        "out_of_sample_sharpe": oos_sharpes[:m],
    })


def sensitivity_to_mu(returns: pd.DataFrame, delta: float = 0.001,
                      n_pert: int = 50, seed: int = 0,
                      rf: float = 0.0):
    rng = np.random.default_rng(seed)
    mu = mean_returns(returns)
    cov = cov_matrix(returns)
    w0 = max_sharpe(mu, cov, rf=rf)

    perturbations = []
    for _ in range(n_pert):
        noise = rng.normal(0.0, delta, size=len(mu))
        try:
            w = max_sharpe(mu + noise, cov, rf=rf)
            perturbations.append(w)
        except Exception:
            continue

    W = pd.DataFrame(perturbations)
    stats = W.describe().T[["mean", "std", "min", "max"]]
    return w0, stats