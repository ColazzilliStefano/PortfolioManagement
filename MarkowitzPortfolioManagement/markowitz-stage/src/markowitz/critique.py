from __future__ import annotations

import numpy as np
import pandas as pd

from src.markowitz.mean_variance import max_sharpe
from src.analytics.stats import mean_returns, cov_matrix
from src.analytics.returns import log_to_simple


def estimation_error_experiment(returns: pd.DataFrame, n_sims: int = 300,
                                seed: int = 42, rf: float = 0.0,
                                rf_series: pd.Series | None = None,
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

        train_simple = log_to_simple(train)
        test_simple = log_to_simple(test)
        mu = mean_returns(train_simple)
        cov = cov_matrix(train_simple)
        try:
            w = max_sharpe(mu, cov, rf=rf)
        except Exception:
            continue

        r_is = (train_simple * w).sum(axis=1)
        r_oos = (test_simple * w).sum(axis=1)

        rf_is = (rf_series.reindex(r_is.index).fillna(0.0)
                 if rf_series is not None else pd.Series(rf / 12, index=r_is.index))
        rf_oos = (rf_series.reindex(r_oos.index).fillna(0.0)
                  if rf_series is not None else pd.Series(rf / 12, index=r_oos.index))
        if r_is.std() > 0:
            is_sharpes.append((r_is - rf_is).mean() /
                              r_is.std() * np.sqrt(12))
        if r_oos.std() > 0:
            oos_sharpes.append((r_oos - rf_oos).mean() /
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
    returns_simple = log_to_simple(returns)
    mu = mean_returns(returns_simple)
    cov = cov_matrix(returns_simple)
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