
from __future__ import annotations

import numpy as np
import pandas as pd

from src.markowitz.mean_variance import max_sharpe, min_variance
from src.analytics.stats import mean_returns, cov_matrix
from src.analytics.returns import log_to_simple


def walk_forward(returns: pd.DataFrame, window: int = 60, rebalance: int = 12,
                 method: str = "max_sharpe", long_only: bool = True,
                 max_weight: float = 1.0, rf: float = 0.0,
                 rf_series: pd.Series | None = None,
                 return_type: str = "log"):
    if return_type not in {"log", "simple"}:
        raise ValueError("return_type must be 'log' or 'simple'")

    dates = returns.index
    weights_history = {}
    port_returns = []

    for i in range(window, len(dates), rebalance):
        train = returns.iloc[i - window:i]
        train_simple = log_to_simple(train) if return_type == "log" else train
        mu = mean_returns(train_simple)
        cov = cov_matrix(train_simple)

        rf_train = rf
        if rf_series is not None:
            rf_window = rf_series.reindex(train.index).dropna()
            if not rf_window.empty:
                rf_train = float(rf_window.mean() * 12)

        if method == "max_sharpe":
            w = max_sharpe(mu, cov, rf=rf_train,
                           long_only=long_only, max_weight=max_weight)
        elif method == "min_var":
            w = min_variance(mu, cov,
                             long_only=long_only, max_weight=max_weight)
        else:
            raise ValueError(f"unknown method: {method}")

        weights_history[dates[i]] = w

        end = min(i + rebalance, len(dates))
        test = returns.iloc[i:end]
        test_simple = log_to_simple(test) if return_type == "log" else test
        port_returns.append((test_simple * w).sum(axis=1))

    if not port_returns:
        raise RuntimeError("No OOS period was calculated")
    port = pd.concat(port_returns).sort_index()
    weights_df = pd.DataFrame(weights_history).T
    return port, weights_df


def equal_weight(returns: pd.DataFrame, rebalance: int = 12,
                 return_type: str = "log") -> pd.Series:
    if return_type not in {"log", "simple"}:
        raise ValueError("return_type must be 'log' or 'simple'")

    w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    simple = log_to_simple(returns) if return_type == "log" else returns
    dates = returns.index
    port_returns = []

    for i in range(0, len(dates), rebalance):
        end = min(i + rebalance, len(dates))
        test = simple.iloc[i:end]
        port_returns.append((test * w).sum(axis=1))

    return pd.concat(port_returns).sort_index()


def compute_turnover(weights_df: pd.DataFrame) -> pd.Series:
    return weights_df.diff().abs().sum(axis=1).fillna(0.0)