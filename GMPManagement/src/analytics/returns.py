from __future__ import annotations
import numpy as np
import pandas as pd


def log_to_simple(r):
    return np.expm1(r)


def simple_to_log(r):
    return np.log1p(r)


def cumulative_wealth(returns: pd.Series, return_type: str = "simple",
                      base: float = 1.0) -> pd.Series:
    r = returns.fillna(0.0)
    if return_type == "simple":
        return base * (1.0 + r).cumprod()
    if return_type == "log":
        return base * np.exp(r.cumsum())
    raise ValueError(return_type)


def portfolio_returns_from_weights(asset_returns: pd.DataFrame,
                                   weights: pd.Series,
                                   return_type: str = "log") -> pd.Series:
    if return_type == "simple":
        return (asset_returns * weights).sum(axis=1)
    if return_type == "log":
        simple = log_to_simple(asset_returns)
        port_simple = (simple * weights).sum(axis=1)
        return simple_to_log(port_simple)
    raise ValueError(return_type)