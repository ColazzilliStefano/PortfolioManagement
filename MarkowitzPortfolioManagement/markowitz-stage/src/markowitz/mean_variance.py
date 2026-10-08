from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


#Portfolio statistics
def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(np.dot(w, mu))


def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(w @ cov @ w))


def portfolio_sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                     rf: float = 0.0) -> float:
    r = portfolio_return(w, mu)
    v = portfolio_vol(w, cov)
    if v <= 0:
        return np.nan
    return (r - rf) / v


def portfolio_stats(w: pd.Series, mu: pd.Series, cov: pd.DataFrame,
                    rf: float = 0.0) -> dict:
    wv = w.values
    muv = mu.values
    covv = cov.values
    return {
        "return": portfolio_return(wv, muv),
        "vol": portfolio_vol(wv, covv),
        "sharpe": portfolio_sharpe(wv, muv, covv, rf),
    }


#Constraints and bounds
def _sum_to_one(n: int) -> list:
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]


def _bounds(n: int, long_only: bool, max_weight: float) -> list:
    if long_only:
        return [(0.0, max_weight)] * n
    return [(-1.0, 1.0)] * n


def _x0(n: int) -> np.ndarray:
    return np.ones(n) / n


#Optimal portfolios

def min_variance(mu: pd.Series, cov: pd.DataFrame,
                 long_only: bool = True, max_weight: float = 1.0) -> pd.Series:
    n = len(mu)
    res = minimize(
        fun=lambda w: portfolio_vol(w, cov.values),
        x0=_x0(n),
        method="SLSQP",
        bounds=_bounds(n, long_only, max_weight),
        constraints=_sum_to_one(n),
        options={"ftol": 1e-12, "max": 2000},
    )
    if not res.success:
        raise RuntimeError(f"min_variance failed: {res.message}")
    return pd.Series(res.x, index=mu.index, name="min_var")


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0,
               long_only: bool = True, max_weight: float = 1.0) -> pd.Series:
    n = len(mu)
    res = minimize(
        fun=lambda w: -portfolio_sharpe(w, mu.values, cov.values, rf),
        x0=_x0(n),
        method="SLSQP",
        bounds=_bounds(n, long_only, max_weight),
        constraints=_sum_to_one(n),
        options={"ftol": 1e-12, "max": 2000},
    )
    if not res.success:
        raise RuntimeError(f"max_sharpe failed: {res.message}")
    return pd.Series(res.x, index=mu.index, name="max_sharpe")


def target_return_portfolio(mu: pd.Series, cov: pd.DataFrame, target: float,
                            long_only: bool = True,
                            max_weight: float = 1.0) -> pd.Series | None:
    n = len(mu)
    cons = _sum_to_one(n) + [
        {"type": "eq",
         "fun": lambda w: portfolio_return(w, mu.values) - target}
    ]
    res = minimize(
        fun=lambda w: portfolio_vol(w, cov.values),
        x0=_x0(n),
        method="SLSQP",
        bounds=_bounds(n, long_only, max_weight),
        constraints=cons,
        options={"ftol": 1e-12, "max": 2000},
    )
    if not res.success:
        return None
    return pd.Series(res.x, index=mu.index, name=f"target_{target:.4f}")