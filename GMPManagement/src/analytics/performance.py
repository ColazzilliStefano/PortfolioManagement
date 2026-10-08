from __future__ import annotations
import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, freq: int = 12,
                      return_type: str = "simple") -> float:
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return np.nan
    if return_type == "log":
        return float(np.exp(r.mean() * freq) - 1)
    total = float((1.0 + r).prod())
    return float(total ** (freq / n) - 1)


def annualized_vol(returns: pd.Series, freq: int = 12) -> float:
    return float(returns.std() * np.sqrt(freq))


def _align_rf(r: pd.Series, rf_series: pd.Series | None) -> pd.Series:
    if rf_series is None:
        return pd.Series(0.0, index=r.index)
    return rf_series.reindex(r.index).fillna(0.0)


def sharpe_ratio(returns: pd.Series, rf_series: pd.Series | None = None,
                 freq: int = 12) -> float:
    r = returns.dropna()
    rf = _align_rf(r, rf_series)
    ex = r - rf
    if ex.std() == 0:
        return np.nan
    return float(np.sqrt(freq) * ex.mean() / ex.std())


def sortino_ratio(returns: pd.Series, rf_series: pd.Series | None = None,
                  freq: int = 12) -> float:
    r = returns.dropna()
    rf = _align_rf(r, rf_series)
    ex = r - rf
    downside = ex[ex < 0].std()
    if downside == 0 or np.isnan(downside):
        return np.nan
    return float(np.sqrt(freq) * ex.mean() / downside)


def max_drawdown(returns: pd.Series, return_type: str = "simple") -> float:
    r = returns.dropna()
    if return_type == "log":
        cum = np.exp(r.cumsum())
    else:
        cum = (1.0 + r).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return float(dd.min())


def calmar_ratio(returns: pd.Series, freq: int = 12,
                 return_type: str = "simple") -> float:
    mdd = max_drawdown(returns, return_type=return_type)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return annualized_return(returns, freq, return_type) / abs(mdd)


def summary(returns: pd.Series, name: str = "portfolio", freq: int = 12,
            rf_series: pd.Series | None = None,
            return_type: str = "simple") -> dict:
    return {
        "name": name,
        "CAGR": annualized_return(returns, freq, return_type),
        "Vol": annualized_vol(returns, freq),
        "Sharpe": sharpe_ratio(returns, rf_series=rf_series, freq=freq),
        "Sortino": sortino_ratio(returns, rf_series=rf_series, freq=freq),
        "MaxDD": max_drawdown(returns, return_type=return_type),
        "Calmar": calmar_ratio(returns, freq, return_type),
    }