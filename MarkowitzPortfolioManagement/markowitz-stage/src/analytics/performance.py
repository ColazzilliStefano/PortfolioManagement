import numpy as np
import pandas as pd


def _align_rf(returns: pd.Series, rf_series: pd.Series | None) -> pd.Series:
    if rf_series is None:
        return pd.Series(0.0, index=returns.index)
    return rf_series.reindex(returns.index).fillna(0.0)

def annualized_return(returns: pd.Series, freq: int = 12) -> float:
    """CAGR from monthly simple returns."""
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    return float((1.0 + r).prod() ** (freq / len(r)) - 1)

def annualized_vol(returns: pd.Series, freq: int = 12) -> float:
    return float(returns.std() * np.sqrt(freq))

def sharpe_ratio(returns: pd.Series, rf_series: pd.Series | None = None,
                 freq: int = 12) -> float:
    r = returns.dropna()
    ex = r - _align_rf(r, rf_series)
    if ex.std() == 0:
        return np.nan
    return float(np.sqrt(freq) * ex.mean() / ex.std())

def sortino_ratio(returns: pd.Series, rf_series: pd.Series | None = None,
                  freq: int = 12) -> float:
    r = returns.dropna()
    ex = r - _align_rf(r, rf_series)
    downside = ex[ex < 0].std()
    if downside == 0 or np.isnan(downside):
        return np.nan
    return float(np.sqrt(freq) * ex.mean() / downside)

def max_drawdown(returns: pd.Series) -> float:
    cum = (1.0 + returns.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    return float(dd.min())

def calmar_ratio(returns: pd.Series, freq: int = 12) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return annualized_return(returns, freq) / abs(mdd)

def summary(returns: pd.Series, name: str = "portfolio", freq: int = 12,
            rf_series: pd.Series | None = None,
            rf: float | None = None) -> dict:
    if rf_series is None and rf is not None:
        rf_series = pd.Series(rf / freq, index=returns.index)

    return {
        "name": name,
        "CAGR": annualized_return(returns, freq),
        "Vol": annualized_vol(returns, freq),
        "Sharpe": sharpe_ratio(returns, rf_series=rf_series, freq=freq),
        "Sortino": sortino_ratio(returns, rf_series=rf_series, freq=freq),
        "MaxDD": max_drawdown(returns),
        "Calmar": calmar_ratio(returns, freq),
    }