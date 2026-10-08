"""GMP backtest: aggregates returns at the GMP asset class level."""

from __future__ import annotations
import pandas as pd
from src.analytics.returns import log_to_simple


def _aggregate_to_gmp_classes(returns: pd.DataFrame,
                              ticker_to_gmp: dict) -> pd.DataFrame:
    mapping = {c: ticker_to_gmp.get(c, c) for c in returns.columns}
    grouped = returns.T.groupby(mapping).mean().T
    return grouped


def backtest_gmp(returns: pd.DataFrame, weights_history: pd.DataFrame,
                 ticker_to_gmp: dict, return_type: str = "log") -> pd.Series:
    r = returns.copy()
    w = weights_history.copy()
    r.index = pd.to_datetime(r.index).tz_localize(None).astype("datetime64[ns]")
    w.index = pd.to_datetime(w.index, utc=True).tz_localize(None).astype("datetime64[ns]")
    r_class = _aggregate_to_gmp_classes(r, ticker_to_gmp)   # aggregates returns at the GMP asset class level

    w_monthly = w.reindex(r_class.index, method="ffill") # aligns weights to the return dates
    w_monthly = w_monthly.reindex(columns=r_class.columns).fillna(0.0)

    # computes portfolio return
    r_simple = log_to_simple(r_class) if return_type == "log" else r_class
    port = (r_simple * w_monthly).sum(axis=1)
    port.name = "GMP"
    return port


def compute_turnover(weights_history: pd.DataFrame) -> pd.Series:
    return weights_history.diff().abs().sum(axis=1).fillna(0.0)