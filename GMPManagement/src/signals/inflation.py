"""Inflation signals from FRED/ALFRED."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.fred_loader import fetch_fred, resample_monthly


CPI_SERIES = "CPIAUCSL"


def cpi_yoy(start: str = "2000-01-01") -> pd.Series:
    cpi = resample_monthly(fetch_fred(CPI_SERIES, start=start))
    return cpi.pct_change(12).mul(100).rename("cpi_yoy").dropna()


def inflation_to_view(cpi: pd.Series, lookback: int = 60,
                      scale: float = 1.0,
                      max_magnitude: float = 0.04) -> pd.Series:
    if cpi.empty:
        return pd.Series(dtype=float, name="inflation_view")
    mean = cpi.rolling(lookback).mean()
    std = cpi.rolling(lookback).std().replace(0, np.nan)
    z = (cpi - mean) / std
    return (np.tanh(z / scale) * max_magnitude).rename("inflation_view").dropna()


def cpi_yoy_vintage(vintage_date: str, start: str = "2000-01-01",
                    lookback: int = 60, scale: float = 1.0,
                    max_magnitude: float = 0.04) -> float | None:
    cpi = resample_monthly(fetch_fred(
        CPI_SERIES, start=start, vintage_date=vintage_date,
    ))
    cpi_yoy_series = cpi.pct_change(12).mul(100).dropna()
    available = cpi_yoy_series.loc[:pd.Timestamp(vintage_date)]
    view = inflation_to_view(available, lookback=lookback,
                             scale=scale, max_magnitude=max_magnitude)
    if view.empty:
        return None
    return float(view.iloc[-1])


def term_premium_view(start: str = "2000-01-01",
                      lookback: int = 60,
                      scale: float = 1.0,
                      max_magnitude: float = 0.04) -> pd.Series:
    try:
        y10 = resample_monthly(fetch_fred("DGS10", start=start))
        y3m = resample_monthly(fetch_fred("DTB3", start=start))
        spread = (y10 - y3m).dropna()
        spread.name = "term_premium"
    except Exception as e:
        from loguru import logger
        logger.warning(f"Term premium unavailable: {e}")
        return pd.Series(dtype=float)

    rolling_mean = spread.rolling(lookback).mean()
    rolling_std = spread.rolling(lookback).std()
    z = (spread - rolling_mean) / rolling_std.replace(0, np.nan)
    mag = np.tanh(z / scale) * max_magnitude
    mag.name = "term_premium_view"
    return mag.dropna()