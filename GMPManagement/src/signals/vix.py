"""VIX term structure signal.

Source: yfinance (^VIX, ^VIX3M, ^VIX9D).
The VIX term structure is the ratio between short-term VIX and 3-month VIX:
- Contango (VIX < VIX3M) -> calm market -> positive growth view
- Backwardation (VIX > VIX3M) -> short-term stress -> recession view

The VIX/VIX3M ratio is a leading indicator of risk-on/risk-off.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from pathlib import Path


CACHE_DIR = Path("data/raw/macro")
VIX_TICKERS = {
    "vix_short": "^VIX9D",   # 9 days
    "vix_mid": "^VIX",       # 30 days
    "vix_long": "^VIX3M",    # 3 months
}


def fetch_vix_term_structure(start: str = "2011-01-01") -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "vix_term_structure.parquet"

    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        try:
            raw = yf.download(
                list(VIX_TICKERS.values()),
                start=start,
                auto_adjust=True,
                progress=False,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw["Close"]
            else:
                df = raw[["Close"]]
            df.columns = list(VIX_TICKERS.keys())
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            df = df.sort_index().ffill()
            df.to_parquet(cache)
            logger.info(f"VIX term structure downloaded: {df.shape}")
        except Exception as e:
            logger.warning(f"VIX unavailable: {e}")
            return pd.DataFrame()

    return df.loc[start:].dropna()


def vix_to_view(vix_df: pd.DataFrame,
                lookback: int = 60,
                scale: float = 1.0,
                max_magnitude: float = 0.05) -> pd.Series:
    if vix_df.empty:
        return pd.Series(dtype=float)

    ratio = vix_df["vix_long"] / vix_df["vix_mid"].replace(0, np.nan)

    rolling_mean = ratio.rolling(lookback).mean()
    rolling_std = ratio.rolling(lookback).std()
    z = (ratio - rolling_mean) / rolling_std.replace(0, np.nan)

    mag = np.tanh(z / scale) * max_magnitude
    mag.name = "vix_view"

    mag_m = mag.resample("ME").last().ffill()
    return mag_m.dropna()