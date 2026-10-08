"""Value signal
Value is the tendency of prices to revert toward their historical mean
(mean reversion). CAPE (Shiller) is the reference for the equity market.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.signals.shiller import fetch_shiller_cape


def value_to_view(lookback: int = 120,
                  scale: float = 1.0,
                  max_magnitude: float = 0.05) -> pd.Series:
    cape_df = fetch_shiller_cape()
    if cape_df.empty:
        return pd.Series(dtype=float)

    cape = cape_df["cape"].dropna()
    if len(cape) < lookback:
        logger.warning(f"Value: CAPE has only {len(cape)} observations, "
                       f"at least {lookback} are needed")
        return pd.Series(dtype=float)

    rm = cape.rolling(lookback).mean()
    rs = cape.rolling(lookback).std()
    z = (cape - rm) / rs.replace(0, np.nan)

    mag = -np.tanh(z / scale) * max_magnitude
    mag.name = "value_view"
    return mag.dropna()