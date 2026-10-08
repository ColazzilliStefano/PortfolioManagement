"""Carry signal (equity risk premium).

Carry is the expected return if prices do not change.
For a multi-asset portfolio: equity risk premium (ERP) = earnings yield
of the equity market - yield of government bonds.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.signals.shiller import fetch_shiller_cape
from src.signals.fred_loader import fetch_fred, resample_monthly


def compute_erp(start: str = "2000-01-01") -> pd.Series:
    """Monthly Equity Risk Premium: earnings yield - 10y Treasury yield."""
    cape_df = fetch_shiller_cape()
    if cape_df.empty:
        logger.warning("Carry: CAPE not available")
        return pd.Series(dtype=float)

    try:
        y10 = resample_monthly(fetch_fred("DGS10", start=start))
    except Exception as e:
        logger.warning(f"Carry: DGS10 unavailable ({e})")
        return pd.Series(dtype=float)

    ey = cape_df["earnings_yield"]
    common = ey.index.intersection(y10.index)
    erp = (ey.loc[common] * 100 - y10.loc[common])  # percentage points
    erp.name = "erp"
    return erp.dropna()


def carry_to_view(start: str = "2000-01-01",
                  lookback: int = 60,
                  scale: float = 1.0,
                  max_magnitude: float = 0.05) -> pd.Series:
    erp = compute_erp(start=start)
    if erp.empty:
        return pd.Series(dtype=float)

    rm = erp.rolling(lookback).mean()
    rs = erp.rolling(lookback).std()
    z = (erp - rm) / rs.replace(0, np.nan)

    mag = np.tanh(z / scale) * max_magnitude
    mag.name = "carry_view"
    return mag.dropna()