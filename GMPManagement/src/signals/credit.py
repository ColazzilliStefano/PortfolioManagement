"""Credit spread signal (proxy from HYG vs LQD ETFs).

Note: the ICE BofA series on FRED (BAMLH0A0HYM2, etc.) are limited to 795
observations (3 years) via API due to license restrictions. It is not possible
to get the full history for free.

Alternative approach: we use the relative performance HYG vs LQD as a
proxy for the credit spread. The two ETFs have similar duration (~4-5 years), so
the return differential captures the movement of the HY-IG spread.

Interpretation:
- HYG outperforms LQD -> spread tightens -> risk-on -> positive growth view
- HYG underperforms LQD -> spread widens -> risk-off -> negative growth view
"""

import numpy as np
import pandas as pd
from loguru import logger


def credit_spread_proxy(returns_monthly: pd.DataFrame,
                        lookback: int = 6) -> pd.Series:
    if "HYG" not in returns_monthly.columns or "LQD" not in returns_monthly.columns:
        logger.warning("HYG or LQD not present in the DataFrame")
        return pd.Series(dtype=float)

    hyg = returns_monthly["HYG"]
    lqd = returns_monthly["LQD"]

    # cumulative return over lookback months
    hyg_cum = (1 + hyg).rolling(lookback).apply(np.prod, raw=True) - 1
    lqd_cum = (1 + lqd).rolling(lookback).apply(np.prod, raw=True) - 1

    spread = hyg_cum - lqd_cum
    spread.name = "credit_spread_proxy"
    return spread.dropna()


def credit_to_view(returns_monthly: pd.DataFrame,
                   lookback_signal: int = 6,
                   lookback_zscore: int = 60,
                   scale: float = 1.0,
                   max_magnitude: float = 0.06) -> pd.Series:
    spread = credit_spread_proxy(returns_monthly, lookback=lookback_signal)
    if spread.empty:
        return pd.Series(dtype=float)

    rolling_mean = spread.rolling(lookback_zscore).mean()
    rolling_std = spread.rolling(lookback_zscore).std()

    z = (spread - rolling_mean) / rolling_std.replace(0, np.nan)
    mag = np.tanh(z / scale) * max_magnitude
    mag.name = "credit_view"
    return mag.dropna()


def credit_spread_from_fred(series_id: str = "BAMLH0A0HYM2",
                            start: str = "2023-09-01") -> pd.Series:
    from src.signals.fred_loader import fetch_fred, resample_monthly
    try:
        s = fetch_fred(series_id, start=start)
        s = resample_monthly(s)
        s.name = "credit_spread_fred"
        return s.dropna()
    except Exception as e:
        logger.warning(f"FRED credit spread not available: {e}")
        return pd.Series(dtype=float)