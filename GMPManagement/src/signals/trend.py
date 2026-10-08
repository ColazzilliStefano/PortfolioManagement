"""Trend following signal (12m momentum).

Trend following is documented as one of the strategies with the best
information ratio (Moskowitz, Ooi, Pedersen 2012: IR 0.3-0.5).

Signal: 12-month cumulative return on global equity.
- Positive trend -> positive growth view
- Negative trend -> negative growth view

Difference compared to 6m momentum (already in use):
- 6m captures short-term reversal/reaction
- 12m captures the structural trend (more persistent)
"""

import numpy as np
import pandas as pd


def trend_12m_signal(returns_class: pd.DataFrame,
                     asset: str = "equity",
                     lookback: int = 12) -> pd.Series:
    if asset not in returns_class.columns:
        return pd.Series(dtype=float)
    r = returns_class[asset]
    cum = (1 + r).rolling(lookback).apply(np.prod, raw=True) - 1
    cum.name = "trend_12m"
    return cum.dropna()


def trend_to_view(returns_class: pd.DataFrame,
                  asset: str = "equity",
                  lookback: int = 12,
                  scale: float = 0.15,
                  max_magnitude: float = 0.05) -> pd.Series:
    trend = trend_12m_signal(returns_class, asset=asset, lookback=lookback)
    if trend.empty:
        return pd.Series(dtype=float)

    mag = np.tanh(trend / scale) * max_magnitude
    mag.name = "trend_view"
    return mag.dropna()