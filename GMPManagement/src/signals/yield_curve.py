"""Yield curve signal from FRED (10y - 2y)."""

import pandas as pd

from src.signals.fred_loader import fetch_fred, resample_monthly


def yield_spread(start: str = "2000-01-01") -> pd.Series:
    """Spread 10y - 2y (in percentage points)."""
    y10 = resample_monthly(fetch_fred("DGS10", start=start))
    y2 = resample_monthly(fetch_fred("DGS2", start=start))
    spread = (y10 - y2).dropna()
    spread.name = "yield_spread"
    return spread


def yield_to_view(spread: pd.Series,
                  threshold: float = 0.0,
                  scale: float = 1.0,
                  max_magnitude: float = 0.05) -> pd.Series:
    """
    Converts the yield spread into the magnitude of the recession view.
    - spread < 0 (inverted): positive recession view
    - spread > 0: negative recession view (growth)
    """
    import numpy as np
    inverted = -(spread - threshold)
    return np.tanh(inverted / scale) * max_magnitude