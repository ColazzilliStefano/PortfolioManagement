"""View engine: combines signals into (P, Q, Omega) for Black-Litterman."""

import numpy as np
import pandas as pd
from loguru import logger

from src.signals.momentum import momentum_signal, momentum_to_view
from src.signals.inflation import cpi_yoy, inflation_to_view
from src.signals.yield_curve import yield_spread, yield_to_view
from src.signals.regimes import regime_probs, regime_to_view
from src.black_litterman.views import build_views


def build_view_magnitudes(returns_class: pd.DataFrame,
                          date: pd.Timestamp,
                          max_view_growth: float = 0.06,
                          max_view_inflation: float = 0.04,
                          max_view_recession: float = 0.05) -> dict:
    result = {}

    # equity momentum
    mom = momentum_signal(returns_class, asset="equity", lookback=6)
    mom_series = momentum_to_view(mom, max_magnitude=max_view_growth)
    if date in mom_series.index:
        result["growth"] = float(mom_series.loc[date])
    else:
        result["growth"] = 0.0

    # inflation
    try:
        cpi = cpi_yoy()
        infl_series = inflation_to_view(cpi, max_magnitude=max_view_inflation)
        infl_series = infl_series.reindex(returns_class.index, method="ffill")
        if date in infl_series.index:
            result["inflation_up"] = float(infl_series.loc[date])
        else:
            result["inflation_up"] = 0.0
    except Exception as e:
        logger.warning(f"Inflation unavailable: {e}")
        result["inflation_up"] = 0.0

    # yield curve
    try:
        yc = yield_spread()
        yc_series = yield_to_view(yc, max_magnitude=max_view_recession)
        yc_series = yc_series.reindex(returns_class.index, method="ffill")
        if date in yc_series.index:
            result["recession"] = float(yc_series.loc[date])
        else:
            result["recession"] = 0.0
    except Exception as e:
        logger.warning(f"Yield curve unavailable: {e}")
        result["recession"] = 0.0

    # HMM regime
    try:
        regime = regime_probs(returns_class["equity"], n_states=3)
        regime_series = regime_to_view(regime, max_magnitude=max_view_growth)
        regime_series = regime_series.reindex(returns_class.index, method="ffill")
        if date in regime_series.index:
            result["growth"] = 0.5 * result.get("growth", 0.0) + \
                               0.5 * float(regime_series.loc[date])
    except Exception as e:
        logger.warning(f"HMM unavailable: {e}")

    return result


def build_view_matrices(classes: list[str],
                        magnitudes: dict) -> tuple[np.ndarray, np.ndarray]:

    P_rows, Q_vals = [], []
    for scenario, mag in magnitudes.items():
        if abs(mag) < 1e-4:
            continue
        P, Q = build_views(classes, scenario, magnitude=mag)
        P_rows.append(P[0])
        Q_vals.append(Q[0])

    if not P_rows:
        return np.zeros((0, len(classes))), np.zeros(0)

    return np.array(P_rows), np.array(Q_vals)


def build_omega(P: np.ndarray, Q: np.ndarray,
                cov: pd.DataFrame, tau: float = 0.05,
                uncertainty_scale: float = 1.0) -> np.ndarray | None:
    if P.shape[0] == 0:
        return None
    omega = tau * P @ cov.values @ P.T * uncertainty_scale
    return omega