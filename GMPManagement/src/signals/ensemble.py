import numpy as np
import pandas as pd
from loguru import logger


def compute_ir_weights(signals_hist: dict[str, pd.Series],
                       date: pd.Timestamp,
                       lookback: int = 60,
                       min_obs: int = 24) -> dict[str, float]:
    ir_values = {}

    for name, series in signals_hist.items():
        if series is None or series.empty:
            continue

        # select history up to date (excluded)
        hist = series.loc[series.index < date].tail(lookback)
        if len(hist) < min_obs:
            continue

        std = hist.std()
        if std <= 0 or np.isnan(std):
            continue

        ir = hist.mean() / std
        if np.isnan(ir):
            continue

        # we use |IR| as weight (are negative signals useful if inverted?
        # no: here each signal already produces the corrected view, so
        # positive IR = predictive signal, negative IR = noise
        ir_values[name] = max(0.0, ir)

    if not ir_values:
        return {}

    total = sum(ir_values.values())
    if total <= 0:
        # all IRs negative: fallback to equal weights
        n = len(ir_values)
        return {k: 1.0 / n for k in ir_values}

    return {k: v / total for k, v in ir_values.items()}


def weighted_ensemble(signals_current: dict[str, float],
                      weights: dict[str, float]) -> float:
    if not weights or not signals_current:
        return 0.0

    total = 0.0
    weight_sum = 0.0
    for name, val in signals_current.items():
        w = weights.get(name, 0.0)
        if w > 0:
            total += w * val
            weight_sum += w

    if weight_sum <= 0:
        return 0.0

    return total / weight_sum