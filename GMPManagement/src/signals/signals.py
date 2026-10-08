"""Signal engine: centralized factory of signals.
The module reads the active signals from config/config.yaml.
"""

import pandas as pd

from src.utils.config import get_signal_params


# Index normalization

def normalize_index(s: pd.Series) -> pd.Series:
    """Normalizes index to datetime64[ns] tz-naive."""
    out = s.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.astype("datetime64[ns]")
    return out


# Function factory

def _signal_momentum(returns_monthly, **params):
    """Equity momentum 6m. Handled internally by bl_walk_forward, returns empty."""
    return pd.Series(dtype=float)


def _signal_hmm(returns_monthly, **params):
    """HMM regime. Handled internally by bl_walk_forward, returns empty."""
    return pd.Series(dtype=float)


def _signal_credit(returns_monthly, **params):
    from src.signals.credit import credit_to_view
    return credit_to_view(
        returns_monthly,
        lookback_signal=params.get("lookback_signal", 6),
        lookback_zscore=params.get("lookback_zscore", 60),
        scale=params.get("scale", 1.0),
        max_magnitude=params.get("max_magnitude", 0.06),
    )


def _signal_vix(returns_monthly, **params):
    from src.signals.vix import fetch_vix_term_structure, vix_to_view
    vix = fetch_vix_term_structure(start=params.get("start", "2011-01-01"))
    if vix.empty:
        return pd.Series(dtype=float)
    return vix_to_view(
        vix,
        lookback=params.get("lookback", 60),
        scale=params.get("scale", 1.0),
        max_magnitude=params.get("max_magnitude", 0.05),
    )


def _signal_term_premium(returns_monthly, **params):
    from src.signals.inflation import term_premium_view
    return term_premium_view(
        start=params.get("start", "2000-01-01"),
        lookback=params.get("lookback", 60),
        scale=params.get("scale", 1.0),
        max_magnitude=params.get("max_magnitude", 0.04),
    )


def _signal_carry(returns_monthly, **params):
    from src.signals.carry import carry_to_view
    return carry_to_view(
        start=params.get("start", "2000-01-01"),
        lookback=params.get("lookback", 60),
        scale=params.get("scale", 1.0),
        max_magnitude=params.get("max_magnitude", 0.05),
    )


def _signal_trend(returns_monthly, **params):
    from src.signals.trend import trend_to_view
    asset = params.get("asset", "ACWI")
    if asset not in returns_monthly.columns:
        return pd.Series(dtype=float)
    return trend_to_view(
        returns_monthly,
        asset=asset,
        lookback=params.get("lookback", 12),
        scale=params.get("scale", 0.15),
        max_magnitude=params.get("max_magnitude", 0.05),
    )


def _signal_value(returns_monthly, **params):
    from src.signals.value import value_to_view
    return value_to_view(
        lookback=params.get("lookback", 120),
        scale=params.get("scale", 1.0),
        max_magnitude=params.get("max_magnitude", 0.05),
    )


# Map name -> function
SIGNAL_FACTORY = {
    "momentum": _signal_momentum,
    "hmm": _signal_hmm,
    "credit": _signal_credit,
    "vix": _signal_vix,
    "term_premium": _signal_term_premium,
    "carry": _signal_carry,
    "trend": _signal_trend,
    "value": _signal_value,
}


# Signals handled internally by bl_walk_forward (not external)
INTERNAL_SIGNALS = {"momentum", "hmm"}


# Builder

def build_external_signals(returns_monthly: pd.DataFrame,
                           active_signals: list[str]) -> dict:
    signals = {}

    for name in active_signals:
        # momentum and hmm are handled internally by bl_walk_forward
        if name in INTERNAL_SIGNALS:
            continue

        factory = SIGNAL_FACTORY.get(name)
        if factory is None:
            print(f"Unknown signal: {name}")
            continue

        params = get_signal_params(name)
        try:
            s = factory(returns_monthly, **params)
            if s is not None and not s.empty:
                signals[name] = normalize_index(s)
        except Exception as e:
            print(f"Signal {name} unavailable: {e}")

    return signals