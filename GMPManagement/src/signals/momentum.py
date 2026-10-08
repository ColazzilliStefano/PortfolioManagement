import numpy as np
import pandas as pd


def momentum_signal(returns_class: pd.DataFrame, asset: str = "equity",
                    lookback: int = 6, freq: int = 12) -> pd.Series:
    r = returns_class[asset]
    cum = (1 + r).rolling(lookback).apply(lambda x: x.prod() - 1, raw=True)
    # annualize
    return ((1 + cum) ** (freq / lookback) - 1).shift(1)


def momentum_to_view(momentum: pd.Series, max_magnitude: float = 0.06,
                     scale: float = 0.10) -> pd.Series:
    """
    Converts momentum into the magnitude of the growth view.
    - momentum > 0: positive view (growth)
    - momentum < 0: negative view (recession)
    - magnitude: tanh(momentum/scale) * max_magnitude
    """
    return np.tanh(momentum / scale) * max_magnitude