"""Covariance shrinkage utilities for the Markowitz model."""

from __future__ import annotations

import pandas as pd
from sklearn.covariance import LedoitWolf


def ledoit_wolf_covariance(returns: pd.DataFrame, freq: int = 12):
    """Return annualized Ledoit-Wolf covariance and shrinkage intensity."""
    estimator = LedoitWolf().fit(returns.dropna().values)
    covariance = pd.DataFrame(
        estimator.covariance_ * freq,
        index=returns.columns,
        columns=returns.columns,
    )
    return covariance, float(estimator.shrinkage_)