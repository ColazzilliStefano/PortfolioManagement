"""Reverse optimization: implied equilibrium returns.

Pi = lambda * Sigma * w_mkt
where:
  w_mkt = GMP weights (asset class)
  Sigma = covariance (shrinkage)
  lambda = risk aversion, calibrated on the market Sharpe
"""

import numpy as np
import pandas as pd


def calibrate_risk_aversion(cov: pd.DataFrame, weights: pd.Series,
                            market_sharpe: float = 0.4) -> float:
    """
    Calibrates lambda using the expected Sharpe of the market portfolio.
    lambda = expected_Sharpe / sigma_p
    """
    w = weights.values
    sigma_p = float(np.sqrt(w @ cov.values @ w))
    if sigma_p <= 0:
        return 2.0
    return market_sharpe / sigma_p


def implied_returns(cov: pd.DataFrame, weights: pd.Series,
                    risk_aversion: float) -> pd.Series:
    """
    Pi = lambda * Sigma * w
    """
    w = weights.reindex(cov.columns).fillna(0.0).values
    pi = risk_aversion * cov.values @ w
    return pd.Series(pi, index=cov.columns, name="pi")


def implied_returns_from_market(returns_class: pd.DataFrame,
                                weights_history: pd.DataFrame,
                                cov_method: str = "ledoit_wolf",
                                market_sharpe: float = 0.4,
                                freq: int = 12) -> pd.Series:
    """
    Computes the implied returns using:
    - shrinkage covariance on the asset class returns
    - average GMP weights (last year)
    """
    from src.estimators.shrinkage_cov import shrinkage_cov

    cov = shrinkage_cov(returns_class, method=cov_method, freq=freq)

    # GMP weights: mean of the last available year
    w_recent = weights_history.tail(2).mean()
    w_recent = w_recent.reindex(cov.columns).fillna(0.0)
    w_recent = w_recent / w_recent.sum()

    lam = calibrate_risk_aversion(cov, w_recent, market_sharpe)
    pi = implied_returns(cov, w_recent, lam)

    return pi, lam, cov, w_recent