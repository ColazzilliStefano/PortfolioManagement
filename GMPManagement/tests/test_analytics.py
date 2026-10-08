"""Tests for return aggregation and performance metrics."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.returns import log_to_simple, portfolio_returns_from_weights
from src.analytics.performance import (
    annualized_return, max_drawdown, sharpe_ratio,
)


def test_log_to_simple_inverse():
    log_r = np.array([0.01, -0.02, 0.03])
    simple = log_to_simple(log_r)
    back = np.log1p(simple)
    np.testing.assert_allclose(back, log_r, atol=1e-12)


def test_portfolio_aggregation_log_returns():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    log_returns = pd.DataFrame({
        "A": [0.02, -0.01, 0.03],
        "B": [0.005, 0.008, -0.002],
    }, index=dates)
    w = pd.Series({"A": 0.6, "B": 0.4})

    port = portfolio_returns_from_weights(log_returns, w, return_type="log")
    # manual check
    simple = np.expm1(log_returns)
    expected_simple = (simple * w).sum(axis=1)
    expected_log = np.log1p(expected_simple)
    np.testing.assert_allclose(port.values, expected_log.values, atol=1e-12)


def test_max_drawdown_simple():
    r = pd.Series([0.05, -0.10, 0.02])
    dd = max_drawdown(r, return_type="simple")
    assert dd < -0.05
    assert dd > -0.15


def test_annualized_return_zero():
    r = pd.Series([0.0] * 12)
    assert abs(annualized_return(r, freq=12)) < 1e-10


def test_sharpe_zero_vol():
    r = pd.Series([0.0] * 12)
    assert np.isnan(sharpe_ratio(r)) or sharpe_ratio(r) == 0