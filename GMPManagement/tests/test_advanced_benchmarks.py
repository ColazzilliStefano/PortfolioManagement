import numpy as np
import pandas as pd
import pytest

from src.benchmarks.risk_parity import risk_parity_weights, risk_parity_backtest
from src.benchmarks.hrp import hrp_weights, hrp_backtest
from src.risk.cvar import cvar_weights, cvar_backtest


# Fixtures
@pytest.fixture
def returns_3x36():
    """3 assets, 36 months."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-31", periods=36, freq="ME")
    return pd.DataFrame({
        "A": rng.normal(0.005, 0.03, 36),
        "B": rng.normal(0.003, 0.015, 36),
        "C": rng.normal(0.004, 0.05, 36),
    }, index=dates)


@pytest.fixture
def returns_5x120():
    """5 assets, 120 months."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2010-01-31", periods=120, freq="ME")
    return pd.DataFrame(
        rng.normal(0.004, 0.03, size=(120, 5)),
        columns=[f"A{i}" for i in range(5)],
        index=dates,
    )


# Risk Parity

def test_risk_parity_weights_sum_to_one(returns_5x120):
    cov = returns_5x120.cov() * 12
    w = risk_parity_weights(cov)
    assert abs(w.sum() - 1.0) < 1e-6


def test_risk_parity_weights_positive(returns_5x120):
    cov = returns_5x120.cov() * 12
    w = risk_parity_weights(cov, long_only=True)
    assert (w >= -1e-9).all()


def test_risk_parity_max_weight_respected(returns_5x120):
    cov = returns_5x120.cov() * 12
    w = risk_parity_weights(cov, long_only=True, max_weight=0.4)
    assert (w <= 0.4 + 1e-6).all()


def test_risk_parity_infeasible_max_weight(returns_5x120):
    """max_weight < 1/n must raise ValueError."""
    cov = returns_5x120.cov() * 12
    with pytest.raises(ValueError, match="infeasible"):
        risk_parity_weights(cov, long_only=True, max_weight=0.1)


def test_risk_parity_backtest_output(returns_5x120):
    port, w_df = risk_parity_backtest(returns_5x120, window=24, rebalance=6)
    assert len(port) > 0
    assert port.index.is_monotonic_increasing
    assert not w_df.empty
    sums = w_df.sum(axis=1)
    assert (sums.round(6) == 1.0).all()


def test_risk_parity_no_lookahead(returns_5x120):
    """Weights at t must be computed only on data < t."""
    port, w_df = risk_parity_backtest(returns_5x120, window=24, rebalance=6)
    # first rebalancing: at least 24 months after the start
    first_date = w_df.index[0]
    offset = returns_5x120.index.get_loc(first_date)
    assert offset >= 24


# HRP

def test_hrp_weights_sum_to_one(returns_5x120):
    cov = returns_5x120.cov() * 12
    corr = returns_5x120.corr()
    w = hrp_weights(cov, corr)
    assert abs(w.sum() - 1.0) < 1e-6


def test_hrp_weights_positive(returns_5x120):
    cov = returns_5x120.cov() * 12
    w = hrp_weights(cov)
    assert (w >= -1e-9).all()


def test_hrp_backtest_output(returns_5x120):
    port, w_df = hrp_backtest(returns_5x120, window=24, rebalance=6)
    assert len(port) > 0
    assert port.index.is_monotonic_increasing

# CVaR

def test_cvar_weights_sum_to_one(returns_5x120):
    w = cvar_weights(returns_5x120, alpha=0.05)
    assert abs(w.sum() - 1.0) < 1e-6


def test_cvar_weights_positive(returns_5x120):
    w = cvar_weights(returns_5x120, long_only=True)
    assert (w >= -1e-9).all()


def test_cvar_max_weight_respected(returns_5x120):
    w = cvar_weights(returns_5x120, max_weight=0.4, long_only=True)
    assert (w <= 0.4 + 1e-6).all()


def test_cvar_infeasible_max_weight(returns_5x120):
    with pytest.raises(ValueError, match="infeasible"):
        cvar_weights(returns_5x120, long_only=True, max_weight=0.05)


def test_cvar_short_sample():
    dates = pd.date_range("2024-01-31", periods=5, freq="ME")
    df = pd.DataFrame({"A": [0.01]*5, "B": [0.005]*5}, index=dates)
    w = cvar_weights(df)
    assert abs(w.sum() - 1.0) < 1e-6


def test_cvar_backtest_output(returns_5x120):
    port, w_df = cvar_backtest(returns_5x120, window=24, rebalance=6)
    assert len(port) > 0

def test_no_nan_in_returns(returns_5x120):
    _, w_rp = risk_parity_backtest(returns_5x120, window=24, rebalance=6)
    _, w_hrp = hrp_backtest(returns_5x120, window=24, rebalance=6)
    _, w_cvar = cvar_backtest(returns_5x120, window=24, rebalance=6)

    for name, w in [("rp", w_rp), ("hrp", w_hrp), ("cvar", w_cvar)]:
        assert not w.isna().any().any(), f"NaN in {name}"
        assert np.isfinite(w.values).all(), f"Inf in {name}"