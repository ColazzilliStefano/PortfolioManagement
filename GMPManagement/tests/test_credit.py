import numpy as np
import pandas as pd
import pytest

from src.signals.credit import (
    credit_spread_proxy,
    credit_to_view,
    credit_spread_from_fred,
)


@pytest.fixture(scope="module")
def returns_monthly():
    """Monthly returns from the project dataset."""
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "data/processed/returns_monthly.parquet"
    if not p.exists():
        pytest.skip("returns_monthly.parquet not present")
    return pd.read_parquet(p)


def test_credit_spread_proxy_shape(returns_monthly):
    """The proxy has a reasonable number of observations."""
    spread = credit_spread_proxy(returns_monthly, lookback=6)
    assert len(spread) > 150, \
        f"Spread proxy too short: {len(spread)} obs"
    assert spread.index.is_monotonic_increasing


def test_credit_spread_proxy_range(returns_monthly):
    """The spread proxy is in a plausible range (return differential)."""
    spread = credit_spread_proxy(returns_monthly, lookback=6)
    # 6m cumulative differential between HYG and LQD: typically in [-0.25, +0.25]
    assert spread.min() > -0.50, f"Min too low: {spread.min()}"
    assert spread.max() < 0.50, f"Max too high: {spread.max()}"


def test_credit_to_view_bounds(returns_monthly):
    """The view is bounded in [-max_magnitude, +max_magnitude]."""
    max_mag = 0.06
    view = credit_to_view(returns_monthly, max_magnitude=max_mag)
    assert len(view) > 100, f"View too short: {len(view)} obs"
    assert view.min() >= -max_mag - 1e-9, f"Min out of bounds: {view.min()}"
    assert view.max() <= max_mag + 1e-9, f"Max out of bounds: {view.max()}"


def test_credit_to_view_not_saturated(returns_monthly):
    """The view is not saturated (reasonable std, no binary distribution)."""
    view = credit_to_view(returns_monthly)
    std = view.std()
    assert 0.02 < std < 0.05, f"Anomalous std: {std:.4f}"

    # saturation: % of |view| > 0.95 * max_magnitude
    saturated = (view.abs() > 0.057).sum() / len(view)
    assert saturated < 0.10, \
        f"Too much saturation: {saturated:.1%} of values at the limit"


def test_credit_to_view_directional():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-31", periods=36, freq="ME")

    # HYG: positive drift with noise
    hyg = 0.01 + rng.normal(0, 0.005, size=36)
    # LQD: zero drift with noise
    lqd = 0.00 + rng.normal(0, 0.003, size=36)

    df = pd.DataFrame({
        "HYG": hyg,
        "LQD": lqd,
        "ACWI": rng.normal(0, 0.02, size=36),
    }, index=dates)

    view = credit_to_view(df, lookback_signal=6, lookback_zscore=12)

    assert not view.empty, "Empty view: problem in the rolling"
    assert view.mean() > 0, \
        f"Mean view not positive with HYG outperformer: {view.mean():.4f}"

    positive_frac = (view > 0).mean()
    assert positive_frac > 0.5, \
        f"Only {positive_frac:.1%} of positive views with HYG outperformer"


def test_credit_to_view_empty_input():
    """Input without HYG or LQD returns an empty Series, not an exception."""
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    df = pd.DataFrame({"ACWI": [0.01] * 12}, index=dates)

    spread = credit_spread_proxy(df, lookback=6)
    assert spread.empty

    view = credit_to_view(df)
    assert view.empty


def test_credit_fred_fallback_returns_series():
    """The FRED fallback returns a Series (possibly empty)."""
    s = credit_spread_from_fred(start="2023-09-01")
    assert isinstance(s, pd.Series)
    if not s.empty:
        assert s.index.is_monotonic_increasing
        assert s.name == "credit_spread_fred"