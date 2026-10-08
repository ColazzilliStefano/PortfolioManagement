import numpy as np
import pandas as pd

from src.signals import inflation
from src.signals.credit import credit_to_view
from src.signals.vix import vix_to_view


def _monthly_levels(periods=96):
    index = pd.date_range("2015-01-31", periods=periods, freq="ME")
    values = 100 * np.cumprod(1 + np.full(periods, 0.002))
    return pd.Series(values, index=index, name="CPIAUCSL")


def test_cpi_api_returns_yoy_and_view(monkeypatch):
    monkeypatch.setattr(inflation, "fetch_fred", lambda *args, **kwargs: _monthly_levels())
    cpi = inflation.cpi_yoy(start="2015-01-01")
    view = inflation.inflation_to_view(cpi, lookback=12)
    assert cpi.name == "cpi_yoy"
    assert not cpi.empty
    assert view.name == "inflation_view"


def test_high_inflation_is_positive_inflation_view():
    cpi = pd.Series(
        np.r_[np.full(71, 2.0), 6.0],
        index=pd.date_range("2015-01-31", periods=72, freq="ME"),
    )
    view = inflation.inflation_to_view(cpi, lookback=12)
    assert view.iloc[-1] > 0


def test_vix_backwardation_is_negative_view():
    index = pd.date_range("2015-01-31", periods=72, freq="ME")
    ratio = np.r_[np.full(71, 1.05), 0.75]
    vix = pd.DataFrame(
        {"vix_short": 20.0, "vix_mid": 20.0, "vix_long": ratio * 20.0},
        index=index,
    )
    view = vix_to_view(vix, lookback=12)
    assert view.iloc[-1] < 0
    assert view.abs().max() <= 0.05 + 1e-12