import numpy as np
import pandas as pd

from src.signals.regimes import (
    fit_hmm, smooth_states, regime_probs, regime_to_view,
)


def _make_data(seed=42, T=120, N=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 0.03, size=(T, N)),
        columns=["equity", "bonds", "gold"],
        index=pd.date_range("2010-01-31", periods=T, freq="ME"),
    )


def test_fit_hmm_multivariate():
    df = _make_data()
    model = fit_hmm(df, n_states=3, n_iter=50)
    assert model.n_components == 3
    assert model.n_features == 3


def test_smooth_states_multivariate():
    df = _make_data()
    model = fit_hmm(df, n_states=3, n_iter=50)
    probs = smooth_states(model, df)
    assert probs.shape[0] == len(df)
    assert probs.shape[1] == 3
    # probabilities sum to 1 for each row
    sums = probs.sum(axis=1)
    np.testing.assert_allclose(sums.values, np.ones(len(df)), atol=1e-10)


def test_regime_probs_returns_three_labels():
    df = _make_data()
    rp = regime_probs(df, n_states=3, seed=42)
    assert set(rp.columns) == {"bull", "transition", "bear"}
    assert len(rp) == len(df)


def test_regime_to_view():
    df = _make_data()
    rp = regime_probs(df, n_states=3, seed=42)
    view = regime_to_view(rp, max_magnitude=0.06)
    assert len(view) == len(rp)
    assert (view.abs() <= 0.06 + 1e-10).all()