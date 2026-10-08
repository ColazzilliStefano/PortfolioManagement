import numpy as np
import pandas as pd

from src.estimators.shrinkage_cov import (
    sample_cov, ledoit_wolf_shrinkage, constant_correlation_shrinkage,
)


def _make_returns(seed=42, T=120, N=5):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 0.03, size=(T, N)),
        columns=[f"a{i}" for i in range(N)],
        index=pd.date_range("2010-01-31", periods=T, freq="ME"),
    )


def test_sample_cov_symmetric():
    r = _make_returns()
    cov = sample_cov(r, freq=12)
    np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-12)


def test_ledoit_wolf_psd():
    r = _make_returns()
    cov = ledoit_wolf_shrinkage(r, freq=12)
    eigvals = np.linalg.eigvalsh(cov.values)
    assert (eigvals >= -1e-10).all()


def test_constant_corr_psd():
    r = _make_returns()
    cov = constant_correlation_shrinkage(r, freq=12)
    eigvals = np.linalg.eigvalsh(cov.values)
    assert (eigvals >= -1e-10).all()


def test_shrinkage_diagonal_close_to_sample():
    r = _make_returns(T=500, N=3)
    s = sample_cov(r, freq=12).values
    lw = ledoit_wolf_shrinkage(r, freq=12).values
    np.testing.assert_allclose(np.diag(s), np.diag(lw), rtol=0.1)