import numpy as np
import pandas as pd
import pytest
from src.black_litterman.bl_model import bl_posterior, optimal_weights_bl


@pytest.fixture
def setup_data():
    pi = pd.Series([0.05, 0.02, 0.04], index=["equity", "bonds", "gold"])
    cov = pd.DataFrame(
        np.diag([0.03, 0.005, 0.025]),
        index=pi.index, columns=pi.index,
    )
    return pi, cov


def test_bl_no_view_returns_prior(setup_data):
    pi, cov = setup_data
    P = np.zeros((0, 3))
    Q = np.zeros(0)
    mu_post, _ = bl_posterior(pi, cov, P, Q, tau=0.05)
    np.testing.assert_allclose(mu_post.values, pi.values, atol=1e-10)


def test_bl_one_view_shifts_mu(setup_data):
    pi, cov = setup_data
    P = np.array([[-1.0, 1.0, 0.0]])
    Q = np.array([0.04])
    mu_post, _ = bl_posterior(pi, cov, P, Q, tau=0.05)
    assert mu_post["equity"] < pi["equity"]
    assert mu_post["bonds"] > pi["bonds"]


def test_bl_singular_omega_does_not_crash(setup_data):
    pi, cov = setup_data
    P = np.array([
        [-1.0, 1.0, 0.0],   # equity - bonds
        [1.0, -1.0, 0.0],   # bonds - equity
    ])
    Q = np.array([0.04, -0.04])
    mu_post, _ = bl_posterior(pi, cov, P, Q, tau=0.05)
    assert len(mu_post) == 3
    assert not np.isnan(mu_post).any()


def test_optimal_weights_sum_to_one(setup_data):
    pi, cov = setup_data
    mu_post, _ = bl_posterior(pi, cov,
                              np.array([[1.0, -1.0, 0.0]]),
                              np.array([0.01]),
                              tau=0.05)
    w = optimal_weights_bl(mu_post, cov, risk_aversion=4.0, max_weight=0.7)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= 0).all()
    assert (w <= 0.7 + 1e-6).all()