"""Black-Litterman model.

Prior:    r ~ N(Pi, tau * Sigma)
View:     P r = Q + eps, eps ~ N(0, Omega)
Posterior:
  E[r]   = [(tau Sigma)^-1 + P' Omega^-1 P]^-1
           * [(tau Sigma)^-1 Pi + P' Omega^-1 Q]
  Cov[r] = [(tau Sigma)^-1 + P' Omega^-1 P]^-1

The model uses pinv (pseudo-inverse) and regularization to handle
collinear views or singular Omega.
"""

import numpy as np
import pandas as pd


def bl_posterior(pi: pd.Series, cov: pd.DataFrame,
                 P: np.ndarray, Q: np.ndarray,
                 omega: np.ndarray | None = None,
                 tau: float = 0.05,
                 regularize: float = 1e-8):
    n = len(pi)
    Sigma = cov.values
    pi_v = pi.values

    # no views: return the prior
    if P.shape[0] == 0:
        return (
            pd.Series(pi_v, index=pi.index, name="mu_bl"),
            pd.DataFrame(Sigma, index=cov.index, columns=cov.columns),
        )

    # Omega: default tau * P Sigma P'
    if omega is None:
        omega = tau * P @ Sigma @ P.T

    # ensure 2D and add regularization
    omega = np.atleast_2d(omega)
    omega = omega + regularize * np.eye(omega.shape[0])

    # tau * Sigma with regularization
    tau_Sigma = tau * Sigma + regularize * np.eye(n)

    # pseudo-inverse for numerical robustness
    tau_Sigma_inv = np.linalg.pinv(tau_Sigma)
    omega_inv = np.linalg.pinv(omega)

    A = tau_Sigma_inv + P.T @ omega_inv @ P
    b = tau_Sigma_inv @ pi_v + P.T @ omega_inv @ Q

    # solve with pinv to handle possibly singular A
    A_inv = np.linalg.pinv(A)
    mu_post = A_inv @ b
    cov_post = A_inv

    return (
        pd.Series(mu_post, index=pi.index, name="mu_bl"),
        pd.DataFrame(cov_post, index=cov.index, columns=cov.columns),
    )


def optimal_weights_bl(mu_bl: pd.Series, cov: pd.DataFrame,
                       risk_aversion: float,
                       long_only: bool = True,
                       max_weight: float = 0.5) -> pd.Series:
    from scipy.optimize import minimize

    n = len(mu_bl)
    mu_v = mu_bl.values
    cov_v = cov.values

    def neg_utility(w):
        return -(w @ mu_v) + 0.5 * risk_aversion * (w @ cov_v @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight)] * n if long_only else [(-1.0, 1.0)] * n

    res = minimize(neg_utility, np.ones(n) / n, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 2000})

    if not res.success:
        raise RuntimeError(f"BL optimization failed: {res.message}")

    return pd.Series(res.x, index=mu_bl.index, name="w_bl")


def bl_summary(pi: pd.Series, mu_bl: pd.Series, w_gmp: pd.Series,
               w_bl: pd.Series) -> pd.DataFrame:
    """Summary table: prior, posterior, GMP weights, BL weights."""
    return pd.DataFrame({
        "pi_prior": pi,
        "mu_bl": mu_bl,
        "delta_mu": mu_bl - pi,
        "w_gmp": w_gmp,
        "w_bl": w_bl,
        "delta_w": w_bl - w_gmp,
    })