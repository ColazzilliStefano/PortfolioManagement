"""Robust estimation of the covariance matrix.

Implemented methods:
- sample: sample covariance
- ledoit_wolf: shrinkage toward diagonal target
- constant_corr: shrinkage toward constant correlation matrix

All work on monthly simple returns.
"""

import numpy as np
import pandas as pd


def sample_cov(returns: pd.DataFrame, freq: int = 12) -> pd.DataFrame:
    """Annualized sample covariance."""
    return returns.cov() * freq


def ledoit_wolf_shrinkage(returns: pd.DataFrame, freq: int = 12,
                          target: str = "diagonal") -> pd.DataFrame:
    """
    Ledoit-Wolf shrinkage toward:
    - diagonal: F = diag(S)
    - identity: F = mean(var) * I

    Returns annualized shrinkage covariance.
    """
    X = returns.dropna().values
    T, N = X.shape
    X = X - X.mean(axis=0, keepdims=True)
    S = (X.T @ X) / T  # non-annualized sample covariance

    # target F
    if target == "identity":
        mu_var = np.trace(S) / N
        F = mu_var * np.eye(N)
    else:  # diagonal
        F = np.diag(np.diag(S))

    # optimal delta 
    # pi_hat = sum_{ij} Var(x_i * x_j)
    Y = X ** 2
    pi_mat = (Y.T @ Y) / T - S ** 2
    pi_hat = pi_mat.sum()

    # gamma_hat = sum_{ij} (F_ij - S_ij)^2
    gamma_hat = ((F - S) ** 2).sum()

    if gamma_hat <= 0:
        delta = 0.0
    else:
        delta = max(0.0, min(1.0, (pi_hat / T) / gamma_hat))

    Sigma_shrunk = delta * F + (1 - delta) * S

    return pd.DataFrame(Sigma_shrunk * freq,
                        index=returns.columns, columns=returns.columns)


def constant_correlation_shrinkage(returns: pd.DataFrame,
                                   freq: int = 12) -> pd.DataFrame:
    """
    Shrinkage toward constant correlation matrix.
    """
    X = returns.dropna().values
    T, N = X.shape
    X = X - X.mean(axis=0, keepdims=True)
    S = (X.T @ X) / T

    # mean correlation
    std = np.sqrt(np.diag(S))
    corr = S / np.outer(std, std)
    r_bar = (corr.sum() - N) / (N * (N - 1))

    # target: F_ij = r_bar * std_i * std_j for i != j, variance on the diagonal
    F = r_bar * np.outer(std, std)
    np.fill_diagonal(F, np.diag(S))

    # simplified delta
    Y = X ** 2
    pi_mat = (Y.T @ Y) / T - S ** 2
    pi_hat = pi_mat.sum()
    gamma_hat = ((F - S) ** 2).sum()

    delta = 0.0 if gamma_hat <= 0 else max(0.0, min(1.0, (pi_hat / T) / gamma_hat))

    Sigma_shrunk = delta * F + (1 - delta) * S

    return pd.DataFrame(Sigma_shrunk * freq,
                        index=returns.columns, columns=returns.columns)


def shrinkage_cov(returns: pd.DataFrame, method: str = "ledoit_wolf",
                  freq: int = 12) -> pd.DataFrame:
    """
    Unified interface.
    method: "sample" | "ledoit_wolf" | "constant_corr"
    """
    if method == "sample":
        return sample_cov(returns, freq)
    if method == "ledoit_wolf":
        return ledoit_wolf_shrinkage(returns, freq)
    if method == "constant_corr":
        return constant_correlation_shrinkage(returns, freq)
    raise ValueError(f"unknown method: {method}")