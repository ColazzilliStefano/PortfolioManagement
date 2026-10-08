import numpy as np
import pandas as pd


def james_stein_shrinkage(mu: pd.Series, target: float = 0.0,
                          freq: int = 12) -> pd.Series:
    """
    James-Stein shrinkage toward a target (default 0).
    Returns annualized mu.
    """
    mu_a = mu * freq
    n = len(mu_a)
    if n < 3:
        return mu_a

    grand_mean = mu_a.mean()
    sq_dev = ((mu_a - grand_mean) ** 2).sum()
    if sq_dev <= 0:
        return mu_a

    # sample variance of the estimated mu, as mean of the variances / n
    # approximation: uses the std of mu_a
    var_mu = mu_a.var(ddof=1)
    sigma2 = var_mu  # rough estimator

    shrink_factor = max(0.0, 1.0 - (n - 2) * sigma2 / sq_dev)
    target_series = pd.Series(target * freq, index=mu_a.index)

    return target_series + shrink_factor * (mu_a - target_series)


def shrinkage_mu(mu: pd.Series, method: str = "james_stein",
                 freq: int = 12, target: float = 0.0) -> pd.Series:
    if method == "none":
        return mu * freq
    if method == "james_stein":
        return james_stein_shrinkage(mu, target=target, freq=freq)
    raise ValueError(f"unknown method: {method}")