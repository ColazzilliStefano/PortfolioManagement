"""Regime detection with Hidden Markov Model.

The HMM model estimates the probabilities of being in each regime
(bull, bear, transition) at each month, based on the historical
returns of the equity.

The regimes are used to calibrate the confidence of the BL view.
"""

"""Regime detection with Hidden Markov Model (multivariate)."""

import warnings
import logging
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from loguru import logger


# hmmlearn uses standard logging to print non-convergence warnings.
# We filter them to avoid polluting the output, but we keep track of the number of
# iterations for debugging.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


def fit_hmm(returns, n_states: int = 3, n_iter: int = 100,
            tol: float = 1e-2, seed: int = 42) -> GaussianHMM:
    if isinstance(returns, pd.Series):
        X = returns.dropna().values.reshape(-1, 1)
    else:
        X = returns.dropna().values

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        tol=tol,
        random_state=seed,
    )

    # silence the "Model is not converging" warning specific to hmmlearn
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not converging.*")
        model.fit(X)

    logger.info(
        f"HMM fit: {n_states} states, converged={model.monitor_.converged}, "
        f"iter={model.monitor_.iter}"
    )
    return model


def smooth_states(model: GaussianHMM, returns) -> pd.DataFrame:
    if isinstance(returns, pd.Series):
        X = returns.dropna().values.reshape(-1, 1)
        idx = returns.dropna().index
    else:
        X = returns.dropna().values
        idx = returns.dropna().index

    _, state_probs = model.score_samples(X)
    df = pd.DataFrame(
        state_probs,
        index=idx,
        columns=[f"state_{i}" for i in range(model.n_components)],
    )
    return df


def label_states_by_return(model: GaussianHMM, returns) -> dict:
    if isinstance(returns, pd.Series):
        X = returns.dropna().values.reshape(-1, 1)
        target = X[:, 0]
    else:
        X = returns.dropna().values
        target = X[:, 0]

    states = model.predict(X)
    means = {}
    for s in range(model.n_components):
        if (states == s).any():
            means[s] = float(target[states == s].mean())
        else:
            means[s] = 0.0

    sorted_states = sorted(means, key=means.get, reverse=True)
    labels = {sorted_states[0]: "bull"}
    if len(sorted_states) >= 3:
        labels[sorted_states[1]] = "transition"
        labels[sorted_states[2]] = "bear"
    else:
        labels[sorted_states[1]] = "bear"
    return labels


def regime_probs(returns, n_states: int = 3, seed: int = 42) -> pd.DataFrame:
    model = fit_hmm(returns, n_states=n_states, seed=seed)
    probs = smooth_states(model, returns)
    labels = label_states_by_return(model, returns)
    renamed = probs.rename(
        columns={f"state_{k}": v for k, v in labels.items()}
    )

    for col in ["bull", "transition", "bear"]:
        if col not in renamed.columns:
            renamed[col] = 0.0

    return renamed[["bull", "transition", "bear"]]


def regime_to_view(regime_df: pd.DataFrame,
                   max_magnitude: float = 0.05) -> pd.Series:
    signal = regime_df["bull"] - regime_df["bear"]
    return signal * max_magnitude