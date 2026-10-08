"""CVaR optimization
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from loguru import logger


#CVaR metric
def compute_cvar(returns_matrix: np.ndarray, w: np.ndarray,
                 alpha: float = 0.05) -> float:
    port_returns = returns_matrix @ w
    var_threshold = np.quantile(port_returns, alpha)
    tail = port_returns[port_returns <= var_threshold]
    if len(tail) == 0:
        return -float(var_threshold)
    return -float(tail.mean())


#Validation
def _validate_max_weight(n: int, max_weight: float) -> None:
    """Raises ValueError if max_weight is infeasible with n assets."""
    if max_weight < 1.0 / n:
        raise ValueError(
            f"max_weight={max_weight:.4f} infeasible with {n} assets. "
            f"Minimum required: {1.0/n:.4f}"
        )
    if max_weight > 1.0:
        raise ValueError(f"max_weight={max_weight} > 1.0 not valid")


#CVaR weights
def cvar_weights(returns: pd.DataFrame,
                 alpha: float = 0.05,
                 long_only: bool = True,
                 max_weight: float = 1.0) -> pd.Series:
    X = returns.dropna().values
    T, N = X.shape

    if N == 0:
        raise ValueError("No asset in the DataFrame")

    _validate_max_weight(N, max_weight)

    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha={alpha} must be in (0, 1)")

    if T < 10:
        logger.warning(f"CVaR: few scenarios ({T}). Using equal weight fallback.")
        return _equal_weight_fallback(returns.columns, max_weight)


    def objective(z):
        w = z[:N]
        v = z[N]
        port_returns = X @ w
        excess = np.maximum(-port_returns - v, 0)
        return v + excess.sum() / (alpha * T)

    z0 = np.concatenate([np.ones(N) / N, [0.0]])
    cons = [{"type": "eq", "fun": lambda z: np.sum(z[:N]) - 1.0}]

    if long_only:
        bounds = [(0.0, max_weight)] * N + [(None, None)]
    else:
        bounds = [(-1.0, 1.0)] * N + [(None, None)]

    res = minimize(
        fun=objective,
        x0=z0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 2000},
    )

    if not res.success:
        logger.warning(f"CVaR did not converge: {res.message}. Using fallback.")
        return _equal_weight_fallback(returns.columns, max_weight)

    return pd.Series(res.x[:N], index=returns.columns, name="cvar")


def _equal_weight_fallback(columns, max_weight: float) -> pd.Series:
    n = len(columns)
    w = np.full(n, 1.0 / n)

    if 1.0 / n > max_weight + 1e-12:
        w = np.full(n, max_weight)
        w = w / w.sum()

    return pd.Series(w, index=columns, name="cvar")


#Walk-forward backtest
def cvar_backtest(returns: pd.DataFrame,
                  window: int = 60,
                  rebalance: int = 6,
                  alpha: float = 0.05,
                  long_only: bool = True,
                  max_weight: float = 1.0,
                  return_type: str = "log"
                  ) -> tuple[pd.Series, pd.DataFrame]:
    from src.analytics.returns import log_to_simple

    dates = returns.index
    n = returns.shape[1]
    _validate_max_weight(n, max_weight)

    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha={alpha} must be in (0, 1)")

    port_returns = []
    weights_history = {}

    for i in range(window, len(dates), rebalance):
        train = returns.iloc[i - window:i]
        train_simple = log_to_simple(train) if return_type == "log" else train

        try:
            w = cvar_weights(train_simple, alpha=alpha,
                             long_only=long_only, max_weight=max_weight)
        except Exception as e:
            logger.warning(f"CVaR failed at {dates[i].date()}: {e}. "
                           f"Using fallback.")
            w = _equal_weight_fallback(train.columns, max_weight)

        weights_history[dates[i]] = w

        end = min(i + rebalance, len(dates))
        test = returns.iloc[i:end]
        test_simple = log_to_simple(test) if return_type == "log" else test
        port_returns.append((test_simple * w).sum(axis=1))

    if not port_returns:
        raise RuntimeError("No OOS period calculated")

    port = pd.concat(port_returns).sort_index()
    port.name = "CVaR"

    weights_df = pd.DataFrame(weights_history).T
    weights_df.index = pd.to_datetime(weights_df.index)

    return port, weights_df