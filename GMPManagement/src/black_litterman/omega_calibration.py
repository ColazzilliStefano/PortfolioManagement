import numpy as np
import pandas as pd
from loguru import logger


# Mapping view -> realized differential return
VIEW_DIFFERENTIALS = {
    "growth": ("equity", "bonds"),
    "recession": ("bonds", "equity"),
    "inflation_up": (("gold", "commodities"), "bonds"),
    "inflation_down": ("bonds", ("gold", "commodities")),
}


def _asset_return(r_class: pd.DataFrame, date: pd.Timestamp,
                  asset) -> float:
    """Simple return of an asset or mean if tuple."""
    if date not in r_class.index:
        return np.nan
    row = r_class.loc[date]
    if isinstance(asset, tuple):
        vals = [row.get(a, np.nan) for a in asset]
        vals = [v for v in vals if not pd.isna(v)]
        if not vals:
            return np.nan
        return float(np.mean(vals))
    v = row.get(asset, np.nan)
    return float(v) if not pd.isna(v) else np.nan


def realized_differential_series(r_class: pd.DataFrame,
                                 scenario: str) -> pd.Series:
    """
    Builds the historical series of the realized differential return
    for a scenario, to be used as the regression target.

    Returns Series with index = dates of r_class, values = monthly
    differential return (long - short).
    """
    if scenario not in VIEW_DIFFERENTIALS:
        return pd.Series(dtype=float)

    long_a, short_a = VIEW_DIFFERENTIALS[scenario]
    r_long = pd.Series(
        [_asset_return(r_class, d, long_a) for d in r_class.index],
        index=r_class.index,
    )
    r_short = pd.Series(
        [_asset_return(r_class, d, short_a) for d in r_class.index],
        index=r_class.index,
    )
    diff = (r_long - r_short).dropna()
    diff.name = f"realized_{scenario}"
    return diff


# OLS regression with residuals
def fit_view_regression(view_hist: pd.Series,
                        realized_hist: pd.Series,
                        min_obs: int = 24) -> dict | None:
    
    # align and shift realized by -1 (realized at time t+1 associated with view t)
    view = view_hist.dropna()
    realized = realized_hist.dropna()

    # for each view at t, we want realized at t+1
    # build a DataFrame with view(t) and realized(t+1)
    df = pd.DataFrame({"view": view})
    df["realized_fwd"] = realized.reindex(df.index).shift(-1)
    df = df.dropna()

    if len(df) < min_obs:
        return None

    y = df["realized_fwd"].values
    x = df["view"].values

    # OLS: y = alpha + beta * x
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    if ss_xx == 0:
        return None

    ss_xy = ((x - x_mean) * (y - y_mean)).sum()
    beta = ss_xy / ss_xx
    alpha = y_mean - beta * x_mean

    y_pred = alpha + beta * x
    residuals = y - y_pred

    # R²
    ss_res = (residuals ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "residuals": pd.Series(residuals, index=df.index),
        "r_squared": float(r_squared),
        "n_obs": len(df),
    }


def calibrate_omega_and_q(view_hist: pd.Series,
                          realized_hist: pd.Series,
                          view_current: float,
                          default_omega: float,
                          lookback: int = 60,
                          min_obs: int = 24,
                          cap_multiplier: float = 5.0,
                          floor: float = 1e-8,
                          calibrate_q: bool = False
                          ) -> tuple[float, float | None, dict]:
    
    # take only history < last date (point-in-time)
    view_hist = view_hist.dropna().sort_index()
    if not view_hist.empty:
        view_hist = view_hist.iloc[-lookback:]
    realized_hist = realized_hist.dropna().sort_index()

    if len(view_hist) < min_obs:
        return default_omega, None, {"status": "insufficient_history"}

    fit = fit_view_regression(view_hist, realized_hist, min_obs=min_obs)
    if fit is None:
        return default_omega, None, {"status": "regression_failed"}

    # Omega = variance of residuals
    var_resid = float(fit["residuals"].var(ddof=1))
    if not np.isfinite(var_resid) or var_resid <= 0:
        return default_omega, None, {"status": "invalid_variance",
                                      "r_squared": fit["r_squared"]}

    # cap and floor
    omega = min(var_resid, cap_multiplier * default_omega)
    omega = max(omega, floor)

    # calibrated Q (optional)
    q_cal = None
    if calibrate_q:
        q_cal = float(fit["alpha"] + fit["beta"] * view_current)

    diagnostics = {
        "status": "ok",
        "alpha": fit["alpha"],
        "beta": fit["beta"],
        "r_squared": fit["r_squared"],
        "var_residuals": var_resid,
        "omega_capped": omega,
        "omega_default": default_omega,
        "omega_ratio": omega / default_omega if default_omega > 0 else np.nan,
        "n_obs": fit["n_obs"],
    }
    return omega, q_cal, diagnostics


# Building the Omega matrix

def build_omega_matrix(view_order: list[str],
                       view_histories: dict[str, pd.Series],
                       r_class: pd.DataFrame,
                       default_omega: np.ndarray,
                       lookback: int = 60,
                       cap_multiplier: float = 5.0,
                       calibrate_q: bool = False,
                       log_diagnostics: bool = True
                       ) -> tuple[np.ndarray, np.ndarray | None]:
    n = len(view_order)
    omega = np.array(default_omega, copy=True)

    q_calibrated = np.full(n, np.nan) if calibrate_q else None

    for i, scenario in enumerate(view_order):
        default_var = float(default_omega[i, i])
        view_hist = view_histories.get(scenario)
        if view_hist is None or view_hist.empty:
            if log_diagnostics:
                logger.debug(f"Omega {scenario}: no history, using default")
            continue

        realized = realized_differential_series(r_class, scenario)
        if realized.empty:
            if log_diagnostics:
                logger.debug(f"Omega {scenario}: no realized, using default")
            continue

        # last known value of the view (to calibrate Q)
        view_current = float(view_hist.dropna().iloc[-1]) if not view_hist.dropna().empty else 0.0

        omega_val, q_val, diag = calibrate_omega_and_q(
            view_hist=view_hist,
            realized_hist=realized,
            view_current=view_current,
            default_omega=default_var,
            lookback=lookback,
            cap_multiplier=cap_multiplier,
            calibrate_q=calibrate_q,
        )

        omega[i, i] = omega_val
        if calibrate_q and q_val is not None:
            q_calibrated[i] = q_val

        if log_diagnostics:
            logger.info(
                f"Omega[{scenario}]: "
                f"default={default_var:.6f}, "
                f"calib={omega_val:.6f}, "
                f"ratio={diag.get('omega_ratio', float('nan')):.2f}, "
                f"beta={diag.get('beta', float('nan')):.3f}, "
                f"R²={diag.get('r_squared', float('nan')):.3f}, "
                f"n={diag.get('n_obs', 0)}"
            )

    return omega, q_calibrated