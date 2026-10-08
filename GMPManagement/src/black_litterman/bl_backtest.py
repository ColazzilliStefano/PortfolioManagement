"""Black-Litterman walk-forward backtest.

See extended docstring in previous files. Focus on Omega calibration (G7):
- omega_mode="default"     : Omega = tau * P Sigma P'
- omega_mode="calibrated"  : Omega_i = var(OLS residuals) with cap/floor
- q_mode="default"         : Q = view magnitude
- q_mode="calibrated"      : Q_i = alpha + beta * view_current (from the same OLS)
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.estimators.shrinkage_cov import shrinkage_cov
from src.black_litterman.reverse_optimization import (
    calibrate_risk_aversion,
    implied_returns,
)
from src.black_litterman.bl_model import bl_posterior, optimal_weights_bl
from src.black_litterman.views import build_multi_views, build_views
from src.black_litterman.omega_calibration import (
    build_omega_matrix,
    realized_differential_series,
)
from src.analytics.returns import log_to_simple
from src.signals.ensemble import compute_ir_weights, weighted_ensemble


# ---------------------------------------------------------------------------
# Internal helpers (unchanged)
# ---------------------------------------------------------------------------

def _normalize_index(df, tz_source="naive"):
    out = df.copy()
    if tz_source == "utc":
        out.index = pd.to_datetime(out.index, utc=True).tz_localize(None)
    else:
        out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index = out.index.astype("datetime64[ns]")
    return out


def _recent_gmp_weights(wg, date, classes):
    gmp_dates = wg.index[wg.index <= date]
    if len(gmp_dates) == 0:
        return None
    w = wg.loc[gmp_dates[-1]].reindex(classes).fillna(0.0)
    if w.sum() <= 0:
        return None
    return w / w.sum()


def _compute_period_return(r, start_idx, end_idx, weights):
    test = r.iloc[start_idx:end_idx]
    test_simple = log_to_simple(test)
    return (test_simple * weights).sum(axis=1)


def _normalize_external_signals(signals):
    if not signals:
        return {}
    normalized = {}
    for k, s in signals.items():
        if s is None or s.empty:
            continue
        s = s.copy()
        idx = pd.to_datetime(s.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        s.index = idx.astype("datetime64[ns]")
        normalized[k] = s
    return normalized


def _precompute_hmm_series(r, dates, window, rebalance,
                           n_states=3, n_iter=50):
    from src.signals.regimes import (
        fit_hmm, smooth_states, label_states_by_return,
    )
    cols_mv = [c for c in ["equity", "bonds", "gold"] if c in r.columns]
    if len(cols_mv) < 2:
        logger.warning("HMM pre-computation: fewer than 2 asset classes")
        return pd.Series(dtype=float)

    records = {}
    n_fits = (len(dates) - window) // rebalance
    logger.info(f"HMM walk-forward pre-computation: {n_fits} fits expected")

    for j in range(window, len(dates), rebalance):
        d = dates[j]
        train_mv = r[cols_mv].loc[:d].iloc[:-1]
        if len(train_mv) < 60:
            continue
        try:
            model = fit_hmm(train_mv, n_states=n_states, n_iter=n_iter)
            probs = smooth_states(model, train_mv)
            labels = label_states_by_return(model, train_mv)
            renamed = probs.rename(
                columns={f"state_{k}": v for k, v in labels.items()}
            )
            for col in ["bull", "transition", "bear"]:
                if col not in renamed.columns:
                    renamed[col] = 0.0
            last = renamed[["bull", "bear"]].iloc[-1]
            records[d] = float(last["bull"] - last["bear"]) * 0.06
        except Exception as e:
            logger.warning(f"HMM pre-computation failed at {d.date()}: {e}")

    if not records:
        return pd.Series(dtype=float)
    s = pd.Series(records).sort_index()
    s.index = pd.to_datetime(s.index).astype("datetime64[ns]")
    logger.info(f"HMM pre-computed series: {len(s)} observations")
    return s


def _build_growth_composite_history(mom_series, hmm_series,
                                    external_signals, dates_index):
    """Rebuilds the historical series of the growth composite (mean)."""
    parts = []
    if mom_series is not None and not mom_series.empty:
        parts.append(mom_series.rename("momentum"))
    if hmm_series is not None and not hmm_series.empty:
        parts.append(hmm_series.rename("hmm"))
    for k, s in (external_signals or {}).items():
        if s is not None and not s.empty:
            parts.append(s.rename(k))
    if not parts:
        return pd.Series(dtype=float)
    df = pd.concat(parts, axis=1)
    composite = df.mean(axis=1, skipna=True).dropna()
    composite = composite.reindex(dates_index, method="ffill")
    composite.name = "growth"
    return composite



# Walk-forward

def bl_walk_forward(returns_class: pd.DataFrame,
                    weights_gmp: pd.DataFrame,
                    window: int = 60,
                    rebalance: int = 6,
                    view_scenarios=None,
                    dynamic_views: bool = False,
                    tau: float = 0.05,
                    market_sharpe: float = 0.4,
                    cov_method: str = "ledoit_wolf",
                    max_weight: float = 0.3,
                    long_only: bool = True,
                    name: str = "BL",
                    external_signals=None,
                    ensemble_mode: str = "mean",
                    ir_lookback: int = 60,
                    omega_mode: str = "default",
                    omega_lookback: int = 60,
                    omega_cap_multiplier: float = 5.0,
                    q_mode: str = "default"
                    ):
    """
    omega_mode : "default" | "calibrated"
    q_mode : "default" | "calibrated"
    """
    if view_scenarios is None:
        view_scenarios = [("growth", 0.055), ("inflation_up", 0.030)]
    if ensemble_mode not in ("mean", "ir_weighted"):
        raise ValueError(f"unknown ensemble_mode: {ensemble_mode}")
    if omega_mode not in ("default", "calibrated"):
        raise ValueError(f"unknown omega_mode: {omega_mode}")
    if q_mode not in ("default", "calibrated"):
        raise ValueError(f"unknown q_mode: {q_mode}")

    r = _normalize_index(returns_class, tz_source="naive")
    wg = _normalize_index(weights_gmp, tz_source="utc")
    classes = list(r.columns)
    dates = r.index

    if window < 1:
        raise ValueError("window must be >= 1")

    external_signals = _normalize_external_signals(external_signals or {})

    weights_history = {}
    port_returns = []

    yc_series, mom_series, hmm_series = None, None, None
    growth_composite_hist = None
    yield_curve_hist = None
    inflation_hist = None
    realized_by_scenario = {}

    if dynamic_views:
        try:
            from src.signals.yield_curve import yield_spread, yield_to_view
            yc = yield_spread()
            yc_series = yield_to_view(yc, max_magnitude=0.05)
            yc_series = yc_series.reindex(r.index, method="ffill")
            yield_curve_hist = yc_series.rename("recession")
            logger.info("Yield curve signal loaded")
        except Exception as e:
            logger.warning(f"Yield curve unavailable: {e}")

        try:
            from src.signals.momentum import momentum_signal, momentum_to_view
            mom = momentum_signal(r, asset="equity", lookback=6)
            mom_series = momentum_to_view(mom, max_magnitude=0.06)
            logger.info("Momentum signal loaded")
        except Exception as e:
            logger.warning(f"Momentum unavailable: {e}")

        try:
            hmm_series = _precompute_hmm_series(
                r, dates, window=window, rebalance=rebalance,
            )
        except Exception as e:
            logger.warning(f"HMM precomputation unavailable: {e}")
            hmm_series = None

        # growth composite history (for Omega/Q calibration)
        growth_composite_hist = _build_growth_composite_history(
            mom_series, hmm_series, external_signals, r.index
        )
        logger.info(f"Growth composite history: {len(growth_composite_hist)} obs")

        # realized differential pre-computed for each scenario
        if omega_mode == "calibrated" or q_mode == "calibrated":
            for sc in ["growth", "recession", "inflation_up", "inflation_down"]:
                realized_by_scenario[sc] = realized_differential_series(r, sc)
                n_obs = len(realized_by_scenario[sc])
                logger.info(f"Realized {sc}: {n_obs} observations")
        inflation_hist = {}

        logger.info("Inflation signal: vintage availability (lazy load)")

    n_iters = (len(dates) - window) // rebalance
    logger.info(f"Walk-forward: {n_iters} rebalancings expected")

    for iter_num, i in enumerate(range(window, len(dates), rebalance)):
        date = dates[i]
        train = r.iloc[i - window:i]

        cov = shrinkage_cov(train, method=cov_method)

        prev_date = dates[i - 1]
        w_gmp = _recent_gmp_weights(wg, prev_date, classes)
        if w_gmp is None:
            continue

        lam = calibrate_risk_aversion(cov, w_gmp, market_sharpe=market_sharpe)
        pi = implied_returns(cov, w_gmp, lam)

        if dynamic_views:
            mags = {}
            growth_signals = {}

            if mom_series is not None and date in mom_series.index:
                growth_signals["momentum"] = float(mom_series.loc[date])
            if hmm_series is not None and date in hmm_series.index:
                growth_signals["hmm"] = float(hmm_series.loc[date])
            if external_signals:
                for key, series in external_signals.items():
                    if date in series.index:
                        val = series.loc[date]
                        if not pd.isna(val):
                            growth_signals[key] = float(val)

            if growth_signals:
                if ensemble_mode == "mean":
                    mags["growth"] = float(np.mean(list(growth_signals.values())))
                else:
                    hist_signals = {}
                    if mom_series is not None:
                        hist_signals["momentum"] = mom_series
                    if hmm_series is not None:
                        hist_signals["hmm"] = hmm_series
                    for k, s in external_signals.items():
                        hist_signals[k] = s
                    weights = compute_ir_weights(hist_signals, date,
                                                 lookback=ir_lookback)
                    mags["growth"] = weighted_ensemble(growth_signals, weights)

            # CPI vintage
            try:
                from src.signals.inflation import cpi_yoy_vintage
                vintage_str = date.strftime("%Y-%m-%d")
                cpi_mag = cpi_yoy_vintage(vintage_str, max_magnitude=0.04)
                if cpi_mag is not None:
                    mags["inflation_up"] = cpi_mag
                    inflation_hist[date] = cpi_mag
            except Exception as e:
                logger.warning(f"CPI vintage failed at {date.date()}: {e}")

            # recession
            if yc_series is not None and date in yc_series.index:
                mags["recession"] = float(yc_series.loc[date])

            # P, Q, view_order
            P_rows, Q_vals, view_order = [], [], []
            for scenario, mag in mags.items():
                if abs(mag) < 1e-4:
                    continue
                P, Q = build_views(classes, scenario, magnitude=mag)
                P_rows.append(P[0])
                Q_vals.append(Q[0])
                view_order.append(scenario)

            if not P_rows:
                w_bl = w_gmp
            else:
                P = np.array(P_rows)
                Q = np.array(Q_vals)
                omega_default = tau * P @ cov.values @ P.T

                # Omega calibration
                if omega_mode == "calibrated":
                    # histories for each view
                    view_histories = {}
                    # growth: composite
                    if "growth" in view_order and growth_composite_hist is not None:
                        view_histories["growth"] = growth_composite_hist.loc[
                            growth_composite_hist.index < date
                        ]
                    # recession: yield curve
                    if "recession" in view_order and yield_curve_hist is not None:
                        view_histories["recession"] = yield_curve_hist.loc[
                            yield_curve_hist.index < date
                        ]
                    # inflation: from the accumulated history
                    if "inflation_up" in view_order and inflation_hist:
                        view_histories["inflation_up"] = pd.Series(
                            inflation_hist
                        ).sort_index()

                    omega, q_cal = build_omega_matrix(
                        view_order=view_order,
                        view_histories=view_histories,
                        r_class=r,
                        default_omega=omega_default,
                        lookback=omega_lookback,
                        cap_multiplier=omega_cap_multiplier,
                        calibrate_q=(q_mode == "calibrated"),
                        log_diagnostics=True,
                    )

                    if q_mode == "calibrated" and q_cal is not None:
                        # replace the calibrated Qs where available
                        for k, qv in enumerate(q_cal):
                            if not np.isnan(qv):
                                Q[k] = qv
                else:
                    omega = omega_default

                mu_bl, _ = bl_posterior(pi, cov, P, Q, omega=omega, tau=tau)
                try:
                    w_bl = optimal_weights_bl(mu_bl, cov, lam,
                                              long_only=long_only,
                                              max_weight=max_weight)
                except Exception:
                    w_bl = w_gmp
        else:
            P, Q = build_multi_views(classes, view_scenarios)
            mu_bl, _ = bl_posterior(pi, cov, P, Q, tau=tau)
            try:
                w_bl = optimal_weights_bl(mu_bl, cov, lam,
                                          long_only=long_only,
                                          max_weight=max_weight)
            except Exception:
                w_bl = w_gmp

        weights_history[date] = w_bl

        start = i + 1
        end = min(start + rebalance, len(dates))
        if start >= len(dates):
            break
        r_port = _compute_period_return(r, start, end, w_bl)
        port_returns.append(r_port)

        if (iter_num + 1) % 5 == 0:
            logger.info(f"  [{iter_num + 1}/{n_iters}] {date.date()} completed")

    if not port_returns:
        raise RuntimeError("No OOS period calculated")

    port = pd.concat(port_returns).sort_index()
    port.name = name
    weights_df = pd.DataFrame(weights_history).T
    weights_df.index = pd.to_datetime(weights_df.index)
    return port, weights_df


def compute_turnover_bl(weights_history):
    return weights_history.diff().abs().sum(axis=1).fillna(0.0)