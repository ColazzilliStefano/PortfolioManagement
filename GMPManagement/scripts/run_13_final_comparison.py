import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.config import (
    project_root, get_config, get_assets,
    get_bl_config, get_active_signals, get_views_config,
)
from src.utils.logger import setup_logger
from src.utils.manifest import create_manifest
from src.data.loader import load_processed
from src.signals.engine import build_external_signals, normalize_index
from src.gmp.analytics import get_class_returns_and_weights, portfolio_returns
from src.black_litterman.bl_backtest import bl_walk_forward, compute_turnover_bl
from src.benchmarks.equal_weight import backtest_equal_weight
from src.benchmarks.sixty_forty import backtest_sixty_forty
from src.benchmarks.all_weather import backtest_all_weather
from src.markowitz.backtest import walk_forward as markowitz_walk_forward
from src.costs.transaction_costs import apply_costs_to_returns, DEFAULT_COSTS_BPS
from src.analytics.performance import summary
from src.analytics.returns import log_to_simple
from src.viz import plots as P


def _static_weights_history(index, weights, rebalance):
    """Create rebalance dates for a fixed-weight benchmark."""
    dates = pd.DatetimeIndex(index)
    return pd.DataFrame(
        [weights.copy() for _ in dates[::rebalance]],
        index=dates[::rebalance],
    )


def _ticker_costs(ticker_to_gmp):
    return {
        ticker: DEFAULT_COSTS_BPS.get(asset_class, 10.0)
        for ticker, asset_class in ticker_to_gmp.items()
    }


def _weights_for_oos(weights, start):
    """returns_monthly"""
    weights = weights.copy()
    weights.index = pd.to_datetime(weights.index, utc=True).tz_localize(None)
    start = pd.Timestamp(start).tz_localize(None)
    weights = weights.sort_index()
    before = weights.loc[weights.index <= start]
    after = weights.loc[weights.index > start]
    if before.empty:
        raise ValueError("Weights are missing before the start of OOS")
    initial = before.iloc[[-1]].copy()
    initial.index = [start - pd.Timedelta(days=1)]
    return pd.concat([initial, after])


def main():
    cfg = get_config()
    log = setup_logger(cfg["logging"]["level"], cfg["logging"]["file"])

    # 1) GMP
    manifest = create_manifest(ROOT)
    log.info(f"Run ID: {manifest['run_id']} | git: {manifest['git_commit']}")

    bl_cfg = get_bl_config()
    active_signals = get_active_signals()
    views_cfg = get_views_config()
    assets_cfg = get_assets()
    ticker_to_gmp = assets_cfg["ticker_to_gmp"]

    log.info(f"Active signals from config: {active_signals}")
    log.info(f"BL config: {bl_cfg}")

    # data
    returns = load_processed("returns_monthly")
    weights_gmp = pd.read_parquet(
        project_root() / "data/processed/gmp_weights_history.parquet"
    )

    rf_ticker = cfg["data"]["risk_free_ticker"]
    if rf_ticker in returns.columns:
        rf_simple = normalize_index(log_to_simple(returns[rf_ticker]))
    else:
        rf_simple = None

    returns_risky = returns.drop(columns=[rf_ticker], errors="ignore")

    r_class, w_monthly = get_class_returns_and_weights(
        returns_risky, weights_gmp, ticker_to_gmp
    )

    # GMP
    log.info("Computing GMP...")
    port_gmp = portfolio_returns(r_class, w_monthly, return_type="log")

    # BL fixed
    view_scenarios = [
        tuple(x) for x in views_cfg.get(
            "fixed_scenarios",
            [["growth", 0.055], ["inflation_up", 0.030]],
        )
    ]

    log.info("Backtest BL fixed...")
    port_bl_fix, w_bl_fix = bl_walk_forward(
        r_class, weights_gmp,
        window=bl_cfg.get("window", 60),
        rebalance=bl_cfg.get("rebalance", 6),
        view_scenarios=view_scenarios,
        dynamic_views=False,
        tau=bl_cfg.get("tau", 0.05),
        market_sharpe=bl_cfg.get("market_sharpe", 0.4),
        cov_method=bl_cfg.get("cov_method", "ledoit_wolf"),
        max_weight=bl_cfg.get("max_weight", 0.30),
        long_only=bl_cfg.get("long_only", True),
        name="BL fixed",
    )

    # BL dynamics (with signals from the config)
    log.info(f"Construction of external signals: {active_signals}")
    external_signals = build_external_signals(returns_risky, active_signals)
    log.info(f"Signals loaded: {list(external_signals.keys())}")

    log.info("Backtest BL dynamics (HMM walk-forward)...")
    port_bl_dyn, w_bl_dyn = bl_walk_forward(
        r_class, weights_gmp,
        window=bl_cfg.get("window", 60),
        rebalance=bl_cfg.get("rebalance", 6),
        dynamic_views=True,
        tau=bl_cfg.get("tau", 0.05),
        market_sharpe=bl_cfg.get("market_sharpe", 0.4),
        cov_method=bl_cfg.get("cov_method", "ledoit_wolf"),
        max_weight=bl_cfg.get("max_weight", 0.30),
        long_only=bl_cfg.get("long_only", True),
        name="BL dynamics",
        external_signals=external_signals,
        ensemble_mode=bl_cfg.get("ensemble_mode", "mean"),
        ir_lookback=bl_cfg.get("ir_lookback", 60),
        omega_mode=bl_cfg.get("omega_mode", "default"),
        omega_lookback=bl_cfg.get("omega_lookback", 60),
        omega_cap_multiplier=bl_cfg.get("omega_cap_multiplier", 5.0),
        q_mode=bl_cfg.get("q_mode", "default"),
    )

    # Benchmarks
    log.info("Computing benchmarks...")
    rebalance = bl_cfg.get("rebalance", 6)
    port_ew = normalize_index(
        backtest_equal_weight(returns_risky, rebalance=rebalance)
    )
    port_6040 = normalize_index(
        backtest_sixty_forty(returns_risky, rebalance=rebalance)
    )
    port_aw = normalize_index(
        backtest_all_weather(returns_risky, rebalance=rebalance)
    )

    log.info("Computing Markowitz max-Sharpe and minimum-variance...")
    port_mkv, w_mkv = markowitz_walk_forward(
        returns_risky,
        window=bl_cfg.get("window", 60),
        rebalance=rebalance,
        method="max_sharpe",
        long_only=bl_cfg.get("long_only", True),
        max_weight=bl_cfg.get("max_weight", 0.30),
        rf_series=rf_simple,
        return_type="log",
    )
    port_mkv_min, w_mkv_min = markowitz_walk_forward(
        returns_risky,
        window=bl_cfg.get("window", 60),
        rebalance=rebalance,
        method="min_var",
        long_only=bl_cfg.get("long_only", True),
        max_weight=bl_cfg.get("max_weight", 0.30),
        rf_series=rf_simple,
        return_type="log",
    )
    port_mkv = normalize_index(port_mkv)
    port_mkv_min = normalize_index(port_mkv_min)

    # Alignment summary
    start = max(
        port_bl_dyn.index.min(),
        port_bl_fix.index.min(),
        port_gmp.index.min(),
        port_mkv.index.min(),
        port_mkv_min.index.min(),
    )
    log.info(f"Common window: from {start.date()}")

    curves = {
        "BL dynamics": port_bl_dyn.loc[start:],
        "BL fixed": port_bl_fix.loc[start:],
        "GMP": port_gmp.loc[start:],
        "Equal Weight": port_ew.loc[start:],
        "60/40": port_6040.loc[start:],
        "All Weather": port_aw.loc[start:],
        "Markowitz Max Sharpe": port_mkv.loc[start:],
        "Markowitz Min Variance": port_mkv_min.loc[start:],
    }

    # Performance
    ew_weights = pd.Series(1.0 / returns_risky.shape[1],
                           index=returns_risky.columns)
    w_6040 = pd.Series(0.0, index=returns_risky.columns)
    if "ACWI" in w_6040.index:
        w_6040["ACWI"] = 0.60
    if "AGG" in w_6040.index:
        w_6040["AGG"] = 0.40
    w_aw = pd.Series(0.0, index=returns_risky.columns)
    for ticker, weight in {"ACWI": 0.30, "TLT": 0.40, "LQD": 0.15,
                           "GLD": 0.075, "DBC": 0.075}.items():
        if ticker in w_aw.index:
            w_aw[ticker] = weight

    ticker_cost_bps = _ticker_costs(ticker_to_gmp)
    cost_inputs = {
        "BL dynamics": (curves["BL dynamics"], w_bl_dyn,
                        DEFAULT_COSTS_BPS),
        "BL fixed": (curves["BL fixed"], w_bl_fix, DEFAULT_COSTS_BPS),
        "GMP": (curves["GMP"], w_monthly, DEFAULT_COSTS_BPS),
        "Equal Weight": (
            curves["Equal Weight"],
            _static_weights_history(pd.DatetimeIndex([start]), ew_weights, 1),
            ticker_cost_bps,
        ),
        "60/40": (
            curves["60/40"],
            _static_weights_history(pd.DatetimeIndex([start]), w_6040, 1),
            ticker_cost_bps,
        ),
        "All Weather": (
            curves["All Weather"],
            _static_weights_history(pd.DatetimeIndex([start]), w_aw, 1),
            ticker_cost_bps,
        ),
        "Markowitz Max Sharpe": (curves["Markowitz Max Sharpe"], w_mkv,
                                  ticker_cost_bps),
        "Markowitz Min Variance": (curves["Markowitz Min Variance"],
                                    w_mkv_min, ticker_cost_bps),
    }
    curves_net = {
        name: apply_costs_to_returns(
            port, _weights_for_oos(weights, start), costs
        )
        for name, (port, weights, costs) in cost_inputs.items()
    }

    # Gross and net performance summary
    rows_gross = [summary(r, name, rf_series=rf_simple)
                  for name, r in curves.items()]
    rows_net = [summary(r, name, rf_series=rf_simple)
                for name, r in curves_net.items()]
    perf_gross = pd.DataFrame(rows_gross).set_index("name")
    perf_net = pd.DataFrame(rows_net).set_index("name")
    cost_impact = perf_gross[["CAGR", "Sharpe"]].subtract(
        perf_net[["CAGR", "Sharpe"]]
    ).rename(columns={"CAGR": "CAGR_drag", "Sharpe": "Sharpe_drag"})
    log.info(f"\nNet performance:\n{perf_net.round(4).to_string()}")

    out_tables = project_root() / "outputs/tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    perf_gross.to_csv(out_tables / "final_comparison_gross.csv")
    perf_net.to_csv(out_tables / "final_comparison_net.csv")
    perf_net.to_csv(out_tables / "final_comparison.csv")
    cost_impact.to_csv(out_tables / "final_cost_impact.csv")

    # Annual returns
    yearly = pd.DataFrame({
        name: r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
        for name, r in curves_net.items()
    })
    yearly.to_csv(out_tables / "final_yearly_returns.csv")
    log.info(f"\nAnnual returns:\n{yearly.round(4).to_string()}")

    # Turnover
    turnover_dyn = compute_turnover_bl(w_bl_dyn)
    turnover_fix = compute_turnover_bl(w_bl_fix)

    log.info(f"\nTurnover BL dynamics: mean {turnover_dyn.mean():.4f}, "
             f"max {turnover_dyn.max():.4f}")
    log.info(f"Turnover BL fixed:    mean {turnover_fix.mean():.4f}, "
             f"max {turnover_fix.max():.4f}")

    turnover_dyn.to_frame("turnover").to_csv(
        out_tables / "bl_dynamic_turnover.csv")
    turnover_fix.to_frame("turnover").to_csv(
        out_tables / "bl_fixed_turnover.csv")

    turnover_rows = []
    for name, (_, weights, _) in cost_inputs.items():
        turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
        turnover_rows.append({
            "name": name,
            "mean_turnover": turnover.mean(),
            "max_turnover": turnover.max(),
        })
    pd.DataFrame(turnover_rows).set_index("name").to_csv(
        out_tables / "turnover_comparison.csv")

    #Historical weights
    w_bl_dyn.to_csv(out_tables / "bl_dynamic_weights.csv")
    w_bl_fix.to_csv(out_tables / "bl_fixed_weights.csv")
    w_mkv.to_csv(out_tables / "markowitz_max_sharpe_weights.csv")
    w_mkv_min.to_csv(out_tables / "markowitz_min_variance_weights.csv")

    # Figures
    log.info("Generating figures...")
    P.plot_equity_curves(curves_net, name="final_equity.png")
    P.plot_drawdowns(curves_net, name="final_drawdowns.png")
    P.plot_rolling_sharpe(curves_net, window=24, name="final_rolling_sharpe.png")
    P.plot_yearly_returns(yearly, name="final_yearly_returns.png")

    # Summary
    log.info("\n" + "=" * 60)
    log.info("FINAL SUMMARY")
    log.info("=" * 60)

    for name in curves:
        row = perf_net.loc[name]
        log.info(
            f"{name:15s} | CAGR {row['CAGR']*100:6.2f}% | "
            f"Vol {row['Vol']*100:5.2f}% | "
            f"Sharpe {row['Sharpe']:.2f} | "
            f"MaxDD {row['MaxDD']*100:6.2f}% | "
            f"Calmar {row['Calmar']:.2f}"
        )

    log.info("=" * 60)
    log.info(f"Manifest: outputs/runs/{manifest['run_id']}/manifest.json")
    log.info(f"Tables: {out_tables}")
    log.info(f"Figures: {project_root() / 'outputs/figures'}")


if __name__ == "__main__":
    main()