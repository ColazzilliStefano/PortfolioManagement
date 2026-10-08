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
from src.data.loader import load_processed
from src.signals.engine import build_external_signals, normalize_index
from src.gmp.analytics import get_class_returns_and_weights, portfolio_returns
from src.black_litterman.bl_backtest import bl_walk_forward, compute_turnover_bl
from src.analytics.performance import summary
from src.analytics.returns import log_to_simple
from src.viz import plots as P


def main():
    cfg = get_config()
    log = setup_logger(cfg["logging"]["level"], cfg["logging"]["file"])

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
    port_gmp = portfolio_returns(r_class, w_monthly, return_type="log")

    log.info(f"Building external signals: {active_signals}")
    external_signals = build_external_signals(returns_risky, active_signals)
    log.info(f"Loaded signals: {list(external_signals.keys())}")

    # view fixed
    view_scenarios = [
        tuple(x) for x in views_cfg.get(
            "fixed_scenarios",
            [["growth", 0.055], ["inflation_up", 0.030]],
        )
    ]

    # BL fixed
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

    # BL dynamic
    log.info("Backtest dynamic BL...")
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

    start = max(port_bl_fix.index.min(), port_bl_dyn.index.min())
    port_gmp = port_gmp.loc[start:]
    port_bl_fix = port_bl_fix.loc[start:]
    port_bl_dyn = port_bl_dyn.loc[start:]

    curves = {
        "BL dynamic": port_bl_dyn,
        "BL fixed": port_bl_fix,
        "GMP": port_gmp,
    }

    rows = [summary(r, name, rf_series=rf_simple) for name, r in curves.items()]
    perf = pd.DataFrame(rows).set_index("name")
    log.info(f"\nPerformance (rf time-varying):\n{perf.round(4).to_string()}")

    perf.to_csv(project_root() / "outputs/tables/bl_dynamic_vs_gmp.csv")

    w_bl_dyn.to_csv(project_root() / "outputs/tables/bl_dynamic_weights.csv")
    w_bl_fix.to_csv(project_root() / "outputs/tables/bl_fixed_weights.csv")
    port_bl_dyn.to_frame("bl_dyn_return").to_parquet(
        project_root() / "data/processed/bl_dyn_returns.parquet"
    )

    turnover = compute_turnover_bl(w_bl_dyn)
    log.info(
        f"\nTurnover BL dynamic: mean {turnover.mean():.4f}, "
        f"max {turnover.max():.4f}"
    )
    turnover.to_frame("turnover").to_csv(
        project_root() / "outputs/tables/bl_dynamic_turnover.csv"
    )

    P.plot_equity_curves(curves, name="bl_dynamic_equity.png")
    P.plot_drawdowns(curves, name="bl_dynamic_drawdowns.png")
    P.plot_rolling_sharpe(curves, window=24, name="bl_dynamic_rolling_sharpe.png")
    P.plot_gmp_weights_evolution(w_bl_dyn, name="bl_dynamic_weights.png")

    log.info(f"\nSaved to {project_root() / 'outputs/tables'}")


if __name__ == "__main__":
    main()