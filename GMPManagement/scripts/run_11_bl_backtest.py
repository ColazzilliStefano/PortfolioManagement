import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.utils.config import project_root, get_config
from src.utils.logger import setup_logger
from src.data.loader import load_processed
from src.gmp.analytics import get_class_returns_and_weights, portfolio_returns
from src.black_litterman.bl_backtest import bl_walk_forward, compute_turnover_bl
from src.analytics.performance import summary
from src.analytics.returns import log_to_simple
from src.viz import plots as P


def main():
    cfg = get_config()
    log = setup_logger(cfg["logging"]["level"], cfg["logging"]["file"])

    with open(ROOT / "config" / "assets.yaml", "r") as f:
        assets_cfg = yaml.safe_load(f)
    ticker_to_gmp = assets_cfg["ticker_to_gmp"]

    returns = load_processed("returns_monthly")
    weights_gmp = pd.read_parquet(
        project_root() / "data/processed/gmp_weights_history.parquet"
    )

    # risk-free time-varying
    rf_ticker = cfg["data"]["risk_free_ticker"]
    rf_simple = (
        log_to_simple(returns[rf_ticker])
        if rf_ticker in returns.columns
        else None
    )
    returns_risky = returns.drop(columns=[rf_ticker], errors="ignore")

    r_class, w_monthly = get_class_returns_and_weights(
        returns_risky, weights_gmp, ticker_to_gmp
    )

    port_gmp = portfolio_returns(r_class, w_monthly, return_type="log")

    # BL fixed
    log.info("Backtest BL fixed...")
    port_bl_fix, w_bl_fix = bl_walk_forward(
        r_class, weights_gmp,
        window=60, rebalance=6,
        view_scenarios=[("growth", 0.055), ("inflation_up", 0.030)],
        dynamic_views=False,
        tau=0.05, max_weight=0.3,
        name="BL fixed",
    )

    # BL dynamic
    log.info("Backtest BL dynamic (signal engine + HMM)...")
    port_bl_dyn, w_bl_dyn = bl_walk_forward(
        r_class, weights_gmp,
        window=60, rebalance=6,
        dynamic_views=True,
        tau=0.05, max_weight=0.3,
        name="BL dynamic",
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

    rows = [
        summary(r, name, rf_series=rf_simple)
        for name, r in curves.items()
    ]
    perf = pd.DataFrame(rows).set_index("name")
    log.info(f"\nPerformance (rf time-varying):\n{perf.round(4).to_string()}")

    perf.to_csv(project_root() / "outputs/tables/bl_dynamic_vs_gmp.csv")

    w_bl_dyn.to_csv(project_root() / "outputs/tables/bl_dynamic_weights.csv")
    w_bl_fix.to_csv(project_root() / "outputs/tables/bl_fixed_weights.csv")
    port_bl_dyn.to_frame("bl_dyn_return").to_parquet(
        project_root() / "data/processed/bl_dyn_returns.parquet"
    )

    # turnover
    turnover = compute_turnover_bl(w_bl_dyn)
    log.info(
        f"\nTurnover BL dynamic: mean {turnover.mean():.4f}, "
        f"max {turnover.max():.4f}"
    )
    turnover.to_frame("turnover").to_csv(
        project_root() / "outputs/tables/bl_dynamic_turnover.csv"
    )

    # plots
    P.plot_equity_curves(curves, name="bl_dynamic_equity.png")
    P.plot_drawdowns(curves, name="bl_dynamic_drawdowns.png")
    P.plot_rolling_sharpe(curves, window=24, name="bl_dynamic_rolling_sharpe.png")
    P.plot_gmp_weights_evolution(w_bl_dyn, name="bl_dynamic_weights.png")

    log.info(f"\nSaved to {project_root() / 'outputs/tables'}")


if __name__ == "__main__":
    main()