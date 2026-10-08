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
from src.black_litterman.bl_backtest import bl_walk_forward
from src.costs.transaction_costs import apply_costs_to_returns
from src.analytics.performance import summary
from src.analytics.returns import log_to_simple
from src.signals.engine import normalize_index
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

    rf = cfg["data"]["risk_free_ticker"]
    rf_simple = (
        normalize_index(log_to_simple(returns[rf]))
        if rf in returns.columns
        else None
    )
    returns_risky = returns.drop(columns=[rf], errors="ignore")

    r_class, w_monthly = get_class_returns_and_weights(
        returns_risky, weights_gmp, ticker_to_gmp
    )

   
    port_gmp = portfolio_returns(r_class, w_monthly, return_type="log")


    log.info("BL dynamics")
    port_bl_dyn, w_bl_dyn = bl_walk_forward(
        r_class, weights_gmp,
        window=60, rebalance=6,
        dynamic_views=True, tau=0.05, max_weight=0.3,
        name="BL dynamics",
    )

    start = port_bl_dyn.index.min()
    port_gmp = port_gmp.loc[start:]

    w_gmp_rebal = weights_gmp.loc[weights_gmp.index >= start]
    port_gmp_net = apply_costs_to_returns(
        port_gmp, w_gmp_rebal
    )

  
    port_bl_net = apply_costs_to_returns(
        port_bl_dyn, w_bl_dyn
    )

    curves = {
        "Net BL dynamics": port_bl_net,
        "Gross BL dynamics": port_bl_dyn,
        "Net GMP": port_gmp_net,
        "Gross GMP": port_gmp,
    }

    rows = [summary(r, n, rf_series=rf_simple) for n, r in curves.items()]
    perf = pd.DataFrame(rows).set_index("name")
    log.info(f"\nPerformance with costs:\n{perf.round(4).to_string()}")

    perf.to_csv(project_root() / "outputs/tables/costs_impact.csv")


    gmp_drag = (perf.loc["Gross GMP", "CAGR"] - perf.loc["Net GMP", "CAGR"]) * 10000
    bl_drag = (perf.loc["Gross BL dynamics", "CAGR"] - perf.loc["Net BL dynamics", "CAGR"]) * 10000
    log.info(f"\nGMP drag: {gmp_drag:.0f} bps/year")
    log.info(f"BL dynamics drag: {bl_drag:.0f} bps/year")

    P.plot_equity_curves(curves, name="costs_impact_equity.png")
    P.plot_drawdowns(curves, name="costs_impact_drawdowns.png")
    P.plot_rolling_sharpe(curves, window=24, name="costs_impact_rolling_sharpe.png")

if __name__ == "__main__":
    main()