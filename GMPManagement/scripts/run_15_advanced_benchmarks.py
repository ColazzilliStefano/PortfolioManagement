import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.config import (
    project_root, get_config, get_assets,
    get_bl_config, get_active_signals,
)
from src.utils.logger import setup_logger
from src.utils.manifest import create_manifest
from src.data.loader import load_processed
from src.signals.engine import build_external_signals, normalize_index
from src.gmp.analytics import get_class_returns_and_weights, portfolio_returns
from src.black_litterman.bl_backtest import bl_walk_forward
from src.benchmarks.risk_parity import risk_parity_backtest
from src.benchmarks.hrp import hrp_backtest
from src.risk.cvar import cvar_backtest
from src.costs.transaction_costs import apply_costs_to_returns
from src.analytics.performance import summary
from src.analytics.returns import log_to_simple
from src.viz import plots as P


def _load_advanced_config() -> dict:
    """Reads the advanced_benchmarks section with explicit fallbacks."""
    cfg = get_config()
    sec = cfg.get("advanced_benchmarks", {})

    params = {
        "window": sec.get("window", 60),
        "rebalance": sec.get("rebalance", 6),
        "risk_parity": {
            "long_only": sec.get("risk_parity", {}).get("long_only", True),
            "max_weight": sec.get("risk_parity", {}).get("max_weight", 1.0),
        },
        "hrp": {
            "enabled": sec.get("hrp", {}).get("enabled", True),
            "max_weight": sec.get("hrp", {}).get("max_weight", 1.0),
        },
        "cvar": {
            "alpha": sec.get("cvar", {}).get("alpha", 0.05),
            "long_only": sec.get("cvar", {}).get("long_only", True),
            "max_weight": sec.get("cvar", {}).get("max_weight", 1.0),
        },
        "costs": {
            "apply": sec.get("costs", {}).get("apply", True),
        },
    }
    return params


def main():
    cfg = get_config()
    log = setup_logger(cfg["logging"]["level"], cfg["logging"]["file"])

    #manifest
    manifest = create_manifest(ROOT)
    log.info(f"Run ID: {manifest['run_id']} | git: {manifest['git_commit']}")

    #config
    adv_cfg = _load_advanced_config()
    bl_cfg = get_bl_config()
    active_signals = get_active_signals()
    assets_cfg = get_assets()
    ticker_to_gmp = assets_cfg["ticker_to_gmp"]

    log.info(f"Advanced config: {adv_cfg}")
    log.info(f"Active signals: {active_signals}")

    #data
    returns = load_processed("returns_monthly")
    weights_gmp = pd.read_parquet(
        project_root() / "data/processed/gmp_weights_history.parquet"
    )

    rf_ticker = cfg["data"]["risk_free_ticker"]
    rf_simple = normalize_index(log_to_simple(returns[rf_ticker])) \
        if rf_ticker in returns.columns else None
    returns_risky = returns.drop(columns=[rf_ticker], errors="ignore")

    r_class, w_monthly = get_class_returns_and_weights(
        returns_risky, weights_gmp, ticker_to_gmp
    )

    #GMP
    log.info("Computing GMP...")
    port_gmp = portfolio_returns(r_class, w_monthly, return_type="log")
    port_gmp = normalize_index(port_gmp)

    #Dynamic BL
    external_signals = build_external_signals(returns_risky, active_signals)
    log.info(f"Signals loaded: {list(external_signals.keys())}")

    log.info("Backtesting dynamic BL...")
    port_bl, w_bl = bl_walk_forward(
        r_class, weights_gmp,
        window=bl_cfg.get("window", 60),
        rebalance=bl_cfg.get("rebalance", 6),
        dynamic_views=True,
        tau=bl_cfg.get("tau", 0.05),
        max_weight=bl_cfg.get("max_weight", 0.30),
        external_signals=external_signals,
        name="Dynamic BL",
    )
    port_bl = normalize_index(port_bl)

    #Risk Parity
    log.info("Backtesting Risk Parity...")
    port_rp, w_rp = risk_parity_backtest(
        returns_risky,
        window=adv_cfg["window"],
        rebalance=adv_cfg["rebalance"],
        long_only=adv_cfg["risk_parity"]["long_only"],
        max_weight=adv_cfg["risk_parity"]["max_weight"],
    )
    port_rp = normalize_index(port_rp)

    # HRP 
    port_hrp, w_hrp = None, None
    if adv_cfg["hrp"]["enabled"]:
        log.info("Backtesting HRP...")
        port_hrp, w_hrp = hrp_backtest(
            returns_risky,
            window=adv_cfg["window"],
            rebalance=adv_cfg["rebalance"],
            max_weight=adv_cfg["hrp"]["max_weight"],
        )
        port_hrp = normalize_index(port_hrp)
    else:
        log.info("HRP disabled in config, skip")

    #CVaR
    log.info("Backtesting CVaR...")
    port_cvar, w_cvar = cvar_backtest(
        returns_risky,
        window=adv_cfg["window"],
        rebalance=adv_cfg["rebalance"],
        alpha=adv_cfg["cvar"]["alpha"],
        long_only=adv_cfg["cvar"]["long_only"],
        max_weight=adv_cfg["cvar"]["max_weight"],
    )
    port_cvar = normalize_index(port_cvar)

    #Transaction costs (NET policy)
    if adv_cfg["costs"]["apply"]:
        log.info("Applying transaction costs (net)...")
        port_bl = apply_costs_to_returns(port_bl, w_bl)
        port_rp = apply_costs_to_returns(port_rp, w_rp)
        if port_hrp is not None:
            port_hrp = apply_costs_to_returns(port_hrp, w_hrp)
        port_cvar = apply_costs_to_returns(port_cvar, w_cvar)
        # GMP: uses historical GMP weights
        port_gmp = apply_costs_to_returns(port_gmp, weights_gmp)

    #Alignment
    starts = [port_bl.index.min(), port_gmp.index.min(), port_rp.index.min(),
              port_cvar.index.min()]
    if port_hrp is not None:
        starts.append(port_hrp.index.min())
    start = max(starts)
    log.info(f"Common window: from {start.date()}")

    curves = {
        "Dynamic BL": port_bl.loc[start:],
        "GMP": port_gmp.loc[start:],
        "Risk Parity": port_rp.loc[start:],
        "CVaR": port_cvar.loc[start:],
    }
    if port_hrp is not None:
        curves["HRP"] = port_hrp.loc[start:]

    #Performance
    rows = [summary(r, name, rf_series=rf_simple) for name, r in curves.items()]
    perf = pd.DataFrame(rows).set_index("name").sort_values(
        "Sharpe", ascending=False)
    log.info(f"\nFinal performance (net):\n{perf.round(4).to_string()}")

    out_tables = project_root() / "outputs/tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    perf.to_csv(out_tables / "advanced_benchmarks.csv")


    w_rp.to_csv(out_tables / "advanced_risk_parity_weights.csv")
    w_cvar.to_csv(out_tables / "advanced_cvar_weights.csv")
    w_bl.to_csv(out_tables / "advanced_bl_weights.csv")
    if w_hrp is not None:
        w_hrp.to_csv(out_tables / "advanced_hrp_weights.csv")
    log.info(f"Weights saved to {out_tables}")

    # plots
   
    P.plot_equity_curves(curves, name="advanced_equity.png")
    P.plot_drawdowns(curves, name="advanced_drawdowns.png")
    P.plot_rolling_sharpe(curves, window=24, name="advanced_rolling_sharpe.png")

    for name in perf.index:
        row = perf.loc[name]
        log.info(
            f"{name:15s} | CAGR {row['CAGR']*100:6.2f}% | "
            f"Vol {row['Vol']*100:5.2f}% | "
            f"Sharpe {row['Sharpe']:.2f} | "
            f"MaxDD {row['MaxDD']*100:6.2f}%"
        )
    log.info("=" * 70)
    log.info(f"Manifest: outputs/runs/{manifest['run_id']}/manifest.json")


if __name__ == "__main__":
    main()