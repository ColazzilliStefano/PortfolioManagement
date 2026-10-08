import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.utils.config import project_root, get_config
from src.utils.logger import setup_logger
from src.data.loader import load_processed
from src.gmp.backtest import backtest_gmp, compute_turnover
from src.analytics.performance import summary
from src.analytics.returns import log_to_simple
from src.viz import plots as P


def main():
    cfg = get_config()
    log = setup_logger(cfg["logging"]["level"], cfg["logging"]["file"])

    # load ticker -> GMP asset class mapping
    with open(ROOT / "config" / "assets.yaml", "r") as f:
        assets_cfg = yaml.safe_load(f)
    ticker_to_gmp = assets_cfg["ticker_to_gmp"]

    returns = load_processed("returns_monthly")
    weights = pd.read_parquet(project_root() / "data/processed/gmp_weights_history.parquet")

    rf_ticker = cfg["data"]["risk_free_ticker"]
    rf_simple = log_to_simple(returns[rf_ticker]) if rf_ticker in returns.columns else None

    returns_risky = returns.drop(columns=[rf_ticker], errors="ignore")

    port = backtest_gmp(returns_risky, weights, ticker_to_gmp, return_type="log")
    turnover = compute_turnover(weights)

    perf = summary(port, "GMP", rf_series=rf_simple)
    log.info(f"Performance GMP: {perf}")
    pd.DataFrame([perf]).set_index("name").to_csv(
        project_root() / "outputs/tables/gmp_performance.csv")

    port.to_frame("gmp_return").to_parquet(
        project_root() / "data/processed/gmp_returns.parquet")
    turnover.to_frame("turnover").to_csv(
        project_root() / "outputs/tables/gmp_turnover.csv")

    P.plot_gmp_weights_evolution(weights)
    P.plot_equity_curves({"GMP": port}, name="gmp_equity_curve.png")
    P.plot_turnover(turnover)

    log.info(f"Mean monthly return: {port.mean():.6f}")
    log.info(f"number of observations: {len(port)}")
    log.info(f"first dates: {port.index[:3].tolist()}")
    log.info(f"last dates: {port.index[-3:].tolist()}")


if __name__ == "__main__":
    main()