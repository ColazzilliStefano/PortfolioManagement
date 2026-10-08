import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.utils.config import project_root, get_config
from src.utils.logger import setup_logger
from src.data.loader import load_processed
from src.gmp.backtest import backtest_gmp
from src.benchmarks.equal_weight import backtest_equal_weight
from src.benchmarks.sixty_forty import backtest_sixty_forty
from src.benchmarks.all_weather import backtest_all_weather
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
    weights = pd.read_parquet(project_root() / "data/processed/gmp_weights_history.parquet")

    rf_ticker = cfg["data"]["risk_free_ticker"]
    rf_simple = log_to_simple(returns[rf_ticker]) if rf_ticker in returns.columns else None

    returns_risky = returns.drop(columns=[rf_ticker], errors="ignore")

    curves = {
        "GMP": backtest_gmp(returns_risky, weights, ticker_to_gmp),
        "Equal Weight": backtest_equal_weight(returns_risky),
        "60/40": backtest_sixty_forty(returns_risky),
        "All Weather": backtest_all_weather(returns_risky),
    }

    rows = [summary(r, name, rf_series=rf_simple) for name, r in curves.items()]
    perf = pd.DataFrame(rows).set_index("name")
    perf.to_csv(project_root() / "outputs/tables/gmp_vs_benchmarks.csv")
    log.info(f"\n{perf.round(4).to_string()}")

    P.plot_equity_curves(curves, name="gmp_vs_benchmarks.png")
    P.plot_drawdowns(curves, name="gmp_drawdowns.png")
    P.plot_rolling_sharpe(curves, window=24)


if __name__ == "__main__":
    main()