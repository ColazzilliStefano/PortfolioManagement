import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.utils.config import project_root, get_config
from src.utils.logger import setup_logger
from src.data.loader import load_processed
from src.gmp.analytics import (
    get_class_returns_and_weights,
    portfolio_returns,
    risk_contribution,
    contribution_to_return,
    performance_by_year,
    rolling_correlation,
    drawdown_table,
)
from src.viz import plots as P


def main():
    cfg = get_config()
    log = setup_logger(cfg["logging"]["level"], cfg["logging"]["file"])

    with open(ROOT / "config" / "assets.yaml", "r") as f:
        assets_cfg = yaml.safe_load(f)
    ticker_to_gmp = assets_cfg["ticker_to_gmp"]

    returns = load_processed("returns_monthly")
    weights = pd.read_parquet(
        project_root() / "data/processed/gmp_weights_history.parquet"
    )

    rf = cfg["data"]["risk_free_ticker"]
    returns_risky = returns.drop(columns=[rf], errors="ignore")

    r_class, w_monthly = get_class_returns_and_weights(
        returns_risky, weights, ticker_to_gmp
    )

    # portfolio GMP 
    port = portfolio_returns(r_class, w_monthly, return_type="log")

    # Risk decomposition
    log.info("Computing risk decomposition...")
    risk_df = risk_contribution(r_class, w_monthly)
    log.info(f"\n{risk_df.round(4).to_string()}")
    risk_df.to_csv(project_root() / "outputs/tables/gmp_risk_contribution.csv")

    # Contribution to return 
    contrib = contribution_to_return(r_class, w_monthly)
    log.info(f"\nReturn contribution:\n{contrib.round(4).to_string()}")
    contrib.to_csv(project_root() / "outputs/tables/gmp_return_contribution.csv")

    # Performance per year 
    yearly = performance_by_year(port)
    log.info(f"\nAnnual returns GMP:\n{yearly.round(4).to_string()}")
    yearly.to_csv(project_root() / "outputs/tables/gmp_yearly_returns.csv")

    # Rolling correlation
    corr = rolling_correlation(r_class, window=36)
    corr.to_frame("avg_corr").to_csv(
        project_root() / "outputs/tables/gmp_rolling_correlation.csv"
    )

    # Drawdown table 
    dd_table = drawdown_table(port, top=5)
    log.info(f"\nTop 5 drawdowns:\n{dd_table.to_string()}")
    dd_table.to_csv(project_root() / "outputs/tables/gmp_drawdowns.csv", index=False)

    # Figure
    P.plot_risk_contribution(risk_df)
    P.plot_return_contribution(contrib)
    P.plot_yearly_returns(yearly)
    P.plot_rolling_correlation(corr, window=36)

    log.info(f"figures saved in {project_root() / 'outputs/figures'}")
    log.info(f"tables saved in {project_root() / 'outputs/tables'}")


if __name__ == "__main__":
    main()