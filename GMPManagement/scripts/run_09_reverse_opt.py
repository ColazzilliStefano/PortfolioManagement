import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.utils.config import project_root, get_config
from src.utils.logger import setup_logger
from src.data.loader import load_processed
from src.gmp.analytics import get_class_returns_and_weights
from src.estimators.shrinkage_cov import shrinkage_cov
from src.black_litterman.reverse_optimization import (
    calibrate_risk_aversion,
    implied_returns,
)


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

    log.info("Comparing covariance estimation methods...")
    cov_sample = shrinkage_cov(r_class, method="sample")
    cov_lw = shrinkage_cov(r_class, method="ledoit_wolf")
    cov_cc = shrinkage_cov(r_class, method="constant_corr")

    log.info(f"\nSample Covariance (diag):\n"
             f"{pd.Series(cov_sample.values.diagonal(), index=r_class.columns).round(4).to_string()}")
    log.info(f"\nLedoit-Wolf Covariance (diag):\n"
             f"{pd.Series(cov_lw.values.diagonal(), index=r_class.columns).round(4).to_string()}")

    w_recent = weights.tail(2).mean()
    w_recent = w_recent.reindex(r_class.columns).fillna(0.0)
    w_recent = w_recent / w_recent.sum()

    log.info(f"\nRecent GMP weights:\n{w_recent.round(4).to_string()}")

    # Rev Opt Ledoit-Wolf 
    cov_use = cov_lw
    lam = calibrate_risk_aversion(cov_use, w_recent, market_sharpe=0.4)
    pi = implied_returns(cov_use, w_recent, lam)

    log.info(f"\nLambda (risk aversion): {lam:.4f}")
    log.info(f"\nImplied returns (annualized):\n{pi.round(4).to_string()}")

    lam_s = calibrate_risk_aversion(cov_sample, w_recent, market_sharpe=0.4)
    pi_sample = implied_returns(cov_sample, w_recent, lam_s)

    comparison = pd.DataFrame({
        "pi_sample": pi_sample,
        "pi_ledoit_wolf": pi,
    })
    log.info(f"\nComparison of implied returns:\n{comparison.round(4).to_string()}")

    pi.to_frame("pi").to_csv(project_root() / "outputs/tables/implied_returns.csv")
    comparison.to_csv(project_root() / "outputs/tables/implied_returns_comparison.csv")
    cov_lw.to_csv(project_root() / "outputs/tables/cov_ledoit_wolf.csv")
    cov_sample.to_csv(project_root() / "outputs/tables/cov_sample.csv")

    log.info(f"Saved to {project_root() / 'outputs/tables'}")


if __name__ == "__main__":
    main()