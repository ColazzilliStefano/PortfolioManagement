import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
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
from src.black_litterman.bl_model import bl_posterior, optimal_weights_bl
from src.black_litterman.views import build_multi_views


def main():
    cfg = get_config()
    log = setup_logger(cfg["logging"]["level"], cfg["logging"]["file"])

    with open(ROOT / "config" / "assets.yaml", "r") as f:
        assets_cfg = yaml.safe_load(f)
    ticker_to_gmp = assets_cfg["ticker_to_gmp"]

    returns = load_processed("returns_monthly")
    weights_history = pd.read_parquet(
        project_root() / "data/processed/gmp_weights_history.parquet"
    )

    rf = cfg["data"]["risk_free_ticker"]
    returns_risky = returns.drop(columns=[rf], errors="ignore")

    r_class, _ = get_class_returns_and_weights(
        returns_risky, weights_history, ticker_to_gmp
    )

    classes = list(r_class.columns)

    cov = shrinkage_cov(r_class, method="ledoit_wolf")

    w_gmp = weights_history.tail(2).mean()
    w_gmp = w_gmp.reindex(classes).fillna(0.0)
    w_gmp = w_gmp / w_gmp.sum()

    lam = calibrate_risk_aversion(cov, w_gmp, market_sharpe=0.4)
    pi = implied_returns(cov, w_gmp, lam)

    log.info(f"Lambda: {lam:.4f}")
    log.info(f"\nPi (prior):\n{pi.round(4).to_string()}")

    scenarios = [
    ("growth", 0.055),       
    ("inflation_up", 0.030),  
    ]
    P, Q = build_multi_views(classes, scenarios)
    log.info(f"\nP:\n{P}")
    log.info(f"Q: {Q}")

    # BL posterior
    tau = 0.05
    mu_bl, cov_bl = bl_posterior(pi, cov, P, Q, tau=tau)

    log.info(f"\nMu Black-Litterman:\n{mu_bl.round(4).to_string()}")
    delta_mu = (mu_bl - pi).round(4)
    log.info(f"\nDelta mu (BL - prior):\n{delta_mu.to_string()}")

    w_bl = optimal_weights_bl(mu_bl, cov, lam, long_only=True, max_weight=0.5)
    log.info(f"\nBlack-Litterman weights:\n{w_bl.round(4).to_string()}")
    log.info(f"\nGMP weights (prior):\n{w_gmp.round(4).to_string()}")

    diff = (w_bl - w_gmp).round(4)
    log.info(f"\nWeight difference (BL - GMP):\n{diff.to_string()}")

    out = project_root() / "outputs/tables"
    out.mkdir(parents=True, exist_ok=True)
    mu_bl.to_frame("mu_bl").to_csv(out / "bl_posterior_mu.csv")
    pi.to_frame("pi").to_csv(out / "bl_prior_pi.csv")
    w_bl.to_frame("w_bl").to_csv(out / "bl_weights.csv")
    w_gmp.to_frame("w_gmp").to_csv(out / "gmp_weights_recent.csv")
    pd.DataFrame({"P": P.tolist(), "Q": Q}).to_csv(out / "bl_views.csv", index=False)

    log.info(f"\nSaved to {out}")


if __name__ == "__main__":
    main()