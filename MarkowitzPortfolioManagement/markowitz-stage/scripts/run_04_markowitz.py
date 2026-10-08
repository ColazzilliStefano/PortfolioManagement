"""
Complete Markowitz pipeline:

1) load cleaned monthly returns;
2) separate the risk-free asset (BIL) from the risky universe;
3) estimate mu/Sigma and calculate the frontier, tangency, and minimum variance;
4) run the OOS walk-forward for Max Sharpe, Min Variance, and Equal Weight;
5) run critique experiments: estimation error, sensitivity, and turnover;
6) save figures to outputs/figures/ and tables to outputs/tables/.
"""


from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_processed
from src.utils.config import get_config, get_assets
from src.utils.logger import setup_logger

from src.analytics.stats import mean_returns, cov_matrix, corr_matrix
from src.analytics.performance import summary
from src.analytics.returns import log_to_simple

from src.markowitz.mean_variance import (
    min_variance, max_sharpe, portfolio_stats,
)
from src.markowitz.shrinkage import ledoit_wolf_covariance
from src.markowitz.efficient_frontier import (
    compute_frontier, random_portfolios,
)
from src.markowitz.backtest import (
    walk_forward, equal_weight, compute_turnover,
)
from src.markowitz.critique import (
    estimation_error_experiment, sensitivity_to_mu,
)
from src.viz import plots as P


WINDOW = 60
REBALANCE = 6
MAX_WEIGHT = 0.30
LONG_ONLY = True

OUT_FIG = ROOT / "outputs" / "figures"
OUT_TAB = ROOT / "outputs" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)


def split_risk_free(returns: pd.DataFrame, rf_ticker: str):
    if rf_ticker in returns.columns:
        rf_monthly = log_to_simple(returns[rf_ticker])
        rf_annual = float(rf_monthly.mean() * 12)
        returns_risky = returns.drop(columns=[rf_ticker])
    else:
        rf_monthly = pd.Series(0.0, index=returns.index)
        rf_annual = 0.0
        returns_risky = returns.copy()
    return returns_risky, rf_monthly, rf_annual


def main() -> None:
    cfg = get_config()
    assets = get_assets()
    log = setup_logger(cfg["logging"]["level"], cfg["logging"]["file"])

    returns = load_processed("returns_monthly")
    returns = returns.iloc[:-1]
    rf_ticker = cfg["data"]["risk_free_ticker"]
    expected = [a["ticker"] for a in assets["assets"]]
    expected.append(rf_ticker)
    missing = sorted(set(expected) - set(returns.columns))
    if missing:
        raise ValueError(f"Inconsistent data universe. Missing: {missing}")
    returns = returns.loc[:, expected]

    returns_risky, rf_simple, RF = split_risk_free(returns, rf_ticker)
    returns_risky_simple = log_to_simple(returns_risky)

    log.info(f"Total monthly returns: {returns.shape}")
    log.info(f"Risky universe: {returns_risky.shape[1]} assets "
             f"({list(returns_risky.columns)})")
    log.info(f"Annualized risk-free rate ({rf_ticker}): {RF:.4f}")
    log.info(f"Period: {returns_risky.index.min().date()} -> "
             f"{returns_risky.index.max().date()}")

    #samples estimations
    mu = mean_returns(returns_risky_simple)
    cov = cov_matrix(returns_risky_simple)
    corr = corr_matrix(returns_risky_simple)

    #efficient frontier and optimal portfolios
    frontier, _ = compute_frontier(
        mu, cov, n_points=60,
        long_only=LONG_ONLY, max_weight=MAX_WEIGHT, rf=RF,
    )
    w_mv = min_variance(mu, cov, long_only=LONG_ONLY, max_weight=MAX_WEIGHT)
    w_ms = max_sharpe(mu, cov, rf=RF,
                      long_only=LONG_ONLY, max_weight=MAX_WEIGHT)

    ms_stats = portfolio_stats(w_ms, mu, cov, rf=RF)
    mv_stats = portfolio_stats(w_mv, mu, cov, rf=RF)

    w_ms.to_frame("weight").to_csv(OUT_TAB / "tangency_weights.csv")
    w_mv.to_frame("weight").to_csv(OUT_TAB / "minvar_weights.csv")

    #shrinkage Ledoit-Wolf
    cov_shrunk, shrinkage_alpha = ledoit_wolf_covariance(returns_risky_simple)
    w_ms_shrunk = max_sharpe(
        mu, cov_shrunk, rf=RF,
        long_only=LONG_ONLY, max_weight=MAX_WEIGHT,
    )
    w_mv_shrunk = min_variance(
        mu, cov_shrunk,
        long_only=LONG_ONLY, max_weight=MAX_WEIGHT,
    )
    w_ms_shrunk.to_frame("weight").to_csv(
        OUT_TAB / "shrinkage_tangency_weights.csv"
    )
    w_mv_shrunk.to_frame("weight").to_csv(
        OUT_TAB / "shrinkage_minvar_weights.csv"
    )
    shrinkage_rows = [
        {"model": "sample_max_sharpe", **portfolio_stats(w_ms, mu, cov, RF)},
        {"model": "sample_min_variance", **portfolio_stats(w_mv, mu, cov, RF)},
        {"model": "ledoit_wolf_max_sharpe",
         **portfolio_stats(w_ms_shrunk, mu, cov_shrunk, RF)},
        {"model": "ledoit_wolf_min_variance",
         **portfolio_stats(w_mv_shrunk, mu, cov_shrunk, RF)},
    ]
    shrinkage_comparison = pd.DataFrame(shrinkage_rows)
    shrinkage_comparison["shrinkage_alpha"] = shrinkage_alpha
    shrinkage_comparison.to_csv(
        OUT_TAB / "shrinkage_comparison.csv", index=False,
    )
    log.info(f"Ledoit-Wolf shrinkage alpha: {shrinkage_alpha:.4f}")

    log.info("Max Sharpe weights (in-sample):\n" + w_ms.round(4).to_string())
    log.info("Minimum variance weights (in-sample):\n" + w_mv.round(4).to_string())

    #efficient frontier charts
    random_df, _ = random_portfolios(
        mu, cov, n=5000, long_only=LONG_ONLY, rf=RF,
    )
    P.plot_frontier_markowitz(
        random_df, frontier, tangency=ms_stats, minvar=mv_stats,
    )
    P.plot_cal(frontier, ms_stats, rf=RF)
    P.plot_weights(w_ms, name="weights_tangency.png",
                   title="Max Sharpe weights (in-sample)")
    P.plot_weights(w_mv, name="weights_minvar.png",
                   title="Minimum variance weights (in-sample)")
    P.plot_correlation(corr)

    #walk-forward
    port_ms, weights_ms = walk_forward(
        returns_risky, window=WINDOW, rebalance=REBALANCE,
        method="max_sharpe", long_only=LONG_ONLY,
        max_weight=MAX_WEIGHT, rf=RF, rf_series=rf_simple,
        return_type="log",
    )
    port_mv, weights_mv = walk_forward(
        returns_risky, window=WINDOW, rebalance=REBALANCE,
        method="min_var", long_only=LONG_ONLY,
        max_weight=MAX_WEIGHT, rf=RF, rf_series=rf_simple,
        return_type="log",
    )
    port_ew_full = equal_weight(returns_risky, rebalance=REBALANCE,
                                return_type="log")

    start = max(port_ms.index.min(), port_mv.index.min())
    port_ms = port_ms.loc[start:]
    port_mv = port_mv.loc[start:]
    port_ew = port_ew_full.loc[start:]

    turnover_ms = compute_turnover(weights_ms)
    weights_ms.to_csv(OUT_TAB / "rolling_weights.csv")
    turnover_ms.to_frame("turnover").to_csv(OUT_TAB / "turnover.csv")

    curves = {
        "Max Sharpe OOS": port_ms,
        "Min Variance OOS": port_mv,
        "Equal Weight": port_ew,
    }

    P.plot_equity_curves(curves)
    P.plot_drawdowns(curves)
    P.plot_rolling_sharpe(curves, window=24, rf_series=rf_simple)
    P.plot_weights_instability(weights_ms)
    P.plot_turnover(turnover_ms)

    #performance
    rows = [
        summary(port_ms, "Max Sharpe OOS", rf_series=rf_simple),
        summary(port_mv, "Min Variance OOS", rf_series=rf_simple),
        summary(port_ew, "Equal Weight", rf_series=rf_simple),
    ]
    perf = pd.DataFrame(rows).set_index("name")
    perf.to_csv(OUT_TAB / "performance_summary.csv")
    log.info("Performance out-of-sample:\n" + perf.round(4).to_string())

    #annual returns
    wf = pd.DataFrame({
        "MaxSharpe": port_ms,
        "MinVar": port_mv,
        "EqualWeight": port_ew,
    })
    wf_year = wf.groupby(wf.index.year).apply(lambda x: (1.0 + x).prod() - 1)
    wf_year.to_csv(OUT_TAB / "walkforward_summary.csv")
    log.info("OOS annual returns:\n" + wf_year.round(4).to_string())

    #estimation error
    ee = estimation_error_experiment(
        returns_risky, n_sims=300, rf=RF, rf_series=rf_simple
    )
    ee.to_csv(OUT_TAB / "estimation_error.csv", index=False)
    P.plot_estimation_error(ee)

    ee_bar = pd.DataFrame(
        {
            "Media": [ee["in_sample_sharpe"].mean(),
                      ee["out_of_sample_sharpe"].mean()],
            "Mediana": [ee["in_sample_sharpe"].median(),
                        ee["out_of_sample_sharpe"].median()],
        },
        index=["In-sample", "Out-of-sample"],
    )
    P.plot_insample_vs_oos(ee_bar)
    log.info("Estimation error (Sharpe):\n" + ee.mean().round(4).to_string())

    #sensitivity to mu
    w0, sens = sensitivity_to_mu(returns_risky, delta=0.001, n_pert=50, rf=RF)
    sens.to_csv(OUT_TAB / "sensitivity_mu.csv")
    log.info("Weight sensitivity to mu perturbations (std):\n" +
             sens["std"].sort_values(ascending=False).round(4).to_string())

    log.info(f"Figures saved to {OUT_FIG}")
    log.info(f"Tables saved to {OUT_TAB}")


if __name__ == "__main__":
    main()