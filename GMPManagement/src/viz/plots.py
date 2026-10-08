"""Plots for GMP and benchmarks."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import project_root
from src.markowitz.efficient_frontier import compute_frontier, random_portfolios
from src.markowitz.mean_variance import (
    max_sharpe,
    min_variance,
    portfolio_stats,
)

OUT = project_root() / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

PALETTE = sns.color_palette("deep")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def plot_markowitz_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    long_only: bool = True,
    max_weight: float = 0.30,
    name: str = "markowitz/frontier_markowitz_constrained.png",
) -> None:
    """Plot the constrained efficient frontier with English labels."""
    frontier, _ = compute_frontier(
        mu, cov, long_only=long_only, max_weight=max_weight, rf=rf
    )
    random_df, _ = random_portfolios(
        mu, cov, long_only=long_only, rf=rf, seed=42,
        max_weight=max_weight,
    )
    w_tangency = max_sharpe(
        mu, cov, rf=rf, long_only=long_only, max_weight=max_weight
    )
    w_min = min_variance(
        mu, cov, long_only=long_only, max_weight=max_weight
    )
    tangency = portfolio_stats(w_tangency, mu, cov, rf)
    minimum = portfolio_stats(w_min, mu, cov, rf)

    fig, ax = plt.subplots(figsize=(11, 7))
    scatter = ax.scatter(
        random_df["vol"], random_df["return"], c=random_df["sharpe"],
        cmap="viridis", s=10, alpha=0.35, edgecolors="none",
        label="Random portfolios",
    )
    if not frontier.empty:
        ax.plot(
            frontier["vol"], frontier["return"], "k.-", lw=1.8,
            ms=4, label="Efficient frontier",
        )
    ax.scatter(
        tangency["vol"], tangency["return"], marker="*", s=260,
        color="red", edgecolors="black", linewidths=0.5,
        label="Maximum Sharpe portfolio",
    )
    ax.scatter(
        minimum["vol"], minimum["return"], marker="s", s=90,
        facecolors="white", edgecolors="black", linewidths=1.2,
        label="Minimum variance portfolio",
    )
    ax.set_title("Portfolio space and efficient frontier")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    ax.legend(loc="upper left")
    fig.colorbar(scatter, ax=ax, label="Sharpe ratio")
    _save(fig, name)


def plot_capital_allocation_line(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    long_only: bool = True,
    max_weight: float = 0.30,
    name: str = "markowitz/capital_allocation_line_constrained.png",
) -> None:
    """Plot the constrained efficient frontier and its capital allocation line."""
    frontier, _ = compute_frontier(
        mu, cov, long_only=long_only, max_weight=max_weight, rf=rf
    )
    w_tangency = max_sharpe(
        mu, cov, rf=rf, long_only=long_only, max_weight=max_weight
    )
    tangency = portfolio_stats(w_tangency, mu, cov, rf)

    max_vol = float(frontier["vol"].max()) if not frontier.empty else tangency["vol"]
    line_vol = np.linspace(0.0, max_vol, 100)
    slope = tangency["sharpe"]
    line_return = rf + slope * line_vol

    fig, ax = plt.subplots(figsize=(11, 7))
    if not frontier.empty:
        ax.plot(
            frontier["vol"], frontier["return"], "b.-", lw=1.8,
            ms=4, label="Efficient frontier",
        )
    ax.plot(
        line_vol, line_return, "r--", lw=1.5,
        label="Capital allocation line",
    )
    ax.scatter(
        tangency["vol"], tangency["return"], marker="*", s=260,
        color="red", edgecolors="black", linewidths=0.5,
        label="Maximum Sharpe portfolio",
    )
    ax.scatter(
        0.0, rf, color="black", s=70, zorder=3,
        label="Risk-free asset",
    )
    ax.set_title("Capital allocation line and constrained efficient frontier")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    ax.legend(loc="upper left")
    _save(fig, name)


def plot_gmp_weights_evolution(weights: pd.DataFrame,
                               name="gmp_weights_evolution.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    weights.plot.area(ax=ax, stacked=True, colormap="tab20", alpha=0.85)
    ax.set_title("GMP weights evolution")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    _save(fig, name)


def plot_equity_curves(curves: dict, name="gmp_equity_curve.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, r in curves.items():
        eq = (1.0 + r.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, label=label, lw=1.6)
    ax.set_yscale("log")
    ax.set_title("Equity curve (log scale, simple returns)")
    ax.set_ylabel("Capital")
    ax.legend()
    _save(fig, name)


def plot_drawdowns(curves: dict, name="gmp_drawdowns.png"):
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, r in curves.items():
        eq = (1.0 + r.fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1.0
        ax.plot(dd.index, dd.values, label=label, lw=1.3)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.legend()
    _save(fig, name)


def plot_rolling_sharpe(curves: dict, window=24, name="gmp_rolling_sharpe.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, r in curves.items():
        rs = (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(12)
        ax.plot(rs.index, rs.values, label=label, lw=1.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Rolling Sharpe ({window}m)")
    ax.legend()
    _save(fig, name)


def plot_correlation(corr: pd.DataFrame, name="gmp_correlation.png"):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation matrix")
    _save(fig, name)


def plot_turnover(turnover: pd.Series, name="gmp_turnover.png"):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(turnover.index, turnover.values, width=20, color=PALETTE[0])
    ax.set_title("GMP turnover")
    ax.set_ylabel("Turnover")
    _save(fig, name)

def plot_risk_contribution(risk_df: pd.DataFrame,
                           name="gmp_risk_contribution.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(risk_df))
    width = 0.35
    ax.bar(x - width/2, risk_df["weight_avg"], width,
           label="Weight", color=PALETTE[0])
    ax.bar(x + width/2, risk_df["pct_cr_avg"], width,
           label="Risk contribution", color=PALETTE[3])
    ax.set_xticks(x)
    ax.set_xticklabels(risk_df.index, rotation=30)
    ax.set_title("Weight vs Risk contribution per asset class")
    ax.set_ylabel("Fraction")
    ax.legend()
    _save(fig, name)


def plot_return_contribution(contrib_df: pd.DataFrame,
                             name="gmp_return_contribution.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    contrib_df["contrib_pct"].sort_values().plot(
        kind="barh", ax=ax, color=PALETTE[2])
    ax.set_title("Contribution to return per asset class")
    ax.set_xlabel("Fraction of total return")
    _save(fig, name)


def plot_yearly_returns(yearly_df: pd.DataFrame,
                        name="gmp_yearly_returns.png"):
    fig, ax = plt.subplots(figsize=(12, 5))
    yearly_df.plot(kind="bar", ax=ax, width=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Annual returns")
    ax.set_ylabel("Return")
    ax.legend()
    _save(fig, name)


def plot_rolling_correlation(corr_series: pd.Series,
                             name="gmp_rolling_correlation.png",
                             window: int = 36):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(corr_series.index, corr_series.values, lw=1.5, color=PALETTE[0])
    ax.axhline(corr_series.mean(), color="red", ls="--", lw=1,
               label=f"Mean = {corr_series.mean():.2f}")
    ax.set_title(f"Mean pairwise correlation ({window}m)")
    ax.set_ylabel("Correlation")
    ax.legend()
    _save(fig, name)