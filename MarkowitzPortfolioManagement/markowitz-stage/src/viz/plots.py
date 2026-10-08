from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = sns.color_palette("deep")

def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)

def plot_frontier_random(random_df, name="frontier_random.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(random_df["vol"], random_df["return"],
                    c=random_df["sharpe"], cmap="viridis", s=8, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Sharpe")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Expected annualized return")
    ax.set_title("Random portfolio frontier")
    _save(fig, name)

def plot_frontier_efficient(frontier, tangency=None, minvar=None,
                            name="frontier_efficient.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frontier["vol"], frontier["return"], "-o", ms=3,
            color=PALETTE[0], label="Efficient frontier")
    if tangency is not None:
        ax.scatter(tangency["vol"], tangency["return"], s=140, marker="*",
                   color="red", zorder=5, label="Max Sharpe (tangency)")
    if minvar is not None:
        ax.scatter(minvar["vol"], minvar["return"], s=100, marker="s",
               color="black", zorder=5, label="Minimum variance")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Expected annualized return")
    ax.set_title("Markowitz efficient frontier")
    ax.legend()
    _save(fig, name)

def plot_frontier_markowitz(random_df, frontier, tangency=None, minvar=None,
                            name="frontier_markowitz.png"):
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(random_df["vol"], random_df["return"],
                    c=random_df["sharpe"], cmap="viridis", s=8,
                    alpha=0.35, label="Random portfolios")
    fig.colorbar(sc, ax=ax, label="Sharpe")
    ax.plot(frontier["vol"], frontier["return"], "-o", ms=3,
            color="black", lw=2, label="Efficient frontier")
    if tangency is not None:
        ax.scatter(tangency["vol"], tangency["return"], s=170,
                   marker="*", color="red", zorder=5,
                   label="Max Sharpe (tangency)")
    if minvar is not None:
        ax.scatter(minvar["vol"], minvar["return"], s=110,
               marker="s", color="white", edgecolor="black",
               zorder=5, label="Minimum variance")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Expected annualized return")
    ax.set_title("Portfolio space and efficient frontier")
    ax.legend()
    _save(fig, name)

def plot_cal(frontier, tangency, rf=0.0, name="capital_allocation_line.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frontier["vol"], frontier["return"], "-",
            color=PALETTE[0], label="Frontier")
    xs = np.linspace(0, frontier["vol"].max() * 1.05, 50)
    slope = (tangency["return"] - rf) / tangency["vol"]
    ax.plot(xs, rf + slope * xs, "--", color="red", label="CAL")
    ax.scatter(tangency["vol"], tangency["return"], s=140, marker="*",
               color="red", zorder=5, label="Tangency")
    ax.scatter(0, rf, s=80, marker="o", color="black", label="Risk-free")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    ax.set_title("Capital Allocation Line")
    ax.legend()
    _save(fig, name)

def plot_weights(w, name="weights_tangency.png", title="Optimal portfolio weights"):
    w_sorted = w.sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(w_sorted))))
    colors = ["#d62728" if v < 0 else PALETTE[0] for v in w_sorted.values]
    ax.barh(w_sorted.index, w_sorted.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Weight")
    ax.set_title(title)
    _save(fig, name)

def plot_weights_instability(weights_df, name="weights_instability.png",
                             top_n=6):
    cols = weights_df.abs().mean().sort_values(ascending=False).head(top_n).index
    fig, ax = plt.subplots(figsize=(11, 5))
    for c in cols:
        ax.plot(weights_df.index, weights_df[c], label=c, lw=1.5)
    ax.set_title(f"Weight instability over time (top {top_n})")
    ax.set_ylabel("Weight")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, name)

def plot_equity_curves(curves: dict, name="equity_curves.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, r in curves.items():
        eq = (1.0 + r.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, label=label, lw=1.6)
    ax.set_yscale("log")
    ax.set_title("Equity curve (log scale)")
    ax.set_ylabel("Capital")
    ax.legend()
    _save(fig, name)

def plot_drawdowns(curves: dict, name="drawdowns.png"):
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, r in curves.items():
        eq = (1.0 + r.fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values, label=label, lw=1.3)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.legend()
    _save(fig, name)

def plot_rolling_sharpe(curves: dict, window=24, rf=0.0,
                        rf_series=None,
                        name="rolling_sharpe.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, r in curves.items():
        if rf_series is None:
            excess = r - rf / 12
        else:
            excess = r - rf_series.reindex(r.index).fillna(0.0)
        rs = (excess.rolling(window).mean() /
              excess.rolling(window).std()) * np.sqrt(12)
        ax.plot(rs.index, rs.values, label=label, lw=1.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Rolling Sharpe ({window}m)")
    ax.legend()
    _save(fig, name)

def plot_correlation(corr, name="correlation_heatmap.png"):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation matrix")
    _save(fig, name)

def plot_estimation_error(df, name="estimation_error.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["in_sample_sharpe"], df["out_of_sample_sharpe"],
               alpha=0.5, s=14)
    lims = [min(df.min()), max(df.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("Sharpe in-sample")
    ax.set_ylabel("Sharpe out-of-sample")
    ax.set_title("Estimation error: in-sample vs out-of-sample")
    _save(fig, name)

def plot_turnover(turnover, name="turnover.png"):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(turnover.index, turnover.values, width=20, color=PALETTE[0])
    ax.set_title("Turnover at rebalance")
    ax.set_ylabel("Turnover")
    _save(fig, name)

def plot_insample_vs_oos(summary_df, name="insample_vs_oos.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    summary_df.plot(kind="bar", ax=ax)
    ax.set_title("Performance in-sample vs out-of-sample")
    ax.set_ylabel("Sharpe")
    _save(fig, name)