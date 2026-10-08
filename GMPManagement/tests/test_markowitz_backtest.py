import numpy as np
import pandas as pd

from src.markowitz.backtest import equal_weight, walk_forward
from src.markowitz.efficient_frontier import compute_frontier
from src.markowitz.mean_variance import max_sharpe, portfolio_stats


def test_equal_weight_converts_log_returns_before_aggregation():
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    log_returns = pd.DataFrame(
        {"a": [np.log1p(0.10), np.log1p(0.0)],
         "b": [np.log1p(0.0), np.log1p(0.10)]},
        index=dates,
    )

    result = equal_weight(log_returns)

    assert np.allclose(result.to_numpy(), [0.05, 0.05])


def test_markowitz_walk_forward_returns_are_simple():
    dates = pd.date_range("2015-01-31", periods=8, freq="ME")
    log_returns = pd.DataFrame(
        np.log1p(np.full((8, 2), 0.01)),
        index=dates,
        columns=["a", "b"],
    )

    result, weights = walk_forward(
        log_returns, window=4, rebalance=2, method="min_var"
    )

    assert len(weights) == 2
    assert np.allclose(result.to_numpy(), 0.01)


def test_frontier_respects_maximum_return_weight_constraint():
    mu = pd.Series([0.01, 0.02, 0.03], index=["a", "b", "c"])
    cov = pd.DataFrame(np.eye(3), index=mu.index, columns=mu.index)

    frontier, _ = compute_frontier(
        mu, cov, n_points=10, long_only=True, max_weight=0.5
    )

    assert frontier.index.max() <= 0.025 + 1e-8
    assert frontier.index.max() >= 0.024


def test_capital_allocation_line_passes_through_risk_free_and_tangency():
    mu = pd.Series([0.06, 0.08, 0.10], index=["a", "b", "c"])
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.05, 0.01], [0.00, 0.01, 0.06]],
        index=mu.index,
        columns=mu.index,
    )
    rf = 0.02
    weights = max_sharpe(mu, cov, rf=rf, long_only=True, max_weight=0.50)
    stats = portfolio_stats(weights, mu, cov, rf)

    assert abs(rf + stats["sharpe"] * 0.0 - rf) < 1e-12
    assert abs(rf + stats["sharpe"] * stats["vol"] - stats["return"]) < 1e-10