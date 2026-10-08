import pandas as pd
import pytest

from src.costs.transaction_costs import (
    compute_cost_for_rebalance, apply_costs_to_returns,
    DEFAULT_COSTS_BPS,
)

def test_cost_zero_if_no_turnover():
    w = pd.Series({"equity": 0.5, "bonds": 0.5})
    cost = compute_cost_for_rebalance(w, w)
    assert cost == 0.0

def test_cost_positive_if_turnover():
    w_prev = pd.Series({"equity": 0.5, "bonds": 0.5})
    w_target = pd.Series({"equity": 0.6, "bonds": 0.4})
    cost = compute_cost_for_rebalance(w_prev, w_target)
    # turnover 0.1 + 0.1 = 0.2 total, mix of 5 and 10 bps
    assert cost > 0
    assert cost < 0.001  # < 10 bps

    return


def test_apply_costs_reduces_returns():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    port = pd.Series([0.01] * 6, index=dates)
    weights_hist = pd.DataFrame(
        {"equity": [0.5, 0.6], "bonds": [0.5, 0.4]},
        index=[dates[0], dates[3]],
    )
    port_net = apply_costs_to_returns(port, weights_hist)
    assert port_net.sum() < port.sum()