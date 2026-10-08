"""Transaction cost model for ETFs and rebalancings.

Costs estimated in basis points (bps) per unit of turnover:
- Equity ETF: 5 bps
- Bond ETF: 10 bps
- Gold ETF: 10 bps
- Commodities ETF: 15 bps
- Real estate ETF: 12 bps

The cost for rebalancing is:
    cost = sum_i |w_target_i - w_prev_i| * bps_i / 10000

Applied to the return of the following period.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger


# Costs per asset class in basis points
DEFAULT_COSTS_BPS = {
    "equity": 5.0,
    "bonds": 10.0,
    "real_estate": 12.0,
    "commodities": 15.0,
    "gold": 10.0,
}


def compute_turnover_per_asset(weights_prev: pd.Series,
                                weights_target: pd.Series) -> pd.Series:
    """Turnover per asset class: |w_target - w_prev|."""
    return (weights_target - weights_prev).abs()


def compute_cost_for_rebalance(weights_prev: pd.Series,
                                weights_target: pd.Series,
                                cost_bps: dict | None = None) -> float:
    if cost_bps is None:
        cost_bps = DEFAULT_COSTS_BPS

    turn = compute_turnover_per_asset(weights_prev, weights_target)
    cost = 0.0
    for asset, t in turn.items():
        bps = cost_bps.get(asset, 10.0)
        cost += t * bps / 10000.0
    return float(cost)


def apply_costs_to_returns(port_returns: pd.Series,
                            weights_history: pd.DataFrame,
                            cost_bps: dict | None = None) -> pd.Series:

    if cost_bps is None:
        cost_bps = DEFAULT_COSTS_BPS

    #normalize indices
    port_net = port_returns.copy()
    idx = pd.to_datetime(port_net.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    port_net.index = idx.astype("datetime64[ns]")

    w = weights_history.copy()
    idx_w = pd.to_datetime(w.index)
    if getattr(idx_w, "tz", None) is not None:
        idx_w = idx_w.tz_localize(None)
    w.index = idx_w.astype("datetime64[ns]")

    if len(w) == 0 or len(port_net) == 0:
        return port_net

    dates = w.index.sort_values()
    prev = None

    for i, date in enumerate(dates):
        w_target = w.loc[date]
        if prev is None:
            cost = compute_cost_for_rebalance(
                pd.Series(0.0, index=w_target.index), w_target, cost_bps
            )
        else:
            cost = compute_cost_for_rebalance(prev, w_target, cost_bps)

        # find the first return available after date
        next_date = dates[i + 1] if i < len(dates) - 1 else port_net.index[-1]

        mask = (port_net.index > date) & (port_net.index <= next_date)
        if mask.any():
            idx_first = port_net.index[mask][0]
            port_net.loc[idx_first] = port_net.loc[idx_first] - cost

        prev = w_target

    return port_net