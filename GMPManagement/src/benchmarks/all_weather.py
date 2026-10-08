from __future__ import annotations
import pandas as pd
from src.analytics.returns import log_to_simple


def backtest_all_weather(returns: pd.DataFrame, rebalance: int = 12,
                         return_type: str = "log") -> pd.Series:
    w = pd.Series(0.0, index=returns.columns)
    target = {"ACWI": 0.30, "TLT": 0.40, "LQD": 0.15, "GLD": 0.075, "DBC": 0.075}
    for k, v in target.items():
        if k in w.index:
            w[k] = v
    simple = log_to_simple(returns) if return_type == "log" else returns
    dates = returns.index
    port = []
    for i in range(0, len(dates), rebalance):
        end = min(i + rebalance, len(dates))
        port.append((simple.iloc[i:end] * w).sum(axis=1))
    return pd.concat(port).sort_index()