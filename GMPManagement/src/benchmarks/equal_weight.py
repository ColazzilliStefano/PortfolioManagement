from __future__ import annotations
import pandas as pd
from src.analytics.returns import log_to_simple


def backtest_equal_weight(returns: pd.DataFrame, rebalance: int = 12,
                          return_type: str = "log") -> pd.Series:
    w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    simple = log_to_simple(returns) if return_type == "log" else returns
    dates = returns.index
    port = []
    for i in range(0, len(dates), rebalance):
        end = min(i + rebalance, len(dates))
        port.append((simple.iloc[i:end] * w).sum(axis=1))
    return pd.concat(port).sort_index()