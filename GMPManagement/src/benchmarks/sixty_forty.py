from __future__ import annotations
import pandas as pd
from src.analytics.returns import log_to_simple


def backtest_sixty_forty(returns: pd.DataFrame, rebalance: int = 12,
                         return_type: str = "log") -> pd.Series:
    w = pd.Series(0.0, index=returns.columns)
    if "ACWI" in w.index: w["ACWI"] = 0.60
    if "AGG" in w.index: w["AGG"] = 0.40
    simple = log_to_simple(returns) if return_type == "log" else returns
    dates = returns.index
    port = []
    for i in range(0, len(dates), rebalance):
        end = min(i + rebalance, len(dates))
        port.append((simple.iloc[i:end] * w).sum(axis=1))
    return pd.concat(port).sort_index()