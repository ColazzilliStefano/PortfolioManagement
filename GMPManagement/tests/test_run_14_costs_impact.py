from __future__ import annotations

import pandas as pd

from src.analytics.performance import sharpe_ratio
from src.signals.engine import normalize_index


def test_run_14_risk_free_index_is_aligned() -> None:
    portfolio = pd.Series(
        [0.01, 0.02],
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )
    risk_free_with_timezone = pd.Series(
        [0.002, 0.002],
        index=pd.to_datetime(
            ["2020-01-31", "2020-02-29"], utc=True
        ),
    )

    aligned = normalize_index(risk_free_with_timezone)

    assert aligned.index.equals(portfolio.index)
    assert sharpe_ratio(portfolio, rf_series=aligned) != sharpe_ratio(
        portfolio
    )
