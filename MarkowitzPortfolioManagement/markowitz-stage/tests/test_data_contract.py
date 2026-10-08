import numpy as np
import pandas as pd
import pytest

from src.analytics.performance import max_drawdown, summary


def test_max_drawdown_uses_simple_returns():
	returns = pd.Series([0.10, -0.10])

	assert max_drawdown(returns) == pytest.approx(-0.10)


def test_summary_uses_monthly_risk_free_series():
	returns = pd.Series([0.01, 0.02, 0.00])

	rf = pd.Series(0.01, index=returns.index)
	result = summary(returns, rf_series=rf)

	assert result["Sharpe"] < summary(returns)["Sharpe"]
