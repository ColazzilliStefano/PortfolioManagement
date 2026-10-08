from __future__ import annotations

import numpy as np


def log_to_simple(returns):
	return np.expm1(returns)


def simple_to_log(returns):
	return np.log1p(returns)
