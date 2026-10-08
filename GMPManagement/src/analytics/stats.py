from __future__ import annotations
import pandas as pd


def mean_returns(returns: pd.DataFrame, freq: int = 12) -> pd.Series:
    return returns.mean() * freq


def cov_matrix(returns: pd.DataFrame, freq: int = 12) -> pd.DataFrame:
    return returns.cov() * freq


def corr_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()