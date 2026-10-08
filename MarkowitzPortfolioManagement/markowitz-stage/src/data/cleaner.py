import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

def clean_prices(prices: pd.DataFrame, min_history_months: int = 120) -> pd.DataFrame:
    prices = prices.copy()
    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.sort_index()
    prices = prices.ffill(limit=3)

    months = prices.resample("ME").last()
    valid = months.columns[months.notna().sum() >= min_history_months]
    dropped = set(prices.columns) - set(valid)
    if dropped:
        logger.warning(f"Ticker dropped for insufficient history: {dropped}")
    prices = prices[valid]

    first_valid = prices.apply(lambda s: s.first_valid_index()).max()
    prices = prices.loc[first_valid:]
    prices = prices.dropna(how="any")

    logger.info(f"Prices puliti: {prices.shape}, da {prices.index.min()} a {prices.index.max()}")
    return prices

def compute_returns(prices: pd.DataFrame, return_type: str = "log") -> pd.DataFrame:
    if return_type == "log":
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        returns = prices.pct_change().dropna()
    return returns

def flag_outliers(returns: pd.DataFrame, sigma: float = 10) -> pd.DataFrame:
    z = (returns - returns.mean()) / returns.std()
    mask = z.abs() > sigma
    outliers = returns.where(mask)
    n = int(mask.sum().sum())
    logger.info(f"Outlier flagged (|z|>{sigma}): {n}")
    return outliers

def to_monthly(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.resample("ME").apply(lambda x: x.sum())