import numpy as np
import pandas as pd
from loguru import logger


def clean_prices(prices: pd.DataFrame, min_history_months: int = 120) -> pd.DataFrame:
    prices = prices.copy()
    prices = prices[~prices.index.duplicated(keep="first")].sort_index()
    prices = prices.ffill(limit=3)

    months = prices.resample("ME").last()
    valid = months.columns[months.notna().sum() >= min_history_months]
    dropped = set(prices.columns) - set(valid)
    if dropped:
        logger.warning(f"Tickers excluded for insufficient history: {dropped}")
    prices = prices[valid]

    first_valid = prices.apply(lambda s: s.first_valid_index()).max()
    prices = prices.loc[first_valid:].dropna(how="any")

    logger.info(f"Cleaned prices: {prices.shape}, from {prices.index.min()} to {prices.index.max()}")
    return prices


def compute_returns(prices: pd.DataFrame, return_type: str = "log") -> pd.DataFrame:
    if return_type == "log":
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def flag_outliers(returns: pd.DataFrame, sigma: float = 10) -> pd.DataFrame:
    z = (returns - returns.mean()) / returns.std()
    mask = z.abs() > sigma
    n = int(mask.sum().sum())
    logger.info(f"Outliers flagged (|z|>{sigma}): {n}")
    return returns.where(mask)


def to_monthly(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.resample("ME").apply(lambda s: s.sum())


def drop_incomplete_month(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Removes the last month if not complete.
    If today is 2026-09-18 and the last data point is 2026-09-30, the month is incomplete.
    """
    if len(returns) == 0:
        return returns
    today = pd.Timestamp.today().normalize()
    last = returns.index[-1]
    if last.year == today.year and last.month == today.month:
        logger.info(f"Removed incomplete month: {last.date()} (today {today.date()})")
        return returns.iloc[:-1]
    return returns


def validate_returns(returns: pd.DataFrame, name: str = "returns") -> None:
    """Sanity checks on returns."""
    assert not returns.empty, f"{name}: empty"
    assert returns.index.is_monotonic_increasing, f"{name}: index not monotonic"
    assert not returns.index.duplicated().any(), f"{name}: duplicate dates"
    assert not returns.isna().any().any(), f"{name}: NaN present"
    logger.info(f"{name}: validated ({returns.shape})")