import pandas as pd

from src.data.fred_client import fetch_series, fetch_vintage


def fetch_fred(series_id: str, start: str = "2000-01-01",
               vintage_date: str | None = None) -> pd.Series:
    if vintage_date is not None:
        return fetch_vintage(series_id, vintage_date, start=start)
    return fetch_series(series_id, start=start)


def resample_monthly(s: pd.Series) -> pd.Series:
    return s.resample("ME").last().ffill()