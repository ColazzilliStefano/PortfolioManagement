from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

import pandas as pd
from loguru import logger
from fredapi import Fred

from src.utils.config import project_root


CACHE_DIR = project_root() / "data" / "raw" / "macro"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_fred() -> Fred:
    """
    Returns a singleton FRED client.
    Reads FRED_API_KEY from:
    1. Environment variable
    2. .env file in the project root
    """
    api_key = os.environ.get("FRED_API_KEY")

    if not api_key:
        env_path = project_root() / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("FRED_API_KEY="):
                        api_key = line.strip().split("=", 1)[1]
                        break

    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not found. "
            "Create a .env file with FRED_API_KEY=your_key "
            "or export the environment variable."
        )

    logger.info(f"FRED client initialized (key: ...{api_key[-4:]})")
    return Fred(api_key=api_key)


def fetch_series(series_id: str,
                 start: str = "2000-01-01",
                 end: str | None = None,
                 use_cache: bool = True) -> pd.Series:
    """
    Downloads a FRED series via API.
    Local cache in parquet to avoid repeated calls.

    Returns Series with monthly DatetimeIndex (resampled to month-end).
    """
    cache = CACHE_DIR / f"{series_id}.parquet"

    if use_cache and cache.exists():
        s = pd.read_parquet(cache)["value"]
        logger.debug(f"FRED {series_id}: cache hit ({len(s)} rows)")
    else:
        fred = get_fred()
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        s.name = "value"
        s.to_frame().to_parquet(cache)
        logger.info(f"FRED {series_id}: downloaded ({len(s)} rows)")

    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def fetch_vintage(series_id: str,
                  vintage_date: str,
                  start: str = "2000-01-01") -> pd.Series:
    """
    Downloads an ALFRED (vintage) series via API.
    Returns the series as it was available at the specified date.
    """
    cache = CACHE_DIR / f"{series_id}_vintage_{vintage_date}.parquet"

    if cache.exists():
        s = pd.read_parquet(cache)["value"]
        logger.debug(f"ALFRED {series_id} vintage {vintage_date}: cache hit")
    else:
        fred = get_fred()
        s = fred.get_series(
            series_id,
            observation_start=start,
            realtime_start=vintage_date,
            realtime_end=vintage_date,
        )
        s.name = "value"
        s.to_frame().to_parquet(cache)
        logger.info(f"ALFRED {series_id} vintage {vintage_date}: downloaded")

    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def clear_cache(series_id: str | None = None) -> None:
    """Clears the cache. If series_id is None, deletes everything."""
    if series_id:
        for p in CACHE_DIR.glob(f"{series_id}*.parquet"):
            p.unlink()
    else:
        for p in CACHE_DIR.glob("*.parquet"):
            p.unlink()
    logger.info(f"FRED cache cleared: {series_id or 'everything'}")