"""Loader for the Shiller dataset (CAPE, P/E, earnings yield S&P 500).

Source: Robert Shiller, Yale. Public dataset downloadable as Excel.
CAPE (Cyclically Adjusted P/E) is the average P/E over 10 years, corrected
for inflation. It is the reference for the value signal.
"""

from pathlib import Path
import pandas as pd
import requests
from loguru import logger

from src.utils.config import project_root


CACHE_DIR = project_root() / "data" / "raw" / "macro"
SHILLER_URL = (
    "https://img1.wsimg.com/blobby/go/"
    "e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/ie_data.xls"
)


def fetch_shiller_cape(force: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "shiller_ie_data.xls"

    if force or not path.exists():
        try:
            r = requests.get(SHILLER_URL, timeout=30)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            logger.info(f"Shiller dataset downloaded: {path.stat().st_size} bytes")
        except Exception as e:
            if path.exists():
                logger.warning(f"Shiller: using cache ({e})")
            else:
                logger.warning(f"Shiller unavailable: {e}")
                return pd.DataFrame()

    try:
        raw = pd.read_excel(path, sheet_name="Data", skiprows=7)
    except Exception as e:
        logger.warning(f"Shiller parsing failed: {e}")
        return pd.DataFrame()

    # normalize columns
    df = raw.rename(columns={"Date": "date", "CAPE": "cape"}).copy()
    df = df[["date", "cape"]].dropna()

    df["date"] = (
        df["date"].astype(str).str.replace(".", "-", regex=False).str[:7]
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df.resample("ME").last().ffill()
    df["earnings_yield"] = 1.0 / df["cape"].replace(0, pd.NA)

    logger.info(f"Shiller CAPE: {len(df)} observations "
                f"({df.index[0].date()} -> {df.index[-1].date()})")
    return df