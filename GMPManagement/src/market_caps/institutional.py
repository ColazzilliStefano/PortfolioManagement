"""Parser for institutional market cap sources.

Free sources do not expose public APIs. Data must be manually
loaded into CSV and placed in data/raw/market_caps/<source>/.

Supported sources:
- MSCI ACWI factsheet     -> data/raw/market_caps/msci/acwi_market_cap.csv
- BIS Debt Securities     -> data/raw/market_caps/bis/debt_securities.csv
- FTSE EPRA Nareit        -> data/raw/market_caps/ftse_epra/reits_market_cap.csv
- CFTC Open Interest      -> data/raw/market_caps/cftc/open_interest.csv
- World Gold Council      -> data/raw/market_caps/wgc/gold_market_cap.csv

"""

from pathlib import Path
import pandas as pd
from loguru import logger

from src.utils.config import project_root


RAW = project_root() / "data" / "raw" / "market_caps"


def _load_csv(rel_path: str) -> pd.DataFrame | None:
    p = RAW / rel_path
    if not p.exists():
        logger.warning(f"File not found: {p}")
        return None
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
    if "market_cap_usd_trn" not in df.columns:
        logger.warning(f"Column market_cap_usd_trn missing in {p}")
        return None
    return df


def _lookup(df: pd.DataFrame | None, date: str) -> float | None:
    """Returns the most recent value <= date (point-in-time)."""
    if df is None or df.empty:
        return None
    subset = df.loc[:date]
    if subset.empty:
        return None
    return float(subset.iloc[-1]["market_cap_usd_trn"])


def fetch_msci_equity(date: str) -> float | None:
    df = _load_csv("msci/acwi_market_cap.csv")
    return _lookup(df, date)


def fetch_bis_bonds(date: str) -> float | None:
    df = _load_csv("bis/debt_securities.csv")
    return _lookup(df, date)


def fetch_ftse_epra_reits(date: str) -> float | None:
    df = _load_csv("ftse_epra/reits_market_cap.csv")
    return _lookup(df, date)


def fetch_cftc_commodities(date: str) -> float | None:
    df = _load_csv("cftc/open_interest.csv")
    return _lookup(df, date)


def fetch_wgc_gold(date: str) -> float | None:
    df = _load_csv("wgc/gold_market_cap.csv")
    return _lookup(df, date)


def fetch_all_institutional(date: str) -> pd.Series:
    """Returns Series with market cap (trillions USD) per asset class."""
    caps = {
        "equity": fetch_msci_equity(date),
        "bonds": fetch_bis_bonds(date),
        "real_estate": fetch_ftse_epra_reits(date),
        "commodities": fetch_cftc_commodities(date),
        "gold": fetch_wgc_gold(date),
    }
    return pd.Series(caps, name=date)