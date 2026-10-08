"""Fetch equity market cap via yfinance."""

from loguru import logger
from src.data.downloader import fetch_ticker_info

EQUITY_PROXY = "ACWI"


def fetch_equity_market_cap(date: str | None = None) -> float | None:
    info = fetch_ticker_info(EQUITY_PROXY)
    if not info:
        return None
    mc = info.get("marketCap")
    if mc and mc > 0:
        value_trn = mc / 1e12
        logger.info(f"Equity marketCap ({EQUITY_PROXY}): {value_trn:.3f} T USD")
        return value_trn
    aum = info.get("totalAssets")
    logger.warning(
        f"Equity market cap unavailable for {EQUITY_PROXY}; "
        f"ETF AUM ignored ({aum / 1e9:.1f}B) to avoid underestimation. "
        "Use the MSCI ACWI factsheet or a manual figure."
    )
    return None