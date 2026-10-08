"""Fetch commodities market cap via yfinance (sum of AUM ETFs)."""

from __future__ import annotations

from loguru import logger
from src.data.downloader import fetch_ticker_info

COMMODITY_PROXIES = ["DBC", "GSG", "PDBC", "DJP"]


def fetch_commodities_market_cap(date: str | None = None) -> float | None:
    total = 0.0
    found = []
    for t in COMMODITY_PROXIES:
        info = fetch_ticker_info(t)
        aum = info.get("totalAssets") if info else None
        if aum and aum > 0:
            total += aum
            found.append(f"{t}={aum/1e9:.1f}B")
    if total <= 0:
        logger.warning("No commodities market cap found via yfinance")
        return None
    value_trn = total / 1e12
    logger.info(f"Commodities proxy: {value_trn:.3f} T USD ({', '.join(found)})")
    return value_trn