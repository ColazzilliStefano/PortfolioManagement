"""Market cap gold via yfinance (sum of ETF AUM)."""

from __future__ import annotations
from loguru import logger
from src.data.downloader import fetch_ticker_info

GOLD_PROXIES = ["GLD", "IAU", "GLDM", "SGOL"]


def fetch_gold_market_cap(date: str | None = None) -> float | None:
    total = 0.0
    found = []
    for t in GOLD_PROXIES:
        info = fetch_ticker_info(t)
        aum = info.get("totalAssets") if info else None
        if aum and aum > 0:
            total += aum
            found.append(f"{t}={aum/1e9:.1f}B")
    if total <= 0:
        logger.warning("No gold AUM via yfinance")
        return None
    value_trn = total / 1e12
    logger.info(f"Gold proxy: {value_trn:.3f} T USD ({', '.join(found)})")
    return value_trn