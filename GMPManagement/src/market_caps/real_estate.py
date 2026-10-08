"""Market cap real estate via yfinance (sum of ETF AUM)."""

from __future__ import annotations

from loguru import logger
from src.data.downloader import fetch_ticker_info

REIT_PROXIES = ["VNQ", "RWR", "ICF", "SCHH"]


def fetch_real_estate_market_cap(date: str | None = None) -> float | None:
    total = 0.0
    found = []
    for t in REIT_PROXIES:
        info = fetch_ticker_info(t)
        aum = info.get("totalAssets") if info else None
        if aum and aum > 0:
            total += aum
            found.append(f"{t}={aum/1e9:.1f}B")
    if total <= 0:
        logger.warning("No AUM REIT via yfinance")
        return None
    value_trn = total / 1e12
    logger.info(f"Real estate proxy: {value_trn:.3f} T USD ({', '.join(found)})")
    return value_trn