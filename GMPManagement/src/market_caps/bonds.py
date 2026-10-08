"""Fetch bonds market cap via yfinance (sum of AUM ETFs)."""

from loguru import logger
from src.data.downloader import fetch_ticker_info

BOND_PROXIES = ["AGG", "BND", "BNDW", "TLT", "LQD", "HYG"]


def fetch_bonds_market_cap(date: str | None = None) -> float | None:
    total = 0.0
    found = []
    for t in BOND_PROXIES:
        info = fetch_ticker_info(t)
        aum = info.get("totalAssets") if info else None
        if aum and aum > 0:
            total += aum
            found.append(f"{t}={aum/1e9:.1f}B")
    if total <= 0:
        logger.warning("No bonds market cap found via yfinance")
        return None
    value_trn = total / 1e12
    logger.info(f"Bonds proxy: {value_trn:.3f} T USD ({', '.join(found)})")
    return value_trn