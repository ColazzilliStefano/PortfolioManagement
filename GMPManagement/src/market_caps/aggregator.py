"""Aggregate market caps from different asset classes."""

import pandas as pd
from loguru import logger

from src.market_caps.equity import fetch_equity_market_cap
from src.market_caps.bonds import fetch_bonds_market_cap
from src.market_caps.real_estate import fetch_real_estate_market_cap
from src.market_caps.commodities import fetch_commodities_market_cap
from src.market_caps.gold import fetch_gold_market_cap


def aggregate_market_caps(date: str | None = None) -> pd.Series:
    caps = {
        "equity": fetch_equity_market_cap(date),
        "bonds": fetch_bonds_market_cap(date),
        "real_estate": fetch_real_estate_market_cap(date),
        "commodities": fetch_commodities_market_cap(date),
        "gold": fetch_gold_market_cap(date),
    }
    s = pd.Series(caps, name=date or "latest")
    logger.info(f"Market caps aggregate ({date}): "
                f"{ {k: round(v,3) if v else None for k,v in caps.items()} }")
    return s