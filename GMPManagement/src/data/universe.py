from __future__ import annotations
import pandas as pd


def build_metadata(assets_cfg: dict) -> pd.DataFrame:
    rows = []
    for a in assets_cfg["assets"]:
        rows.append({
            "ticker": a["ticker"],
            "name": a.get("name"),
            "asset_class": a.get("asset_class"),
            "gmp_category": a.get("gmp_category"),
        })
    rf = assets_cfg.get("risk_free")
    if rf:
        rows.append({
            "ticker": rf["ticker"],
            "name": rf.get("name"),
            "asset_class": rf.get("asset_class"),
            "gmp_category": "cash",
        })
    return pd.DataFrame(rows).set_index("ticker")


def validate_universe(prices: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    tickers = list(prices.columns)
    missing_meta = [t for t in tickers if t not in metadata.index]
    return {
        "tickers": tickers,
        "n_assets": len(tickers),
        "start": str(prices.index.min().date()),
        "end": str(prices.index.max().date()),
        "missing_metadata": missing_meta,
    }