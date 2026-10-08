import pandas as pd
from pathlib import Path

def build_metadata(assets_cfg: dict) -> pd.DataFrame:
    rows = []
    configured_assets = list(assets_cfg["assets"])
    risk_free = assets_cfg.get("risk_free")
    if risk_free:
        configured_assets.append(risk_free)

    for a in configured_assets:
        rows.append({
            "ticker": a["ticker"],
            "name": a.get("name"),
            "asset_class": a.get("asset_class"),
            "currency": a.get("currency"),
            "source": a.get("source"),
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