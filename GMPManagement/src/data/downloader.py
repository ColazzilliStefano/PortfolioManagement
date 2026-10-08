from __future__ import annotations
import pandas as pd
import yfinance as yf
from pathlib import Path
from loguru import logger


def download_prices(tickers: list[str], start: str, end: str | None = None,
                    out_dir: str = "data/raw/prices") -> pd.DataFrame:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Download {len(tickers)} tickers from {start}: {tickers}")

    df = yf.download(
        tickers=tickers, start=start, end=end,
        auto_adjust=True, progress=False,
        group_by="ticker", threads=True,
    )

    if isinstance(df.columns, pd.MultiIndex):
        prices = pd.concat(
            {t: df[t]["Close"] for t in tickers if t in df.columns.levels[0]},
            axis=1,
        )
    else:
        prices = df[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index = pd.to_datetime(prices.index, utc=True)
    prices = prices.sort_index().dropna(axis=1, how="all")

    prices.to_parquet(Path(out_dir) / "prices_raw.parquet")
    logger.info(f"Saved: {prices.shape} -> {list(prices.columns)}")
    return prices


def fetch_ticker_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return info or {}
    except Exception as e:
        logger.warning(f"yfinance info failed for {ticker}: {e}")
        return {}