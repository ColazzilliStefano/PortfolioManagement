import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import get_config, get_assets
from src.utils.logger import setup_logger
from src.data.downloader import download_prices


def main():
    cfg = get_config()
    assets = get_assets()
    setup_logger(cfg["logging"]["level"], cfg["logging"]["file"])

    tickers = [a["ticker"] for a in assets["assets"]]

    #add rf
    rf = assets.get("risk_free")
    if rf and rf.get("ticker") not in tickers:
        tickers.append(rf["ticker"])

    print(f"Ticker da scaricare: {tickers}")

    download_prices(
        tickers=tickers,
        start=assets["start_date"],
        end=assets.get("end_date"),
    )


if __name__ == "__main__":
    main()