import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_config, get_assets
from src.utils.logger import setup_logger
from src.data.cleaner import clean_prices, compute_returns, flag_outliers, to_monthly
from src.data.universe import build_metadata, validate_universe
from src.data.loader import save_processed
import yaml

def main():
    cfg = get_config()
    assets = get_assets()
    log = setup_logger(cfg["logging"]["level"], cfg["logging"]["file"])

    expected = [a["ticker"] for a in assets["assets"]]
    rf = assets.get("risk_free", {}).get("ticker")
    if rf and rf not in expected:
        expected.append(rf)

    raw_path = PROJECT_ROOT / cfg["paths"]["raw"] / "prices_raw.parquet"
    raw = pd.read_parquet(raw_path)
    missing = [ticker for ticker in expected if ticker not in raw.columns]
    if missing:
        raise ValueError(f"Configured tickers missing from raw data: {missing}")
    raw = raw.loc[:, expected]
    prices = clean_prices(raw, cfg["data"]["min_history_months"])
    returns = compute_returns(prices, cfg["data"]["return_type"])
    outliers = flag_outliers(returns, cfg["data"]["outlier_sigma"])
    returns_m = to_monthly(returns)

    metadata = build_metadata(assets)
    universe = validate_universe(prices, metadata)

    save_processed(prices, "prices")
    save_processed(returns, "returns")
    save_processed(returns_m, "returns_monthly")
    save_processed(metadata, "metadata")
    save_processed(outliers.dropna(how="all"), "outliers")

    processed_dir = PROJECT_ROOT / cfg["paths"]["processed"]
    with open(processed_dir / "universe.yaml", "w") as f:
        yaml.safe_dump(universe, f)


    quality_dir = PROJECT_ROOT / cfg["paths"]["outputs"] / "data_quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    returns_m.describe().T.to_csv(quality_dir / "summary_stats.csv")
    returns_m.isna().sum().to_csv(quality_dir / "missing_report.csv")

    log.info(f"Universe: {universe}")

if __name__ == "__main__":
    main()