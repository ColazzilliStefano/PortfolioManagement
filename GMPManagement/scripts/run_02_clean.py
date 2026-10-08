import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.utils.config import get_config, get_assets
from src.utils.logger import setup_logger
from src.data.cleaner import (
    clean_prices, compute_returns, flag_outliers, to_monthly,
    drop_incomplete_month, validate_returns,
)
from src.data.universe import build_metadata, validate_universe
from src.data.loader import save_processed
from src.data.provenance import build_provenance, save_provenance


def main():
    cfg = get_config()
    assets = get_assets()
    setup_logger(cfg["logging"]["level"], cfg["logging"]["file"])

    raw = pd.read_parquet(ROOT / "data/raw/prices/prices_raw.parquet")
    prices = clean_prices(raw, cfg["data"]["min_history_months"])
    returns = compute_returns(prices, cfg["data"]["return_type"])
    outliers = flag_outliers(returns, cfg["data"]["outlier_sigma"])

    returns_m = to_monthly(returns)
    returns_m = drop_incomplete_month(returns_m)
    validate_returns(returns_m, "returns_monthly")

    metadata = build_metadata(assets)
    universe = validate_universe(prices, metadata)

    save_processed(prices, "prices")
    save_processed(returns, "returns")
    save_processed(returns_m, "returns_monthly")
    save_processed(metadata, "metadata")
    save_processed(outliers.dropna(how="all"), "outliers")

    with open(ROOT / "data/processed/universe.yaml", "w") as f:
        yaml.safe_dump(universe, f)

    # provenance
    provenance = build_provenance(assets)
    save_provenance(provenance, ROOT / "data/processed/data_provenance.yaml")

    # report
    out_q = ROOT / "outputs/data_quality"
    out_q.mkdir(parents=True, exist_ok=True)
    returns_m.describe().T.to_csv(out_q / "summary_stats.csv")
    returns_m.isna().sum().to_csv(out_q / "missing_report.csv")


if __name__ == "__main__":
    main()