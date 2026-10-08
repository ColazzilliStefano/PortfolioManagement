"""
Dynamically recalculates the GMP weights via the yfinance API.
No CSV. Configurable frequency: monthly, semiannual, annual.

Limitations (documented):
- yfinance provides ETF AUM, not asset class market cap.
- The calculation changes only when the ETF AUMs change,
  so it is a rough proxy but freely updatable.
- In production: Bloomberg, Refinitiv, BIS, MSCI, WGC.

Dynamic GMP weight calculation from institutional sources, with fallback tracking."""

from datetime import datetime
from pathlib import Path
import pandas as pd
from loguru import logger

from src.utils.config import get_gmp_config, project_root
from src.market_caps.institutional import fetch_all_institutional


def compute_gmp_weights(date: str, fallback: dict,
                        fallback_log: list | None = None) -> pd.Series:
    caps = fetch_all_institutional(date)

    for k, v in caps.items():
        if v is None or pd.isna(v) or v <= 0:
            logger.warning(f"{k} unavailable for {date}; using fallback")
            if fallback_log is not None:
                fallback_log.append({"date": date, "asset": k, "reason": "missing"})
            caps[k] = fallback.get(k, 0.0)

    total = caps.sum()
    if total <= 0:
        raise ValueError(f"Sum of market caps = 0 for {date}")

    w = caps / total
    w.name = date
    return w


def compute_weights_history(start_date: str, end_date: str,
                            frequency: str, fallback: dict):
    freq_map = {"monthly": "ME", "semiannual": "6ME", "annual": "YE"}
    freq = freq_map.get(frequency, "6ME")
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    fallback_log: list = []
    records = []
    for d in dates:
        w = compute_gmp_weights(d.strftime("%Y-%m-%d"), fallback, fallback_log)
        records.append(w)

    df = pd.DataFrame(records)
    idx = pd.to_datetime(df.index).tz_localize(None).astype("datetime64[ns]")
    df.index = idx
    df.index.name = "date"

    if fallback_log:
        fb_df = pd.DataFrame(fallback_log)
        fb_path = project_root() / "outputs/data_quality/fallback_report.csv"
        fb_path.parent.mkdir(parents=True, exist_ok=True)
        fb_df.to_csv(fb_path, index=False)
        logger.info(f"Fallback report saved: {fb_path} ({len(fallback_log)} events)")

    return df


def save_weights(df: pd.DataFrame, parquet_path: str, csv_path: str) -> None:
    p1 = project_root() / parquet_path
    p2 = project_root() / csv_path
    p1.parent.mkdir(parents=True, exist_ok=True)
    p2.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p1)
    df.to_csv(p2)
    logger.info(f"Saved: {p1}")
    logger.info(f"Saved: {p2}")
    logger.info(f"Latest weights:\n{df.tail()}")


def main(frequency: str | None = None):
    cfg = get_gmp_config()
    freq = frequency or cfg["rebalance_frequency"]
    start = cfg["start_date"]
    end = cfg.get("end_date") or datetime.today().strftime("%Y-%m-%d")

    logger.info(f"GMP recomputation: frequency={freq}, start={start}, end={end}")
    df = compute_weights_history(start, end, freq, cfg["fallback_weights"])
    save_weights(df, cfg["output_file"], cfg["output_csv"])


if __name__ == "__main__":
    main()