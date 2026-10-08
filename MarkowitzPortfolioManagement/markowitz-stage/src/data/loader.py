import pandas as pd
from pathlib import Path
from src.utils.config import get_config

def load_processed(name: str) -> pd.DataFrame:
    cfg = get_config()
    path = Path(cfg["paths"]["processed"]) / f"{name}.parquet"
    return pd.read_parquet(path)

def save_processed(df: pd.DataFrame, name: str) -> None:
    cfg = get_config()
    path = Path(cfg["paths"]["processed"]) / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)