from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.utils.config import project_root


def _path(name: str) -> Path:
    return project_root() / "data" / "processed" / f"{name}.parquet"


def save_processed(df: pd.DataFrame, name: str) -> None:
    p = _path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(_path(name))