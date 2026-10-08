import pandas as pd
import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _require_generated_file(relative_path: str) -> Path:
    path = ROOT / relative_path
    if not path.exists():
        pytest.skip(
            f"Pipeline artifact missing: {relative_path}. "
            "Run the data pipeline first."
        )
    return path


def test_weights_sum_to_one_every_row():
    p = _require_generated_file("data/processed/gmp_weights_history.parquet")
    df = pd.read_parquet(p)
    sums = df.sum(axis=1)
    np.testing.assert_allclose(sums.values, np.ones(len(df)), atol=1e-8)


def test_returns_no_nan():
    p = _require_generated_file("data/processed/returns_monthly.parquet")
    df = pd.read_parquet(p)
    assert df.isna().sum().sum() == 0


def test_bl_returns_positive_vol():
    p = _require_generated_file("outputs/tables/final_comparison.csv")
    df = pd.read_csv(p).set_index("name")
    for name in df.index:
        assert df.loc[name, "Vol"] > 0, f"{name} has null vol"


def test_bl_dynamic_has_better_sharpe_than_all_weather():
    p = _require_generated_file("outputs/tables/final_comparison.csv")
    df = pd.read_csv(p).set_index("name")
    assert df.loc["BL dynamics", "Sharpe"] > df.loc["All Weather", "Sharpe"]


def test_bl_dynamic_turnover_reasonable():
    """Average turnover BL dynamics < 20% per rebalancing."""
    p = ROOT / "outputs/tables/bl_dynamic_turnover.csv"
    if not p.exists():
        return  # skip if file not present
    df = pd.read_csv(p, index_col=0)
    assert df["turnover"].mean() < 0.20