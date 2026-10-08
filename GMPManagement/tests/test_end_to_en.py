import pandas as pd
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


def test_gmp_weights_history_exists():
    p = _require_generated_file("data/processed/gmp_weights_history.parquet")
    df = pd.read_parquet(p)
    assert len(df) > 0
    expected_cols = {"equity", "bonds", "real_estate", "commodities", "gold"}
    assert expected_cols.issubset(set(df.columns)), \
    f"Missing columns: {expected_cols - set(df.columns)}"
    sums = df.sum(axis=1)
    assert (sums.round(6) == 1.0).all()


def test_gmp_returns_exists():
    p = _require_generated_file("data/processed/gmp_returns.parquet")
    df = pd.read_parquet(p)
    assert len(df) > 100


def test_final_comparison_has_8_strategies():
    p = _require_generated_file("outputs/tables/final_comparison.csv")
    df = pd.read_csv(p)
    assert len(df) == 8
    expected = {
        "BL dynamics", "BL fixed", "GMP", "Equal Weight", "60/40",
        "All Weather", "Markowitz Max Sharpe", "Markowitz Min Variance",
    }
    assert set(df["name"]) == expected
    # all metrics present and finite
    for col in ["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar"]:
        assert df[col].notna().all()


def test_bl_dynamic_beats_gmp_on_sharpe():
    """Quantitative check: BL dynamics beats GMP on Sharpe."""
    p = _require_generated_file("outputs/tables/final_comparison.csv")
    df = pd.read_csv(p).set_index("name")
    assert df.loc["BL dynamics", "Sharpe"] > df.loc["GMP", "Sharpe"]


def test_no_empty_files_in_critical_dirs():
    """Check that src/ and scripts/ do not contain empty files."""
    for d in ["src", "scripts"]:
        for f in (ROOT / d).rglob("*.py"):
            if f.name == "__init__.py":
                continue
            assert f.stat().st_size > 0, f"Empty file: {f}"