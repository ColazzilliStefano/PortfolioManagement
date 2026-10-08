import pandas as pd
from pathlib import Path

def main():
    processed = Path("data/processed")

    required = [
        "prices.parquet",
        "returns.parquet",
        "returns_monthly.parquet",
        "metadata.parquet",
        "universe.yaml",
    ]
    for f in required:
        p = processed / f
        assert p.exists(), f"MISSING: {f}"
        print(f"OK  {f}")

    rm = pd.read_parquet(processed / "returns_monthly.parquet")
    assert rm.isna().sum().sum() == 0, "NaN values in monthly returns"
    assert rm.index.is_monotonic_increasing, "Index is not sorted"
    assert not rm.index.duplicated().any(), "Duplicate dates"

    print(f"\nData contract satisfied.")
    print(f"Monthly returns shape: {rm.shape}")
    print(f"Period: {rm.index.min().date()} -> {rm.index.max().date()}")
    print(f"Columns: {list(rm.columns)}")

if __name__ == "__main__":
    main()