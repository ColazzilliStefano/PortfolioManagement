import pandas as pd

from src.gmp.dynamic_weights import compute_weights_history


def test_weights_history_shape():
    fallback = {
        "equity": 0.4, "bonds": 0.4, "real_estate": 0.05,
        "commodities": 0.05, "gold": 0.1,
    }
    df = compute_weights_history("2020-01-01", "2021-12-31", "annual", fallback)
    assert len(df) >= 2
    assert df.index.is_monotonic_increasing
    # weights sum to 1 for each row
    sums = df.sum(axis=1)
    assert (sums.round(8) == 1.0).all()