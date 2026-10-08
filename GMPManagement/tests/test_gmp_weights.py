import numpy as np
import pandas as pd

from src.gmp.dynamic_weights import compute_gmp_weights, compute_weights_history


FALLBACK = {
    "equity": 0.4,
    "bonds": 0.4,
    "real_estate": 0.05,
    "commodities": 0.05,
    "gold": 0.1,
}


def test_weights_sum_to_one():
    w = compute_gmp_weights("2024-12-31", FALLBACK)
    assert abs(w.sum() - 1.0) < 1e-8


def test_weights_positive():
    w = compute_gmp_weights("2024-12-31", FALLBACK)
    assert (w >= 0).all()


def test_weights_index():
    w = compute_gmp_weights("2024-12-31", FALLBACK)
    expected = {"equity", "bonds", "real_estate", "commodities", "gold"}
    assert set(w.index) == expected


def test_weights_history_shape():
    df = compute_weights_history(
        "2020-01-01", "2021-12-31", "annual", FALLBACK
    )
    assert len(df) >= 2
    assert df.index.is_monotonic_increasing
    sums = df.sum(axis=1)
    assert (sums.round(8) == 1.0).all()