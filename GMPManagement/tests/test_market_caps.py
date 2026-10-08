from src.market_caps.aggregator import aggregate_market_caps


def test_aggregate_returns_series():
    s = aggregate_market_caps("2024-12-31")
    assert s is not None
    assert hasattr(s, "index")
    assert set(s.index) == {"equity", "bonds", "real_estate", "commodities", "gold"}