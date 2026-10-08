from src.data.downloader import fetch_ticker_info

def test_fetch_ticker_info_returns_dict():
    """Verifica che la function returns un dict, to theso if empty."""
    info = fetch_ticker_info("NONEXISTENT_TICKER_XYZ123")
    assert isinstance(info, dict)