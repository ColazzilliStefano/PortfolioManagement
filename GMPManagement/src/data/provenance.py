from pathlib import Path
import yaml
from loguru import logger


# Mapping series -> type and source
SERIES_PROVENANCE = {
    "ACWI": {"type": "real", "source": "yfinance", "instrument": "ETF"},
    "AGG":  {"type": "real", "source": "yfinance", "instrument": "ETF"},
    "TLT":  {"type": "real", "source": "yfinance", "instrument": "ETF"},
    "LQD":  {"type": "real", "source": "yfinance", "instrument": "ETF"},
    "HYG":  {"type": "real", "source": "yfinance", "instrument": "ETF"},
    "VNQ":  {"type": "real", "source": "yfinance", "instrument": "ETF"},
    "DBC":  {"type": "real", "source": "yfinance", "instrument": "ETF"},
    "GLD":  {"type": "real", "source": "yfinance", "instrument": "ETF"},
    "BIL":  {"type": "real", "source": "yfinance", "instrument": "ETF"},
}

GMP_WEIGHTS_PROVENANCE = {
    "equity": {
        "type": "institutional",
        "source": "MSCI ACWI factsheet",
        "frequency": "annual",
        "metric": "free-float market cap",
        "caveat": "Annual update, not point-in-time",
    },
    "bonds": {
        "type": "institutional",
        "source": "BIS Debt Securities Statistics",
        "frequency": "quarterly",
        "metric": "debt outstanding",
        "caveat": "Includes non-investable debt (household, financial)",
    },
    "real_estate": {
        "type": "institutional",
        "source": "FTSE EPRA Nareit",
        "frequency": "annual",
        "metric": "listed REITs market cap",
        "caveat": "Listed REITs only, no private real estate",
    },
    "commodities": {
        "type": "proxy",
        "source": "CFTC Commitments of Traders",
        "frequency": "monthly",
        "metric": "futures open interest",
        "caveat": "Open interest is NOT market cap",
    },
    "gold": {
        "type": "institutional",
        "source": "World Gold Council",
        "frequency": "annual",
        "metric": "above-ground stock * price",
        "caveat": "Physical stock, not flow",
    },
}

MACRO_PROVENANCE = {
    "cpi": {
        "type": "real",
        "source": "FRED CPIAUCSL",
        "caveat": "Revised series, not point-in-time (ALFRED would be preferable)",
    },
    "yield_10y": {
        "type": "real",
        "source": "FRED DGS10",
        "caveat": "Revised series",
    },
    "yield_2y": {
        "type": "real",
        "source": "FRED DGS2",
        "caveat": "Revised series",
    },
}


def build_provenance(assets_cfg: dict) -> dict:
    """Builds the complete provenance dict."""
    provenance = {
        "market_data": {},
        "gmp_weights": GMP_WEIGHTS_PROVENANCE,
        "macro": MACRO_PROVENANCE,
        "legend": {
            "real": "Real data (price, published series)",
            "proxy": "Approximate substitute",
            "institutional": "Aggregated institutional source",
            "estimated": "Estimate from literature",
        },
    }

    for asset in assets_cfg["assets"]:
        t = asset["ticker"]
        provenance["market_data"][t] = SERIES_PROVENANCE.get(
            t, {"type": "unknown", "source": "unknown"}
        )

    rf = assets_cfg.get("risk_free")
    if rf:
        provenance["market_data"][rf["ticker"]] = SERIES_PROVENANCE.get(
            rf["ticker"], {"type": "unknown", "source": "unknown"}
        )

    return provenance


def save_provenance(provenance: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(provenance, f, sort_keys=False)
    logger.info(f"Provenance saved to {path}")