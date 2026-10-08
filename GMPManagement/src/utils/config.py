"""YAML configuration loading."""

from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, "r") as f:
        return yaml.safe_load(f)


def get_config() -> dict:
    """Master configuration (config/config.yaml)."""
    return load_yaml("config/config.yaml")


def get_assets() -> dict:
    """Asset universe and ticker -> asset class mapping."""
    return load_yaml("config/assets.yaml")


def get_gmp_config() -> dict:
    """GMP configuration (dynamic weights, sources)."""
    return load_yaml("config/gmp_dynamic.yaml")


def project_root() -> Path:
    return ROOT


# Specific helpers for config sections

def get_signals_config() -> dict:
    """Full signals section."""
    return get_config().get("signals", {})


def get_active_signals() -> list[str]:
    """List of active signals."""
    return get_config().get("signals", {}).get("active", [])


def get_signal_params(signal_name: str) -> dict:
    """Parameters of a single signal."""
    return get_config().get("signals", {}).get(signal_name, {})


def get_bl_config() -> dict:
    """Black-Litterman parameters."""
    return get_config().get("bl", {})


def get_views_config() -> dict:
    """Fixed views for baseline."""
    return get_config().get("views", {})