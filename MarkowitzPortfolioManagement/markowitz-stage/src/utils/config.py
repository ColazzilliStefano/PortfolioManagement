from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_yaml(path: str) -> dict:
    with open(ROOT / path, "r") as f:
        return yaml.safe_load(f)

def get_config() -> dict:
    return load_yaml("config/config.yaml")

def get_assets() -> dict:
    return load_yaml("config/assets.yaml")