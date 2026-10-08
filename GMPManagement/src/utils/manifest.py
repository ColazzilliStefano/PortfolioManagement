"""Execution manifest for reproducibility."""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _hash_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def create_manifest(project_root: Path,
                    run_id: str | None = None) -> dict:
    if run_id is None:
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _get_git_commit(),
        "hashes": {
            "config.yaml": _hash_file(project_root / "config/config.yaml"),
            "assets.yaml": _hash_file(project_root / "config/assets.yaml"),
            "gmp_dynamic.yaml": _hash_file(project_root / "config/gmp_dynamic.yaml"),
            "returns_monthly.parquet": _hash_file(
                project_root / "data/processed/returns_monthly.parquet"
            ),
            "gmp_weights_history.parquet": _hash_file(
                project_root / "data/processed/gmp_weights_history.parquet"
            ),
        },
    }

    out_dir = project_root / "outputs" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest