"""Centralized logger via loguru."""

from __future__ import annotations
import sys
from pathlib import Path
from loguru import logger


def setup_logger(level: str = "INFO", file: str | None = None):
    logger.remove()
    logger.add(sys.stdout, level=level, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")
    if file:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(file, level=level, rotation="10 MB")
    return logger