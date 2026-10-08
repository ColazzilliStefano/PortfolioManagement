import sys
from loguru import logger

def setup_logger(level="INFO", file=None):
    logger.remove()
    logger.add(sys.stdout, level=level)
    if file:
        logger.add(file, level=level, rotation="10 MB")
    return logger