import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.logger import setup_logger
from src.gmp.dynamic_weights import main as compute_gmp


def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", choices=["monthly", "semiannual", "annual"],
                        default=None)
    args = parser.parse_args()
    compute_gmp(frequency=args.frequency)


if __name__ == "__main__":
    main()