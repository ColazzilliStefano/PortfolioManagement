import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.logger import setup_logger
from src.market_caps.institutional import fetch_all_institutional


def main():
    setup_logger()
    s = fetch_all_institutional("2025-12-31")
    print("\nInstitutional market caps (trillions USD, 2025-12-31):")
    print(s)
    print(f"\nSum: {s.sum():.3f} T USD")
    print("\nNote: 'None' values indicate missing CSV files in "
          "data/raw/market_caps/. They will be replaced by the fallback.")


if __name__ == "__main__":
    main()