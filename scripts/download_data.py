"""Download the IBM Telco Customer Churn CSV if it is not already present."""

from __future__ import annotations

import urllib.request
from pathlib import Path

SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DEFAULT_PATH = Path("data/raw/Telco-Customer-Churn.csv")


def main() -> None:
    DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DEFAULT_PATH.exists():
        print(f"Already present: {DEFAULT_PATH}")
        return
    print(f"Downloading {SOURCE_URL}")
    urllib.request.urlretrieve(SOURCE_URL, DEFAULT_PATH)
    print(f"Wrote {DEFAULT_PATH}")


if __name__ == "__main__":
    main()
