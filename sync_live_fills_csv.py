import argparse
from pathlib import Path

import pandas as pd

from reconciliation_service import ReconciliationService


def main():
    parser = argparse.ArgumentParser(description="Backfill logs/live_fills.csv from SQLite fill history.")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--exclude-synthetic", action="store_true")
    args = parser.parse_args()

    service = ReconciliationService(logs_dir=args.logs_dir)
    added = service.backfill_live_fills_csv_from_db(include_synthetic=not args.exclude_synthetic)

    fills_path = Path(args.logs_dir) / "live_fills.csv"
    fills_df = pd.read_csv(fills_path, engine="python", on_bad_lines="skip") if fills_path.exists() else pd.DataFrame()
    print(
        {
            "csv_rows_added": int(added),
            "csv_total_rows": int(len(fills_df)),
            "csv_path": str(fills_path),
        }
    )


if __name__ == "__main__":
    main()
