import argparse
from pathlib import Path

import pandas as pd

from reconciliation_service import ReconciliationService


def main():
    parser = argparse.ArgumentParser(description="Backfill logs/live_orders.csv from SQLite order history.")
    parser.add_argument("--logs-dir", default="logs")
    args = parser.parse_args()

    service = ReconciliationService(logs_dir=args.logs_dir)
    result = service.backfill_live_orders_csv_from_db(update_existing=True)

    orders_path = Path(args.logs_dir) / "live_orders.csv"
    orders_df = pd.read_csv(orders_path, engine="python", on_bad_lines="skip") if orders_path.exists() else pd.DataFrame()
    print(
        {
            "csv_rows_added": int(result.get("added", 0)),
            "csv_rows_updated": int(result.get("updated", 0)),
            "csv_total_rows": int(len(orders_df)),
            "csv_path": str(orders_path),
        }
    )


if __name__ == "__main__":
    main()
