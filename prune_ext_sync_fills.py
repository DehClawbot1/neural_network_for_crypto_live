from __future__ import annotations

import argparse
from pathlib import Path

from live_position_book import LivePositionBook


def main():
    parser = argparse.ArgumentParser(description="Archive and prune redundant synthetic ext_sync fills from logs/trading.db.")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--archive-dir", default=None)
    parser.add_argument("--vacuum", action="store_true")
    args = parser.parse_args()

    book = LivePositionBook(logs_dir=args.logs_dir)
    result = book.archive_and_prune_redundant_external_sync_fills(
        archive_dir=args.archive_dir,
        vacuum=bool(args.vacuum),
    )

    print(result)
    open_df = book.get_open_positions()
    print({"open_positions_after_rebuild": int(len(open_df.index))})
    if not open_df.empty:
        cols = [c for c in ["token_id", "condition_id", "outcome_side", "shares", "avg_entry_price", "realized_pnl", "status"] if c in open_df.columns]
        print(open_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
