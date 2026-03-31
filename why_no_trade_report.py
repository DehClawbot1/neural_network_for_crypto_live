import argparse
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd


def build_report(logs_dir: Path, hours: int):
    db_path = logs_dir / "trading.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, int(hours)))
    cutoff_iso = cutoff.isoformat()

    with sqlite3.connect(str(db_path)) as conn:
        decisions = pd.read_sql_query(
            """
            SELECT
                decision_id,
                cycle_id,
                created_at,
                token_id,
                condition_id,
                market,
                final_decision,
                reject_reason,
                reject_category,
                gate,
                confidence,
                p_tp_before_sl,
                expected_return,
                edge_score,
                calibrated_edge,
                calibrated_baseline,
                order_id
            FROM candidate_decisions
            WHERE created_at >= ?
            ORDER BY decision_id DESC
            """,
            conn,
            params=(cutoff_iso,),
        )

    if decisions.empty:
        return {
            "cutoff_utc": cutoff_iso,
            "total_candidates": 0,
            "message": "No candidate decisions found in the selected window.",
        }, decisions

    total_candidates = int(len(decisions))
    skipped = decisions[decisions["final_decision"] == "SKIPPED"]
    rejected = decisions[decisions["final_decision"] == "REJECTED"]
    opened = decisions[decisions["final_decision"].isin(["ENTRY_FILLED", "PAPER_OPENED"])]
    entries_sent = decisions[decisions["order_id"].notna() & (decisions["order_id"].astype(str) != "")]
    fills_received = decisions[decisions["final_decision"] == "ENTRY_FILLED"]

    reject_counts = (
        skipped["reject_reason"]
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .head(20)
        .to_dict()
    )
    rejected_counts = (
        rejected["reject_reason"]
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .head(20)
        .to_dict()
    )
    gate_counts = (
        skipped["gate"]
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .head(20)
        .to_dict()
    )

    summary = {
        "cutoff_utc": cutoff_iso,
        "total_candidates": total_candidates,
        "candidates_skipped": int(len(skipped)),
        "candidates_rejected": int(len(rejected)),
        "candidates_opened": int(len(opened)),
        "entries_sent": int(len(entries_sent)),
        "fills_received": int(len(fills_received)),
        "top_skip_reasons": reject_counts,
        "top_reject_reasons": rejected_counts,
        "top_skip_gates": gate_counts,
    }
    return summary, decisions


def main():
    parser = argparse.ArgumentParser(description="Replay report that explains why the bot did not trade.")
    parser.add_argument("--logs-dir", default="logs", help="Logs directory that contains trading.db")
    parser.add_argument("--hours", type=int, default=12, help="Lookback window in hours")
    parser.add_argument(
        "--out-csv",
        default="why_no_trade_report.csv",
        help="Output CSV filename written under logs-dir",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    summary, decisions = build_report(logs_dir=logs_dir, hours=args.hours)
    print("WHY_NO_TRADE_SUMMARY")
    print(json.dumps(summary, indent=2, default=str))

    out_csv = logs_dir / args.out_csv
    if not decisions.empty:
        decisions.to_csv(out_csv, index=False)
        print(f"Detailed rows written to: {out_csv}")


if __name__ == "__main__":
    main()
