import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from db import Database
from live_position_book import LivePositionBook


def _is_synthetic_fill_id(value) -> bool:
    fid = str(value or "").strip().lower()
    return (
        fid.startswith("fill_dust_clear_")
        or fid.startswith("dust_clear_")
        or fid.startswith("fill_ext_sync_")
        or fid.startswith("ext_sync_")
        or "dust_clear" in fid
    )


def _is_dust_clear_fill_id(value) -> bool:
    fid = str(value or "").strip().lower()
    return (
        fid.startswith("fill_dust_clear_")
        or fid.startswith("dust_clear_")
        or "dust_clear" in fid
    )


def _safe_read_csv(path: Path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _collect_fill_ids(frame: pd.DataFrame):
    ids = set()
    if frame is None or frame.empty:
        return ids
    for column in ["fill_id", "trade_id"]:
        if column not in frame.columns:
            continue
        values = frame[column].dropna().astype(str).tolist()
        ids.update(value.strip() for value in values if str(value).strip() and str(value).strip().lower() not in {"nan", "none"})
    return ids


def _collect_closed_keys(frame: pd.DataFrame):
    keys = set()
    if frame is None or frame.empty:
        return keys
    if "close_fingerprint" in frame.columns:
        values = frame["close_fingerprint"].dropna().astype(str).tolist()
        keys.update(value.strip() for value in values if str(value).strip() and str(value).strip().lower() not in {"nan", "none"})
        if keys:
            return keys
    candidates = ["token_id", "condition_id", "outcome_side", "opened_at", "close_reason", "entry_price", "exit_price", "shares"]
    if not all(column in frame.columns for column in candidates):
        return keys
    work = frame[candidates].copy()
    work["exit_price"] = work["exit_price"].fillna(frame["current_price"] if "current_price" in frame.columns else "")
    for _, row in work.iterrows():
        key = "|".join(str(row.get(col, "") or "") for col in candidates)
        if key.strip("|"):
            keys.add(key)
    return keys


def _print_section(title):
    print(f"\n=== {title} ===")


def run_audit(logs_dir: str):
    logs_path = Path(logs_dir)
    db_path = logs_path / "trading.db"
    if not db_path.exists():
        print(f"Missing DB: {db_path}")
        return 1

    db = Database(db_path)
    report = db.integrity_report()
    _print_section("DB Integrity")
    print(report)

    with sqlite3.connect(db_path) as conn:
        orders = pd.read_sql_query("SELECT order_id, status, token_id, size, price, created_at FROM orders", conn)
        fills = pd.read_sql_query("SELECT fill_id, order_id, token_id, condition_id, outcome_side, side, size, price, filled_at FROM fills", conn)
        live = pd.read_sql_query("SELECT position_key, token_id, condition_id, outcome_side, shares, status FROM live_positions", conn)

    _print_section("Row Counts")
    print({"orders": len(orders), "fills": len(fills), "live_positions": len(live)})

    _print_section("Duplicate Keys")
    print(
        {
            "dup_order_ids": int(orders["order_id"].astype(str).duplicated().sum()) if not orders.empty else 0,
            "dup_fill_ids": int(fills["fill_id"].astype(str).duplicated().sum()) if not fills.empty else 0,
        }
    )

    _print_section("Ghost Position Candidates")
    ghosts = pd.DataFrame()
    if not live.empty and not fills.empty:
        book = LivePositionBook(logs_dir=logs_path)
        rebuilt = book.rebuild_from_db()
        agg = rebuilt[["token_id", "condition_id", "outcome_side", "shares"]].copy() if not rebuilt.empty else pd.DataFrame(columns=["token_id", "condition_id", "outcome_side", "shares"])
        agg = agg.rename(columns={"shares": "net_shares"})
        ghosts = live[live["status"].astype(str).str.upper() == "OPEN"].merge(
            agg[["token_id", "condition_id", "outcome_side", "net_shares"]],
            on=["token_id", "condition_id", "outcome_side"],
            how="left",
        )
        ghosts["net_shares"] = ghosts["net_shares"].fillna(0.0)
        ghosts = ghosts[(ghosts["shares"].astype(float) - ghosts["net_shares"].astype(float)).abs() > 1e-5]
    print(ghosts.head(20).to_string(index=False) if not ghosts.empty else "none")

    _print_section("CSV vs DB Drift")
    live_orders_csv = _safe_read_csv(logs_path / "live_orders.csv")
    live_fills_csv = _safe_read_csv(logs_path / "live_fills.csv")
    positions_csv = _safe_read_csv(logs_path / "positions.csv")
    csv_order_ids = set(live_orders_csv.get("order_id", pd.Series(dtype=str)).dropna().astype(str).tolist())
    db_order_ids = set(orders.get("order_id", pd.Series(dtype=str)).dropna().astype(str).tolist())
    csv_fill_ids = _collect_fill_ids(live_fills_csv)
    db_fill_ids = _collect_fill_ids(fills)
    csv_fill_ids_clean = {fid for fid in csv_fill_ids if not _is_synthetic_fill_id(fid)}
    db_fill_ids_clean = {fid for fid in db_fill_ids if not _is_synthetic_fill_id(fid)}
    print(
        {
            "orders_only_csv": len(csv_order_ids - db_order_ids),
            "orders_only_db": len(db_order_ids - csv_order_ids),
            "fills_only_csv": len(csv_fill_ids - db_fill_ids),
            "fills_only_db": len(db_fill_ids - csv_fill_ids),
            "fills_only_csv_ex_synth": len(csv_fill_ids_clean - db_fill_ids_clean),
            "fills_only_db_ex_synth": len(db_fill_ids_clean - csv_fill_ids_clean),
        }
    )

    _print_section("Positions CSV vs live_positions Drift")
    db_live_open_keys = set()
    if not live.empty:
        live_open = live[live["status"].astype(str).str.upper() == "OPEN"].copy()
        for _, row in live_open.iterrows():
            db_live_open_keys.add(
                (
                    str(row.get("token_id") or ""),
                    str(row.get("condition_id") or ""),
                    str(row.get("outcome_side") or ""),
                )
            )
    csv_open_keys = set()
    if not positions_csv.empty:
        work = positions_csv.copy()
        if "status" in work.columns:
            work = work[work["status"].astype(str).str.upper() == "OPEN"]
        for _, row in work.iterrows():
            csv_open_keys.add(
                (
                    str(row.get("token_id") or ""),
                    str(row.get("condition_id") or ""),
                    str(row.get("outcome_side") or ""),
                )
            )
    print(
        {
            "positions_only_csv": len(csv_open_keys - db_live_open_keys),
            "positions_only_db": len(db_live_open_keys - csv_open_keys),
        }
    )

    _print_section("Closed Lifecycle Drift")
    closed_positions_csv = _safe_read_csv(logs_path / "closed_positions.csv")
    feedback_reports_csv = _safe_read_csv(logs_path / "trade_feedback_reports.csv")
    with sqlite3.connect(db_path) as conn:
        closed_db = pd.read_sql_query(
            """
            SELECT position_id, token_id, condition_id, outcome_side, close_reason, entry_price,
                   current_price, exit_price, shares, opened_at, closed_at, close_fingerprint
            FROM positions
            WHERE UPPER(COALESCE(status, '')) = 'CLOSED'
            """,
            conn,
        )
    closed_csv_keys = _collect_closed_keys(closed_positions_csv)
    closed_db_keys = _collect_closed_keys(closed_db)
    feedback_keys = _collect_closed_keys(feedback_reports_csv)
    expected_feedback = set()
    if not closed_positions_csv.empty:
        feedback_src = closed_positions_csv.copy()
        if "close_reason" in feedback_src.columns:
            feedback_src = feedback_src[
                ~feedback_src["close_reason"].astype(str).str.strip().str.lower().eq("external_manual_close")
            ]
        expected_feedback = _collect_closed_keys(feedback_src)
    print(
        {
            "closed_only_csv": len(closed_csv_keys - closed_db_keys),
            "closed_only_db": len(closed_db_keys - closed_csv_keys),
            "feedback_expected_missing": len(expected_feedback - feedback_keys),
            "feedback_unexpected_extra": len(feedback_keys - expected_feedback),
        }
    )

    return 0


def main():
    parser = argparse.ArgumentParser(description="Audit trading runtime state (DB/CSV sync, ghosts, corruption).")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--reset-if-corrupt", action="store_true")
    args = parser.parse_args()

    exit_code = run_audit(args.logs_dir)

    if args.reset_if_corrupt:
        db = Database(Path(args.logs_dir) / "trading.db")
        report = db.integrity_report()
        if not report.get("ok", True):
            backup = db.backup_and_reset_runtime_state(args.logs_dir)
            print(f"\nRuntime state reset completed. Backup: {backup}")
            print("Model weights were not modified.")

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
