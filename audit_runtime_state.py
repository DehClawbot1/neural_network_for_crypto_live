import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from db import Database


def _is_synthetic_fill_id(value) -> bool:
    fid = str(value or "").strip().lower()
    return (
        fid.startswith("fill_dust_clear_")
        or fid.startswith("dust_clear_")
        or fid.startswith("fill_ext_sync_")
        or fid.startswith("ext_sync_")
        or "dust_clear" in fid
    )


def _safe_read_csv(path: Path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


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
        fills_for_ghosts = fills.copy()
        if "fill_id" in fills_for_ghosts.columns:
            fills_for_ghosts = fills_for_ghosts[
                ~fills_for_ghosts["fill_id"].map(_is_synthetic_fill_id)
            ].copy()
        agg = (
            fills_for_ghosts.assign(
                buy_size=lambda d: d.apply(lambda r: float(r["size"] or 0.0) if str(r.get("side", "")).upper() == "BUY" else 0.0, axis=1),
                sell_size=lambda d: d.apply(lambda r: float(r["size"] or 0.0) if str(r.get("side", "")).upper() == "SELL" else 0.0, axis=1),
            )
            .groupby(["token_id", "condition_id", "outcome_side"], dropna=False)[["buy_size", "sell_size"]]
            .sum()
            .reset_index()
        )
        agg["net_shares"] = agg["buy_size"] - agg["sell_size"]
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
    csv_order_ids = set(live_orders_csv.get("order_id", pd.Series(dtype=str)).dropna().astype(str).tolist())
    db_order_ids = set(orders.get("order_id", pd.Series(dtype=str)).dropna().astype(str).tolist())
    csv_fill_ids = set(live_fills_csv.get("fill_id", pd.Series(dtype=str)).dropna().astype(str).tolist())
    db_fill_ids = set(fills.get("fill_id", pd.Series(dtype=str)).dropna().astype(str).tolist())
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
