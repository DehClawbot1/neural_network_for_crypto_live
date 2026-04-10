import pandas as pd
import pytest
from unittest.mock import MagicMock
import os

from reconciliation_service import ReconciliationService
from live_position_book import LivePositionBook


def test_reconcile_detects_missing_remote_open_order(tmp_path):
    client = MagicMock()
    client.get_open_orders.return_value = [{"order_id": "order_1", "status": "OPEN", "size": 10.0, "token_id": "tok-1"}]
    client.get_trades.return_value = []

    service = ReconciliationService(execution_client=client, logs_dir=tmp_path)
    service.db.execute(
        "INSERT OR REPLACE INTO orders (order_id, token_id, condition_id, outcome_side, order_side, price, size, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("order_1", "tok-1", None, None, "BUY", 0.5, 10.0, "OPEN", pd.Timestamp.now(tz="UTC").isoformat()),
    )
    service.db.execute(
        "INSERT OR REPLACE INTO orders (order_id, token_id, condition_id, outcome_side, order_side, price, size, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("order_2", "tok-2", None, None, "BUY", 0.5, 20.0, "OPEN", pd.Timestamp.now(tz="UTC").isoformat()),
    )
    report, _, _ = service.reconcile()

    assert report["local_order_rows"] == 2
    assert report["remote_open_orders"] == 1
    assert "order_2" in report["missing_remote_orders"]
    assert len(report["order_mismatches"]) == 0


def test_reconcile_filters_out_closed_local_orders(tmp_path):
    client = MagicMock()
    client.get_open_orders.return_value = [{"order_id": "open-1", "status": "OPEN", "token_id": "tok-1"}]
    client.get_trades.return_value = [{"id": "trade-1", "order_id": "filled-1", "token_id": "tok-1", "price": 0.5, "size": 10, "filled_at": pd.Timestamp.now(tz="UTC").isoformat()}]

    service = ReconciliationService(execution_client=client, logs_dir=tmp_path)
    for order_id, status in [("open-1", "SUBMITTED"), ("filled-1", "FILLED"), ("closed-1", "CANCELED")]:
        service.db.execute(
            "INSERT OR REPLACE INTO orders (order_id, token_id, condition_id, outcome_side, order_side, price, size, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (order_id, "tok-1", None, None, "BUY", 0.5, 10.0, status, pd.Timestamp.now(tz="UTC").isoformat()),
        )
    service.db.execute(
        "INSERT OR REPLACE INTO fills (fill_id, order_id, token_id, condition_id, outcome_side, side, price, size, filled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("trade-1", "filled-1", "tok-1", None, None, "BUY", 0.5, 10.0, pd.Timestamp.now(tz="UTC").isoformat()),
    )
    report, remote_orders_df, remote_trades_df = service.reconcile()

    assert report["missing_remote_orders"] == []
    assert report["missing_local_orders"] == []
    assert report["missing_local_trades"] == []
    assert report["missing_remote_trades"] == []
    assert len(remote_orders_df) == 1
    assert len(remote_trades_df) == 1


def test_reconcile_reports_status_and_size_mismatch_for_open_orders(tmp_path):
    client = MagicMock()
    client.get_open_orders.return_value = [{"order_id": "open-1", "status": "OPEN", "size": 12, "token_id": "tok-1"}]
    client.get_trades.return_value = []

    service = ReconciliationService(execution_client=client, logs_dir=tmp_path)
    service.db.execute(
        "INSERT OR REPLACE INTO orders (order_id, token_id, condition_id, outcome_side, order_side, price, size, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("open-1", "tok-1", None, None, "BUY", 0.5, 10.0, "SUBMITTED", pd.Timestamp.now(tz="UTC").isoformat()),
    )
    report, _, _ = service.reconcile()

    assert len(report["order_mismatches"]) == 1
    mismatch = report["order_mismatches"][0]
    assert mismatch["order_id"] == "open-1"
    assert str(mismatch["local_status"]) == "SUBMITTED"
    assert str(mismatch["remote_status"]) == "OPEN"


def test_sync_and_rebuild_uses_trade_side_for_manual_exit_without_order_row(tmp_path, monkeypatch):
    client = MagicMock()
    client.get_open_orders.return_value = []
    client.get_trades.return_value = [
        {
            "id": "buy-1",
            "order_id": "buy-order-1",
            "token_id": "tok-1",
            "condition_id": "cond-1",
            "outcome_side": "Yes",
            "side": "BUY",
            "price": 0.4,
            "size": 5,
            "filled_at": pd.Timestamp.now(tz="UTC").isoformat(),
        },
        {
            "id": "sell-1",
            "order_id": "manual-sell-1",
            "token_id": "tok-1",
            "condition_id": "cond-1",
            "outcome_side": "Yes",
            "side": "SELL",
            "price": 0.6,
            "size": 5,
            "filled_at": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    ]

    service = ReconciliationService(execution_client=client, logs_dir=tmp_path)
    service.db.execute(
        "INSERT OR REPLACE INTO live_positions (position_key, token_id, condition_id, outcome_side, shares, avg_entry_price, realized_pnl, last_fill_at, source, status, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "tok-1|cond-1|Yes",
            "tok-1",
            "cond-1",
            "Yes",
            5.0,
            0.4,
            0.0,
            pd.Timestamp.now(tz="UTC").isoformat(),
            "test_seed",
            "OPEN",
            pd.Timestamp.now(tz="UTC").isoformat(),
        ),
    )
    summary = service.sync_orders_and_fills()
    assert summary["fills"] == 2

    monkeypatch.setattr(LivePositionBook, "_load_profile_snapshot_rows", lambda self: [])
    book = LivePositionBook(logs_dir=tmp_path)
    rebuilt = book.rebuild_from_db()
    open_df = book.get_open_positions()

    assert not rebuilt.empty
    assert open_df.empty
    rows = book.db.query_all("SELECT token_id, condition_id, outcome_side, shares, status FROM live_positions WHERE token_id = ?", ("tok-1",))
    assert rows
    assert rows[0]["shares"] == 0.0
    assert rows[0]["status"] == "CLOSED"


def test_sync_orders_and_fills_mirrors_remote_fills_to_csv(tmp_path):
    client = MagicMock()
    client.get_open_orders.return_value = []
    client.get_trades.return_value = [
        {
            "id": "trade-1",
            "order_id": "order-1",
            "token_id": "tok-1",
            "condition_id": "cond-1",
            "outcome_side": "Yes",
            "side": "BUY",
            "price": 0.45,
            "size": 3,
            "filled_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }
    ]

    service = ReconciliationService(execution_client=client, logs_dir=tmp_path)
    old = os.environ.get("SYNC_ALL_RECENT_REMOTE_TRADES")
    os.environ["SYNC_ALL_RECENT_REMOTE_TRADES"] = "true"
    try:
        summary = service.sync_orders_and_fills()
    finally:
        if old is None:
            os.environ.pop("SYNC_ALL_RECENT_REMOTE_TRADES", None)
        else:
            os.environ["SYNC_ALL_RECENT_REMOTE_TRADES"] = old

    fills_df = pd.read_csv(tmp_path / "live_fills.csv")
    assert summary["fills"] == 1
    assert summary["fill_csv_rows_added"] == 1
    assert len(fills_df) == 1
    assert fills_df.iloc[0]["fill_id"] == "trade-1"
    assert fills_df.iloc[0]["trade_id"] == "trade-1"
    assert fills_df.iloc[0]["condition_id"] == "cond-1"
    assert fills_df.iloc[0]["fill_source"] == "exchange_sync"


def test_sync_orders_and_fills_skips_untracked_remote_trade_by_default(tmp_path):
    client = MagicMock()
    client.get_open_orders.return_value = []
    client.get_trades.return_value = [
        {
            "id": "trade-1",
            "order_id": "remote-order-1",
            "token_id": "tok-remote",
            "condition_id": "cond-remote",
            "outcome_side": "Yes",
            "side": "BUY",
            "price": 0.45,
            "size": 3,
            "filled_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }
    ]

    service = ReconciliationService(execution_client=client, logs_dir=tmp_path)
    old = os.environ.pop("SYNC_ALL_RECENT_REMOTE_TRADES", None)
    try:
        summary = service.sync_orders_and_fills()
    finally:
        if old is not None:
            os.environ["SYNC_ALL_RECENT_REMOTE_TRADES"] = old

    assert summary["fills"] == 0
    assert not (tmp_path / "live_fills.csv").exists()
    db_rows = service.db.query_all("SELECT * FROM fills")
    assert db_rows == []


def test_sync_orders_and_fills_mirrors_remote_orders_to_csv(tmp_path):
    client = MagicMock()
    client.get_open_orders.return_value = [
        {
            "id": "order-1",
            "token_id": "tok-1",
            "condition_id": "cond-1",
            "outcome_side": "Yes",
            "side": "BUY",
            "price": 0.45,
            "size": 3,
            "status": "OPEN",
            "created_at": "2026-04-01T10:00:00+00:00",
        }
    ]
    client.get_trades.return_value = []

    service = ReconciliationService(execution_client=client, logs_dir=tmp_path)
    summary = service.sync_orders_and_fills()

    orders_df = pd.read_csv(tmp_path / "live_orders.csv")
    assert summary["orders"] == 1
    assert summary["order_csv_rows_added"] == 1
    assert len(orders_df) == 1
    assert orders_df.iloc[0]["order_id"] == "order-1"
    assert orders_df.iloc[0]["condition_id"] == "cond-1"
    assert orders_df.iloc[0]["order_source"] == "exchange_sync"


def test_reconcile_uses_recent_exchange_synced_fill_history_as_remote_window(tmp_path):
    client = MagicMock()
    client.get_open_orders.return_value = []
    client.get_trades.return_value = []

    synced_fill_id = "trade-from-history"
    now_iso = pd.Timestamp.now(tz="UTC").isoformat()

    pd.DataFrame(
        [
            {
                "timestamp": now_iso,
                "trade_id": synced_fill_id,
                "order_id": "order-1",
                "token_id": "tok-1",
                "condition_id": "cond-1",
                "outcome_side": "Yes",
                "side": "BUY",
                "price": 0.45,
                "size": 3,
                "filled_at": now_iso,
                "fill_id": synced_fill_id,
                "fill_source": "exchange_sync",
            }
        ]
    ).to_csv(tmp_path / "live_fills.csv", index=False)

    service = ReconciliationService(execution_client=client, logs_dir=tmp_path)
    service.db.execute(
        "INSERT OR REPLACE INTO orders (order_id, token_id, condition_id, outcome_side, order_side, price, size, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("order-1", "tok-1", "cond-1", "Yes", "BUY", 0.45, 3.0, "FILLED", now_iso),
    )
    service.db.execute(
        "INSERT OR REPLACE INTO fills (fill_id, order_id, token_id, condition_id, outcome_side, side, price, size, filled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (synced_fill_id, "order-1", "tok-1", "cond-1", "Yes", "BUY", 0.45, 3.0, now_iso),
    )

    report, _, remote_trades_df = service.reconcile()

    assert report["missing_remote_trades"] == []
    assert report["remote_trade_history_rows"] == 1
    assert synced_fill_id in set(remote_trades_df["fill_id"].astype(str).tolist())


def test_backfill_live_fills_csv_from_db_is_idempotent(tmp_path):
    service = ReconciliationService(execution_client=MagicMock(), logs_dir=tmp_path)
    service.db.execute(
        "INSERT OR REPLACE INTO fills (fill_id, order_id, token_id, condition_id, outcome_side, side, price, size, filled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("fill-1", "order-1", "tok-1", "cond-1", "Yes", "BUY", 0.55, 2.0, "2026-04-01T10:00:00+00:00"),
    )

    first_added = service.backfill_live_fills_csv_from_db()
    second_added = service.backfill_live_fills_csv_from_db()

    fills_df = pd.read_csv(tmp_path / "live_fills.csv")
    assert first_added == 1
    assert second_added == 0
    assert len(fills_df) == 1
    assert fills_df.iloc[0]["fill_id"] == "fill-1"
    assert fills_df.iloc[0]["fill_source"] == "db_backfill"


def test_backfill_live_orders_csv_updates_existing_row_status(tmp_path):
    pd.DataFrame(
        [
            {
                "timestamp": "2026-04-01T10:00:00+00:00",
                "order_id": "order-1",
                "status": "SUBMITTED",
                "idempotency_key": "keep-me",
            }
        ]
    ).to_csv(tmp_path / "live_orders.csv", index=False)

    service = ReconciliationService(execution_client=MagicMock(), logs_dir=tmp_path)
    service.db.execute(
        "INSERT OR REPLACE INTO orders (order_id, token_id, condition_id, outcome_side, order_side, price, size, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("order-1", "tok-1", "cond-1", "Yes", "BUY", 0.55, 2.0, "FILLED", "2026-04-01T10:00:00+00:00"),
    )

    first_result = service.backfill_live_orders_csv_from_db(update_existing=True)
    second_result = service.backfill_live_orders_csv_from_db(update_existing=True)

    orders_df = pd.read_csv(tmp_path / "live_orders.csv")
    assert first_result["added"] == 0
    assert first_result["updated"] >= 1
    assert second_result["added"] == 0
    assert orders_df.iloc[0]["status"] == "FILLED"
    assert orders_df.iloc[0]["idempotency_key"] == "keep-me"


def test_archive_and_prune_unmatched_remote_fills_archives_untracked_rows(tmp_path, monkeypatch):
    now_iso = pd.Timestamp.now(tz="UTC").isoformat()
    service = ReconciliationService(execution_client=MagicMock(), logs_dir=tmp_path)

    db_fill_rows = [
        ("fill-prune-1", "remote-order-1", "tok-prune-1", "cond-prune-1", "Yes", "BUY", 0.45, 3.0, now_iso),
        ("fill-prune-2", "remote-order-2", "tok-prune-2", "cond-prune-2", "No", "SELL", 0.55, 2.0, now_iso),
        ("fill-keep-live", "remote-order-3", "tok-live", "cond-live", "Yes", "BUY", 0.65, 5.0, now_iso),
    ]
    for row in db_fill_rows:
        service.db.execute(
            "INSERT OR REPLACE INTO fills (fill_id, order_id, token_id, condition_id, outcome_side, side, price, size, filled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    pd.DataFrame(
        [
            {
                "timestamp": now_iso,
                "trade_id": "fill-prune-1",
                "order_id": "remote-order-1",
                "token_id": "tok-prune-1",
                "condition_id": "cond-prune-1",
                "outcome_side": "Yes",
                "side": "BUY",
                "price": 0.45,
                "size": 3.0,
                "filled_at": now_iso,
                "fill_id": "fill-prune-1",
                "fill_source": "exchange_sync",
            },
            {
                "timestamp": now_iso,
                "trade_id": "fill-prune-2",
                "order_id": "remote-order-2",
                "token_id": "tok-prune-2",
                "condition_id": "cond-prune-2",
                "outcome_side": "No",
                "side": "SELL",
                "price": 0.55,
                "size": 2.0,
                "filled_at": now_iso,
                "fill_id": "fill-prune-2",
                "fill_source": "db_backfill",
            },
            {
                "timestamp": now_iso,
                "trade_id": "fill-keep-live",
                "order_id": "remote-order-3",
                "token_id": "tok-live",
                "condition_id": "cond-live",
                "outcome_side": "Yes",
                "side": "BUY",
                "price": 0.65,
                "size": 5.0,
                "filled_at": now_iso,
                "fill_id": "fill-keep-live",
                "fill_source": "exchange_sync",
            },
        ]
    ).to_csv(tmp_path / "live_fills.csv", index=False)

    monkeypatch.setattr(
        service,
        "_load_meaningful_profile_position_keys",
        lambda: {("tok-live", "cond-live", "yes")},
    )

    result = service.archive_and_prune_unmatched_remote_fills()

    assert result["archived_rows"] == 2
    assert result["deleted_db_rows"] == 2
    assert result["deleted_live_fill_rows"] == 2
    assert (tmp_path / "archives").exists()
    assert pd.read_csv(result["archive_csv"]).shape[0] == 2

    remaining_db_fill_ids = {
        row["fill_id"] for row in service.db.query_all("SELECT fill_id FROM fills ORDER BY fill_id")
    }
    remaining_csv_fill_ids = set(pd.read_csv(tmp_path / "live_fills.csv")["fill_id"].astype(str).tolist())

    assert remaining_db_fill_ids == {"fill-keep-live"}
    assert remaining_csv_fill_ids == {"fill-keep-live"}
