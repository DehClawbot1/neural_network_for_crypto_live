import pandas as pd
import pytest
from unittest.mock import MagicMock

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


def test_sync_and_rebuild_uses_trade_side_for_manual_exit_without_order_row(tmp_path):
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
    summary = service.sync_orders_and_fills()
    assert summary["fills"] == 2

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
    summary = service.sync_orders_and_fills()

    fills_df = pd.read_csv(tmp_path / "live_fills.csv")
    assert summary["fills"] == 1
    assert summary["fill_csv_rows_added"] == 1
    assert len(fills_df) == 1
    assert fills_df.iloc[0]["fill_id"] == "trade-1"
    assert fills_df.iloc[0]["trade_id"] == "trade-1"
    assert fills_df.iloc[0]["condition_id"] == "cond-1"
    assert fills_df.iloc[0]["fill_source"] == "exchange_sync"


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
