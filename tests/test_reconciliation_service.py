import pandas as pd
import pytest
from unittest.mock import MagicMock

from reconciliation_service import ReconciliationService
from live_position_book import LivePositionBook


def test_reconcile_detects_missing_remote_open_order(tmp_path):
    pd.DataFrame(
        [
            {"order_id": "order_1", "status": "OPEN", "size": 10.0},
            {"order_id": "order_2", "status": "OPEN", "size": 20.0},
        ]
    ).to_csv(tmp_path / "live_orders.csv", index=False)
    pd.DataFrame([]).to_csv(tmp_path / "live_fills.csv", index=False)

    client = MagicMock()
    client.get_open_orders.return_value = [{"order_id": "order_1", "status": "OPEN", "size": 10.0, "token_id": "tok-1"}]
    client.get_trades.return_value = []

    service = ReconciliationService(execution_client=client, logs_dir=tmp_path)
    report, _, _ = service.reconcile()

    assert report["local_order_rows"] == 2
    assert report["remote_open_orders"] == 1
    assert "order_2" in report["missing_remote_orders"]
    assert len(report["order_mismatches"]) == 0


def test_reconcile_filters_out_closed_local_orders(tmp_path):
    pd.DataFrame(
        [
            {"order_id": "open-1", "status": "SUBMITTED"},
            {"order_id": "filled-1", "status": "FILLED"},
            {"order_id": "closed-1", "status": "CANCELED"},
        ]
    ).to_csv(tmp_path / "live_orders.csv", index=False)
    pd.DataFrame([{"trade_id": "trade-1", "order_id": "filled-1"}]).to_csv(tmp_path / "live_fills.csv", index=False)

    client = MagicMock()
    client.get_open_orders.return_value = [{"order_id": "open-1", "status": "OPEN", "token_id": "tok-1"}]
    client.get_trades.return_value = [{"id": "trade-1", "order_id": "filled-1", "token_id": "tok-1", "price": 0.5, "size": 10}]

    service = ReconciliationService(execution_client=client, logs_dir=tmp_path)
    report, remote_orders_df, remote_trades_df = service.reconcile()

    assert report["missing_remote_orders"] == []
    assert report["missing_local_orders"] == []
    assert report["missing_local_trades"] == []
    assert report["missing_remote_trades"] == []
    assert len(remote_orders_df) == 1
    assert len(remote_trades_df) == 1


def test_reconcile_reports_status_and_size_mismatch_for_open_orders(tmp_path):
    pd.DataFrame(
        [{"order_id": "open-1", "status": "SUBMITTED", "size": 10}]
    ).to_csv(tmp_path / "live_orders.csv", index=False)
    pd.DataFrame([]).to_csv(tmp_path / "live_fills.csv", index=False)

    client = MagicMock()
    client.get_open_orders.return_value = [{"order_id": "open-1", "status": "OPEN", "size": 12, "token_id": "tok-1"}]
    client.get_trades.return_value = []

    service = ReconciliationService(execution_client=client, logs_dir=tmp_path)
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
            "filled_at": "2026-03-28T20:00:00+00:00",
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
            "filled_at": "2026-03-28T20:05:00+00:00",
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
