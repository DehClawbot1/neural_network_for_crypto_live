from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from live_risk_manager import LiveRiskManager
from order_manager import OrderManager


@pytest.fixture
def mock_execution_client():
    with patch("order_manager.ExecutionClient") as MockClient:
        client_instance = MagicMock()
        client_instance.get_balance_allowance.return_value = {"balance": 5000.0}
        client_instance.create_and_post_order.return_value = {"orderID": "test-order-123", "status": "SUBMITTED"}
        client_instance.get_order.return_value = {"id": "fill-1", "order_id": "test-order-123", "token_id": "0x123", "price": 0.5, "size": 100, "status": "FILLED"}
        MockClient.return_value = client_instance
        yield client_instance


def _build_manager(tmp_path):
    manager = OrderManager(logs_dir=tmp_path)
    manager.risk = MagicMock(spec=LiveRiskManager)
    manager.risk.pre_trade_check.return_value = MagicMock(allowed=True, reason="ok")
    return manager


def test_order_manager_submit_success(mock_execution_client, tmp_path):
    manager = _build_manager(tmp_path)

    row, response = manager.submit_entry(token_id="0x123", price=0.5, size=100, side="BUY")

    assert response["orderID"] == "test-order-123"
    assert row["status"] == "SUBMITTED"
    assert row["size"] == 100


def test_order_manager_insufficient_funds(mock_execution_client, tmp_path):
    mock_execution_client.get_balance_allowance.return_value = {"balance": 10.0}
    manager = _build_manager(tmp_path)

    row, response = manager.submit_entry(token_id="0x123", price=0.5, size=100, side="BUY")

    assert response is None
    assert row["status"] == "REJECTED"
    assert row["reason"] == "insufficient_funds"


def test_wait_for_fill_persists_fill(mock_execution_client, tmp_path):
    manager = _build_manager(tmp_path)

    result = manager.wait_for_fill("test-order-123", timeout_seconds=1, poll_seconds=0)

    assert result["filled"] is True
    assert result["response"]["status"] == "FILLED"
    fills_df = pd.read_csv(tmp_path / "live_fills.csv")
    assert not fills_df.empty
    assert str(fills_df.iloc[0]["order_id"]) == "test-order-123"
    db_rows = manager.db.query_all("SELECT order_id, token_id FROM fills WHERE order_id = ?", ("test-order-123",))
    assert db_rows
    assert db_rows[0]["token_id"] == "0x123"
