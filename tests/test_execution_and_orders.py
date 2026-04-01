from unittest.mock import MagicMock, patch
import os

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


def test_order_manager_onchain_fallback_allows_buy(mock_execution_client, tmp_path):
    mock_execution_client.get_balance_allowance.return_value = {"balance": 0.0}
    mock_execution_client.get_onchain_collateral_balance.return_value = {
        "wallet": "0x672c1b45553aac41e9dccdf68a65be6a401c3176",
        "balances": {"0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174": 20.50165},
        "total": 20.50165,
    }
    manager = _build_manager(tmp_path)

    with patch.dict(os.environ, {"ALLOW_ONCHAIN_BALANCE_FALLBACK": "true"}):
        row, response = manager.submit_entry(token_id="0x123", price=0.5, size=20, side="BUY")

    assert response["orderID"] == "test-order-123"
    assert row["status"] == "SUBMITTED"


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


def test_record_fill_respects_expanded_live_fill_schema(mock_execution_client, tmp_path):
    pd.DataFrame(
        [
            {
                "timestamp": "2026-04-01T10:00:00+00:00",
                "trade_id": "existing-fill",
                "order_id": "existing-order",
                "token_id": "tok-0",
                "condition_id": "cond-0",
                "outcome_side": "Yes",
                "side": "BUY",
                "price": 0.4,
                "size": 1.0,
                "filled_at": "2026-04-01T10:00:00+00:00",
                "fill_id": "existing-fill",
                "fill_source": "exchange_sync",
            }
        ]
    ).to_csv(tmp_path / "live_fills.csv", index=False)

    manager = _build_manager(tmp_path)
    manager.record_fill(
        {
            "trade_id": "new-fill",
            "order_id": "new-order",
            "token_id": "tok-1",
            "price": 0.5,
            "size": 2.0,
            "filled_at": "2026-04-01T10:05:00+00:00",
        }
    )

    fills_df = pd.read_csv(tmp_path / "live_fills.csv")
    assert len(fills_df) == 2
    assert "fill_source" in fills_df.columns
    assert fills_df.iloc[-1]["fill_id"] == "new-fill"
    assert fills_df.iloc[-1]["order_id"] == "new-order"


def test_automated_exit_trigger(mock_execution_client, tmp_path):
    manager = _build_manager(tmp_path)

    with patch("market_price_service.MarketPriceService.get_quote", return_value={"best_bid": 0.75}):
        manager.submit_entry = MagicMock(return_value=({"status": "SUBMITTED"}, {}))

        manager.monitor_and_trigger_exit(
            token_id="tok-1",
            target_price=0.70,
            size=10,
            condition_id="cond-1",
        )

    manager.submit_entry.assert_called_once()
    _, kwargs = manager.submit_entry.call_args
    assert kwargs["side"] == "SELL"
    assert kwargs["price"] == 0.75
