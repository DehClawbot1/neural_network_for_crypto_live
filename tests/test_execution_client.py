import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


mock_client_module = MagicMock()
mock_clob_types_module = MagicMock()
mock_order_builder_module = MagicMock()
mock_constants_module = MagicMock()

mock_client_module.ClobClient = MagicMock()
mock_clob_types_module.OrderArgs = MagicMock()
mock_clob_types_module.OrderType = SimpleNamespace(GTC="GTC", FOK="FOK")
mock_clob_types_module.MarketOrderArgs = MagicMock()
mock_clob_types_module.BalanceAllowanceParams = MagicMock()
mock_clob_types_module.AssetType = SimpleNamespace(COLLATERAL="COLLATERAL", CONDITIONAL="CONDITIONAL")
mock_clob_types_module.ApiCreds = MagicMock()
mock_constants_module.BUY = "BUY"
mock_constants_module.SELL = "SELL"

sys.modules["py_clob_client"] = MagicMock()
sys.modules["py_clob_client.client"] = mock_client_module
sys.modules["py_clob_client.clob_types"] = mock_clob_types_module
sys.modules["py_clob_client.order_builder"] = mock_order_builder_module
sys.modules["py_clob_client.order_builder.constants"] = mock_constants_module

from execution_client import ExecutionClient


@patch("execution_client.os.getenv")
def test_init_missing_private_key(mock_getenv):
    mock_getenv.side_effect = lambda key, default=None: {"POLYMARKET_CHAIN_ID": "137"}.get(key, default)
    with pytest.raises(ValueError):
        ExecutionClient(private_key=None)


@patch("execution_client.os.getenv")
def test_init_with_api_creds(mock_getenv):
    def fake_env(key, default=None):
        values = {
            "PRIVATE_KEY": "0xdummy_pk",
            "POLYMARKET_API_KEY": "dummy_key",
            "POLYMARKET_API_SECRET": "dummy_secret",
            "POLYMARKET_API_PASSPHRASE": "dummy_pass",
        }
        return values.get(key, default)

    mock_getenv.side_effect = fake_env
    mock_client_module.ClobClient.reset_mock()
    inner_client = MagicMock()
    inner_client.get_balance_allowance.return_value = {"balance": "100.0"}
    mock_client_module.ClobClient.return_value = inner_client
    mock_clob_types_module.ApiCreds.side_effect = lambda key, secret, passphrase: SimpleNamespace(api_key=key, api_secret=secret, api_passphrase=passphrase)

    client = ExecutionClient()

    assert client.client is not None
    assert client.private_key == "0xdummy_pk"
    assert client.api_key == "dummy_key"


@patch("execution_client.os.getenv")
def test_init_derives_api_creds_when_missing(mock_getenv):
    def fake_env(key, default=None):
        values = {
            "PRIVATE_KEY": "0xdummy_pk",
            "POLYMARKET_API_KEY": None,
            "POLYMARKET_API_SECRET": None,
            "POLYMARKET_API_PASSPHRASE": None,
        }
        return values.get(key, default)

    mock_getenv.side_effect = fake_env
    mock_client_module.ClobClient.reset_mock()
    inner_client = MagicMock()
    inner_client.create_or_derive_api_creds.return_value = SimpleNamespace(api_key="derived_k", api_secret="derived_s", api_passphrase="derived_p")
    inner_client.get_balance_allowance.return_value = {"balance": "100.0"}
    mock_client_module.ClobClient.return_value = inner_client

    client = ExecutionClient()

    assert client.api_creds is not None
    assert client.client is inner_client
    assert client.credential_source == "derived_refreshed_env"


@patch("execution_client.os.getenv")
def test_order_placements(mock_getenv):
    mock_getenv.side_effect = lambda key, default=None: {"PRIVATE_KEY": "0xdummy_pk", "POLYMARKET_CHAIN_ID": "137"}.get(key, default)
    mock_client_module.ClobClient.reset_mock()
    inner_client = MagicMock()
    inner_client.create_or_derive_api_creds.return_value = SimpleNamespace(api_key="derived_k", api_secret="derived_s", api_passphrase="derived_p")
    inner_client.get_balance_allowance.return_value = {"balance": "100.0"}
    mock_client_module.ClobClient.return_value = inner_client

    client = ExecutionClient(private_key="0xdummy_pk")
    inner_client.create_order.return_value = "signed_limit_order"
    inner_client.create_market_order.return_value = "signed_market_order"
    inner_client.post_order.return_value = {"status": "success", "orderID": "123"}

    limit_res = client.create_and_post_order("token_1", price=0.5, size=10, side="BUY")
    market_res = client.create_and_post_market_order("token_1", amount=10, side="SELL")

    assert limit_res["orderID"] == "123"
    assert market_res["orderID"] == "123"
    assert inner_client.create_order.called
    assert inner_client.create_market_order.called


@patch("execution_client.os.getenv")
def test_read_methods(mock_getenv):
    mock_getenv.side_effect = lambda key, default=None: {"PRIVATE_KEY": "0xdummy_pk", "POLYMARKET_CHAIN_ID": "137"}.get(key, default)
    mock_client_module.ClobClient.reset_mock()
    inner_client = MagicMock()
    inner_client.create_or_derive_api_creds.return_value = SimpleNamespace(api_key="derived_k", api_secret="derived_s", api_passphrase="derived_p")
    inner_client.get_balance_allowance.return_value = {"balance": "100.0"}
    mock_client_module.ClobClient.return_value = inner_client

    client = ExecutionClient(private_key="0xdummy_pk")
    inner_client.get_order.return_value = {"id": "123", "status": "OPEN"}
    inner_client.get_orders.return_value = [{"id": "123"}]
    inner_client.get_trades.return_value = [{"trade_id": "t1"}]
    inner_client.cancel.return_value = "canceled"

    assert client.get_order("123")["status"] == "OPEN"
    assert len(client.get_open_orders()) == 1
    assert len(client.get_trades()) == 1
    assert client.cancel_order("123") == "canceled"


@patch("execution_client.os.getenv")
def test_balance_retrieval(mock_getenv):
    mock_getenv.side_effect = lambda key, default=None: {"PRIVATE_KEY": "0xdummy_pk", "POLYMARKET_CHAIN_ID": "137"}.get(key, default)
    mock_client_module.ClobClient.reset_mock()
    inner_client = MagicMock()
    inner_client.create_or_derive_api_creds.return_value = SimpleNamespace(api_key="derived_k", api_secret="derived_s", api_passphrase="derived_p")
    inner_client.get_balance_allowance.side_effect = [
        {"balance": "100.0"},
        {"balance": "100.5"},
        {"available_balance": "50.0"},
        {"amount": "200.0"},
        {"other_key": "10"},
        "not_a_dict",
    ]
    mock_client_module.ClobClient.return_value = inner_client

    client = ExecutionClient(private_key="0xdummy_pk")
    assert client.get_available_balance() == 100.5
    assert client.get_available_balance() == 50.0
    assert client.get_available_balance() == 200.0
    assert client.get_available_balance() == 0.0
    assert client.get_available_balance() == 0.0
