import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from execution_client import ExecutionClient
from polymarket_profile_client import PolymarketProfileClient
from test_execution_client import mock_client_module


@patch("execution_client.os.getenv")
def test_execution_client_uses_env_api_creds(mock_getenv):
    def fake_env(key, default=None):
        values = {
            "PRIVATE_KEY": "0x" + ("1" * 64),
            "POLYMARKET_API_KEY": "test-key",
            "POLYMARKET_API_SECRET": "test-secret",
            "POLYMARKET_API_PASSPHRASE": "test-pass",
            "POLYMARKET_CHAIN_ID": "137",
        }
        return values.get(key, default)

    mock_getenv.side_effect = fake_env
    mock_client_module.ClobClient.reset_mock()
    mock_client_module.ClobClient.side_effect = None
    mock_client_module.ClobClient.return_value = MagicMock()
    from test_execution_client import mock_clob_types_module
    mock_clob_types_module.ApiCreds.side_effect = lambda key, secret, passphrase: SimpleNamespace(
        api_key=key,
        api_secret=secret,
        api_passphrase=passphrase,
    )

    client = ExecutionClient()

    assert getattr(client.api_creds, "api_key", None) == "test-key"
    assert getattr(client.api_creds, "api_secret", None) == "test-secret"


@patch("execution_client.os.getenv")
def test_execution_client_derives_creds_when_env_missing(mock_getenv):
    def fake_env(key, default=None):
        values = {
            "PRIVATE_KEY": "0x" + ("2" * 64),
            "POLYMARKET_API_KEY": "",
            "POLYMARKET_API_SECRET": "",
            "POLYMARKET_API_PASSPHRASE": "",
            "POLYMARKET_CHAIN_ID": "137",
        }
        return values.get(key, default)

    mock_getenv.side_effect = fake_env

    temp_client = MagicMock()
    temp_client.create_or_derive_api_creds.return_value = SimpleNamespace(
        api_key="derived-key",
        api_secret="derived-secret",
        api_passphrase="derived-pass",
    )
    final_client = MagicMock()

    with patch.object(ExecutionClient, "__module__", ExecutionClient.__module__):
        pass

    from test_execution_client import mock_client_module

    mock_client_module.ClobClient.side_effect = [temp_client, final_client]

    client = ExecutionClient()

    temp_client.create_or_derive_api_creds.assert_called_once()
    assert getattr(client.api_creds, "api_key", None) == "derived-key"


@patch("execution_client.os.getenv")
def test_execution_client_get_info_balance_api_call(mock_getenv):
    def fake_env(key, default=None):
        values = {
            "PRIVATE_KEY": "0x" + ("3" * 64),
            "POLYMARKET_CHAIN_ID": "137",
        }
        return values.get(key, default)

    mock_getenv.side_effect = fake_env

    from test_execution_client import mock_client_module

    temp_client = MagicMock()
    temp_client.create_or_derive_api_creds.return_value = SimpleNamespace(
        api_key="derived-key",
        api_secret="derived-secret",
        api_passphrase="derived-pass",
    )
    final_client = MagicMock()
    final_client.get_balance_allowance.return_value = {"balance": "1000.50"}
    mock_client_module.ClobClient.side_effect = [temp_client, final_client]

    client = ExecutionClient()
    balance = client.get_available_balance(asset_type="COLLATERAL")

    assert balance == 1000.50
    final_client.get_balance_allowance.assert_called_once()


def test_polymarket_profile_client_get_public_profile():
    client = PolymarketProfileClient(timeout=5)
    mock_response = {"address": "0xabc", "displayName": "whale"}

    with patch.object(client, "_get", return_value=mock_response) as mock_get:
        result = client.get_public_profile("0xabc")

    assert result["address"] == "0xabc"
    mock_get.assert_called_once_with(client.GAMMA_BASE, "/public-profile", {"address": "0xabc"})


def test_profile_client_get_public_info_http_path():
    profile_client = PolymarketProfileClient(timeout=5)
    mock_response = {"displayName": "TestTrader", "proxyAddress": "0x123"}

    with patch.object(profile_client.session, "get") as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status.return_value = None

        info = profile_client.get_public_profile("0xaddress")

    assert info["displayName"] == "TestTrader"
    assert "/public-profile" in mock_get.call_args[0][0]
