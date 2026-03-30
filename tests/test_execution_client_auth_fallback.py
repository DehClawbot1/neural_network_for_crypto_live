from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from execution_client import ExecutionClient
from test_execution_client import mock_client_module, mock_clob_types_module


@patch("execution_client.os.getenv")
def test_execution_client_falls_back_to_derived_creds_when_stored_invalid(mock_getenv):
    def fake_env(key, default=None):
        values = {
            "PRIVATE_KEY": "0x" + ("4" * 64),
            "POLYMARKET_API_KEY": "bad-key",
            "POLYMARKET_API_SECRET": "bad-secret",
            "POLYMARKET_API_PASSPHRASE": "bad-pass",
            "POLYMARKET_CHAIN_ID": "137",
            "POLYMARKET_SIGNATURE_TYPE": "0",
            "POLYMARKET_FUNDER": "0xabc",
        }
        return values.get(key, default)

    mock_getenv.side_effect = fake_env
    mock_client_module.ClobClient.reset_mock()
    mock_clob_types_module.ApiCreds.side_effect = lambda key, secret, passphrase: SimpleNamespace(
        api_key=key,
        api_secret=secret,
        api_passphrase=passphrase,
    )

    inner_client = MagicMock()
    derived_creds = SimpleNamespace(api_key="derived-key", api_secret="derived-secret", api_passphrase="derived-pass")
    inner_client.create_or_derive_api_creds.return_value = derived_creds
    inner_client.get_balance_allowance.side_effect = [Exception("401 Unauthorized"), {"balance": "100.0"}]
    mock_client_module.ClobClient.return_value = inner_client

    client = ExecutionClient()

    assert client.api_creds.api_key == "derived-key"
    assert client.credential_source == "derived_refreshed_env"
    assert inner_client.set_api_creds.call_count == 2


@patch("execution_client.os.getenv")
def test_execution_client_keeps_stored_creds_when_validation_succeeds(mock_getenv):
    def fake_env(key, default=None):
        values = {
            "PRIVATE_KEY": "0x" + ("5" * 64),
            "POLYMARKET_API_KEY": "good-key",
            "POLYMARKET_API_SECRET": "good-secret",
            "POLYMARKET_API_PASSPHRASE": "good-pass",
            "POLYMARKET_CHAIN_ID": "137",
            "POLYMARKET_SIGNATURE_TYPE": "0",
            "POLYMARKET_FUNDER": "0xabc",
        }
        return values.get(key, default)

    mock_getenv.side_effect = fake_env
    mock_client_module.ClobClient.reset_mock()
    mock_clob_types_module.ApiCreds.side_effect = lambda key, secret, passphrase: SimpleNamespace(
        api_key=key,
        api_secret=secret,
        api_passphrase=passphrase,
    )

    inner_client = MagicMock()
    inner_client.create_or_derive_api_creds.return_value = SimpleNamespace(
        api_key="derived-key",
        api_secret="derived-secret",
        api_passphrase="derived-pass",
    )
    inner_client.get_balance_allowance.return_value = {"balance": "50.0"}
    mock_client_module.ClobClient.return_value = inner_client

    client = ExecutionClient()

    assert client.api_creds.api_key == "good-key"
    assert client.credential_source == "stored_env"
    inner_client.set_api_creds.assert_called_once()
