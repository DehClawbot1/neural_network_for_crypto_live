from unittest.mock import MagicMock, patch

from shadow_purgatory import ResilientCLOBClient


@patch("shadow_purgatory.time.sleep", return_value=None)
@patch("shadow_purgatory.requests.get")
def test_clob_client_retry_on_429(mock_get, mock_sleep):
    client = ResilientCLOBClient(max_retries=3, base_delay=1)

    mock_response_429 = MagicMock()
    mock_response_429.status_code = 429

    mock_response_success = MagicMock()
    mock_response_success.status_code = 200
    mock_response_success.json.return_value = [{"id": "trade1"}]

    mock_get.side_effect = [mock_response_429, mock_response_429, mock_response_success]

    result = client.get_trades_with_retry("token_123", 1742731200)

    assert len(result) == 1
    assert mock_get.call_count == 3
    assert mock_sleep.call_count == 2
