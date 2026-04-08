import pandas as pd

from leaderboard_scraper import _keep_latest_open_long_state_per_wallet_market_side


def test_keep_latest_open_long_state_per_wallet_market_side_collapses_historical_entries():
    df = pd.DataFrame(
        [
            {
                "trader_wallet": "0xabc",
                "condition_id": "cond-1",
                "market_slug": "btc-above-70k",
                "outcome_side": "YES",
                "entry_intent": "OPEN_LONG",
                "source_wallet_position_event": "NEW_ENTRY",
                "timestamp": "2026-04-08T10:00:00Z",
            },
            {
                "trader_wallet": "0xabc",
                "condition_id": "cond-1",
                "market_slug": "btc-above-70k",
                "outcome_side": "YES",
                "entry_intent": "OPEN_LONG",
                "source_wallet_position_event": "SCALE_IN",
                "timestamp": "2026-04-08T10:05:00Z",
            },
            {
                "trader_wallet": "0xabc",
                "condition_id": "cond-1",
                "market_slug": "btc-above-70k",
                "outcome_side": "NO",
                "entry_intent": "OPEN_LONG",
                "source_wallet_position_event": "REVERSAL_ENTRY",
                "timestamp": "2026-04-08T10:06:00Z",
            },
            {
                "trader_wallet": "0xabc",
                "condition_id": "cond-1",
                "market_slug": "btc-above-70k",
                "outcome_side": "YES",
                "entry_intent": "CLOSE_LONG",
                "source_wallet_position_event": "FULL_EXIT",
                "timestamp": "2026-04-08T10:07:00Z",
            },
        ]
    )

    result = _keep_latest_open_long_state_per_wallet_market_side(df)

    assert len(result) == 3
    open_yes = result[
        (result["entry_intent"] == "OPEN_LONG")
        & (result["trader_wallet"] == "0xabc")
        & (result["condition_id"] == "cond-1")
        & (result["outcome_side"] == "YES")
    ]
    assert len(open_yes) == 1
    assert open_yes.iloc[0]["source_wallet_position_event"] == "SCALE_IN"

    open_no = result[
        (result["entry_intent"] == "OPEN_LONG")
        & (result["outcome_side"] == "NO")
    ]
    assert len(open_no) == 1

    close_yes = result[result["entry_intent"] == "CLOSE_LONG"]
    assert len(close_yes) == 1
    assert close_yes.iloc[0]["source_wallet_position_event"] == "FULL_EXIT"


def test_keep_latest_open_long_state_uses_market_slug_when_condition_missing():
    df = pd.DataFrame(
        [
            {
                "trader_wallet": "0xdef",
                "market_slug": "btc-updown-5m-123",
                "market_title": "BTC Up or Down",
                "outcome_side": "YES",
                "entry_intent": "OPEN_LONG",
                "source_wallet_position_event": "NEW_ENTRY",
                "timestamp": "2026-04-08T11:00:00Z",
            },
            {
                "trader_wallet": "0xdef",
                "market_slug": "btc-updown-5m-123",
                "market_title": "BTC Up or Down",
                "outcome_side": "YES",
                "entry_intent": "OPEN_LONG",
                "source_wallet_position_event": "SCALE_IN",
                "timestamp": "2026-04-08T11:03:00Z",
            },
        ]
    )

    result = _keep_latest_open_long_state_per_wallet_market_side(df)

    assert len(result) == 1
    assert result.iloc[0]["source_wallet_position_event"] == "SCALE_IN"
