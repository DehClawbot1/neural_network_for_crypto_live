from unittest.mock import patch

import pandas as pd

from position_manager import PositionManager


@patch("position_manager.MarketPriceService")
def test_open_position_success(mock_price_service_cls, tmp_path):
    pm = PositionManager(logs_dir=tmp_path)

    signal = {
        "market_title": "Test Market",
        "token_id": "token_123",
        "outcome_side": "YES",
        "confidence": 0.85,
        "trader_wallet": "0xabc",
    }

    result = pm.open_position(signal, size_usdc=10.0, fill_price=0.5)

    assert result is True
    open_positions = pm.get_open_positions()
    assert len(open_positions) == 1
    assert open_positions.iloc[0]["token_id"] == "token_123"
    assert float(open_positions.iloc[0]["shares"]) == 20.0
    assert float(open_positions.iloc[0]["confidence_at_entry"]) == 0.85
    assert open_positions.iloc[0]["status"] == "OPEN"


@patch("position_manager.MarketPriceService")
def test_open_position_respects_max_limit(mock_price_service_cls, tmp_path):
    pm = PositionManager(logs_dir=tmp_path, max_open_positions=1)

    signal1 = {"market_title": "M1", "token_id": "t1", "trader_wallet": "0x1"}
    signal2 = {"market_title": "M2", "token_id": "t2", "trader_wallet": "0x2"}

    assert pm.open_position(signal1, 10.0, 0.5) is True
    assert pm.open_position(signal2, 10.0, 0.5) is False


@patch("position_manager.MarketPriceService")
def test_close_position_uses_quote_and_removes_open_position(mock_price_service_cls, tmp_path):
    mock_price_service = mock_price_service_cls.return_value
    mock_price_service.get_quote.return_value = {"best_bid": 0.75, "price": 0.76}

    pm = PositionManager(logs_dir=tmp_path, fee_rate=0.0, slippage_rate=0.0)
    signal = {"market_title": "Test Market", "token_id": "token_123", "outcome_side": "YES", "trader_wallet": "0xabc"}
    pm.open_position(signal, size_usdc=10.0, fill_price=0.5)

    position_to_close = pm.get_open_positions().iloc[0].to_dict()
    closed_df = pm.close_position(position_to_close, reason="take_profit")

    assert not closed_df.empty
    assert closed_df.iloc[0]["status"] == "CLOSED"
    assert closed_df.iloc[0]["close_reason"] == "take_profit"
    assert float(closed_df.iloc[0]["exit_price"]) == 0.75
    assert pm.get_open_positions().empty


@patch("position_manager.MarketPriceService")
def test_update_mark_to_market_stores_spread_and_bid_size(mock_price_service_cls, tmp_path):
    mock_price_service = mock_price_service_cls.return_value
    mock_price_service.get_batch_prices.return_value = {
        "token_123": {"price": 0.62, "spread": 0.03, "best_bid_size": 150.0}
    }

    pm = PositionManager(logs_dir=tmp_path)
    signal = {"market_title": "Test Market", "token_id": "token_123", "outcome_side": "YES", "confidence": 0.85, "trader_wallet": "0xabc"}
    pm.open_position(signal, size_usdc=10.0, fill_price=0.5)

    updated = pm.update_mark_to_market()

    assert not updated.empty
    row = updated.iloc[0]
    assert float(row["current_price"]) == 0.62
    assert float(row["spread"]) == 0.03
    assert float(row["bid_size"]) == 150.0


@patch("position_manager.MarketPriceService")
def test_apply_exit_rules_closes_on_confidence_drop(mock_price_service_cls, tmp_path):
    pm = PositionManager(logs_dir=tmp_path, fee_rate=0.0, slippage_rate=0.0)
    signal = {"market_title": "Test Market", "token_id": "token_123", "outcome_side": "YES", "confidence": 0.85, "trader_wallet": "0xabc"}
    pm.open_position(signal, size_usdc=10.0, fill_price=0.5)

    positions = pd.read_csv(pm.positions_file)
    positions.loc[0, "confidence"] = 0.2
    positions.loc[0, "current_price"] = 0.51
    positions.loc[0, "peak_price"] = 0.52
    positions.to_csv(pm.positions_file, index=False)

    closed = pm.apply_exit_rules(pd.DataFrame())

    assert not closed.empty
    assert closed.iloc[0]["status"] == "CLOSED"
    assert closed.iloc[0]["close_reason"] == "confidence_drop"


@patch("position_manager.MarketPriceService")
def test_position_hold_and_exit_rules(mock_price_service_cls, tmp_path):
    mock_price_service = mock_price_service_cls.return_value
    pm = PositionManager(logs_dir=tmp_path, take_profit_roi_pct=0.10, fee_rate=0.0, slippage_rate=0.0)
    signal = {"token_id": "tok-1", "market": "BTC Test", "outcome_side": "YES", "trader_wallet": "0xwhale"}

    assert pm.open_position(signal, size_usdc=100.0, fill_price=0.5) is True

    mock_price_service.get_batch_prices.return_value = {"tok-1": {"price": 0.5}}
    pm.update_mark_to_market()

    open_pos = pm.get_open_positions()
    assert float(open_pos.iloc[0]["unrealized_pnl"]) == 0.0

    mock_price_service.get_batch_prices.return_value = {"tok-1": {"price": 0.6}}
    pm.update_mark_to_market()

    closed = pm.apply_exit_rules()

    assert len(closed) == 1
    assert closed.iloc[0]["close_reason"] == "take_profit_roi"
    assert pm.get_open_positions().empty
