import pandas as pd

from trade_lifecycle import TradeLifecycle
from trade_manager import TradeManager


def _make_open_btc_trade() -> TradeLifecycle:
    trade = TradeLifecycle(
        market="Bitcoin Up or Down - Test Window",
        token_id="btc-token-1",
        condition_id="btc-cond-1",
        outcome_side="YES",
    )
    trade.on_signal(
        {
            "market": "Bitcoin Up or Down - Test Window",
            "market_family": "btc_directional_intraday",
            "confidence": 0.61,
            "signal_label": "HIGHEST-RANKED PAPER SIGNAL",
            "entry_model_family": "runtime_live_stack",
            "entry_model_version": "test-v1",
        }
    )
    trade.enter(size_usdc=5.0, entry_price=0.48)
    trade.current_price = 0.44
    trade.unrealized_pnl = trade.shares * (trade.current_price - trade.entry_price)
    return trade


def _make_open_weather_trade() -> TradeLifecycle:
    trade = TradeLifecycle(
        market="Will the high temperature in NYC be above 70F?",
        token_id="weather-token-1",
        condition_id="weather-cond-1",
        outcome_side="YES",
    )
    trade.on_signal(
        {
            "market": "Will the high temperature in NYC be above 70F?",
            "market_family": "weather_temperature_threshold",
            "confidence": 0.67,
            "signal_label": "STRONG WEATHER OPPORTUNITY",
            "entry_model_family": "weather_temperature_hybrid",
            "entry_model_version": "test-weather-v1",
        }
    )
    trade.enter(size_usdc=4.0, entry_price=0.52)
    trade.current_price = 0.49
    trade.unrealized_pnl = trade.shares * (trade.current_price - trade.entry_price)
    return trade


def _persist_open_trade(manager: TradeManager, trade: TradeLifecycle):
    key = manager._compose_trade_key(
        token_id=trade.token_id,
        condition_id=trade.condition_id,
        outcome_side=trade.outcome_side,
        market=trade.market,
    )
    manager.active_trades[key] = trade
    manager.persist_open_positions()
    manager.db.execute(
        "UPDATE positions SET status = 'OPEN' WHERE token_id = ? AND condition_id = ? AND outcome_side = ?",
        (trade.token_id, trade.condition_id, trade.outcome_side),
    )


def test_btc_reconciliation_requires_repeated_missing_confirmations_and_persists_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("BTC_RECONCILIATION_MISSING_CONFIRMATIONS", "3")
    manager = TradeManager(logs_dir=tmp_path)

    trade = _make_open_btc_trade()
    _persist_open_trade(manager, trade)

    assert manager._close_absent_ledger_positions(pd.DataFrame()) == 0
    assert manager._close_absent_ledger_positions(pd.DataFrame()) == 0
    assert manager._close_absent_ledger_positions(pd.DataFrame()) == 1

    closed_df = pd.read_csv(tmp_path / "closed_positions.csv", engine="python", on_bad_lines="skip")
    closed_row = closed_df.iloc[0]
    assert closed_row["close_reason"] == "external_manual_close"
    assert bool(closed_row["reconciliation_close_flag"])
    assert closed_row["reconciliation_reason"] == "missing_from_reconciled_snapshot"
    assert int(closed_row["reconciliation_snapshot_open_count"]) == 0
    assert int(closed_row["reconciliation_local_open_count"]) == 1
    assert int(closed_row["reconciliation_missing_confirmations"]) == 3
    assert int(closed_row["reconciliation_required_confirmations"]) == 3
    assert closed_row["reconciliation_presence_key"] == "btc-token-1|btc-cond-1|YES"

    attribution_df = pd.read_csv(tmp_path / "btc_trade_attribution.csv", engine="python", on_bad_lines="skip")
    attribution_row = attribution_df.iloc[0]
    assert attribution_row["close_reason"] == "external_manual_close"
    assert bool(attribution_row["reconciliation_close_flag"])
    assert attribution_row["reconciliation_reason"] == "missing_from_reconciled_snapshot"
    assert int(attribution_row["reconciliation_snapshot_open_count"]) == 0
    assert int(attribution_row["reconciliation_local_open_count"]) == 1
    assert int(attribution_row["reconciliation_missing_confirmations"]) == 3
    assert int(attribution_row["reconciliation_required_confirmations"]) == 3
    assert attribution_row["reconciliation_presence_key"] == "btc-token-1|btc-cond-1|YES"


def test_non_btc_reconciliation_still_closes_on_first_missing_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("BTC_RECONCILIATION_MISSING_CONFIRMATIONS", "3")
    manager = TradeManager(logs_dir=tmp_path)

    trade = _make_open_weather_trade()
    _persist_open_trade(manager, trade)

    assert manager._close_absent_ledger_positions(pd.DataFrame()) == 1

    closed_df = pd.read_csv(tmp_path / "closed_positions.csv", engine="python", on_bad_lines="skip")
    closed_row = closed_df.iloc[0]
    assert closed_row["market_family"] == "weather_temperature_threshold"
    assert closed_row["close_reason"] == "external_manual_close"
    assert int(closed_row["reconciliation_missing_confirmations"]) == 1
    assert int(closed_row["reconciliation_required_confirmations"]) == 1
