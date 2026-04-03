from datetime import datetime, timedelta, timezone

from trade_lifecycle import TradeLifecycle
from trade_manager import TradeManager


def _open_trade(outcome_side: str, entry_price: float = 0.50, current_price: float = 0.505) -> TradeLifecycle:
    trade = TradeLifecycle(
        market="BTC Test",
        token_id="token-1",
        condition_id="cond-1",
        outcome_side=outcome_side,
    )
    trade.on_signal(
        {
            "confidence": 0.72,
            "signal_label": "STRONG PAPER OPPORTUNITY",
            "btc_trend_bias": "LONG" if outcome_side == "YES" else "SHORT",
            "alligator_alignment": "BULLISH" if outcome_side == "YES" else "BEARISH",
            "adx_value": 28.0,
            "adx_threshold": 18.0,
            "anchored_vwap": 66600.0,
            "fractal_trigger_direction": "LONG" if outcome_side == "YES" else "SHORT",
        }
    )
    trade.enter(size_usdc=5.0, entry_price=entry_price)
    trade.update_market(current_price)
    trade.opened_at = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    return trade


def test_process_exits_closes_long_on_alligator_reversal(tmp_path):
    manager = TradeManager(logs_dir=str(tmp_path))
    trade = _open_trade("YES", entry_price=0.50, current_price=0.505)
    key = manager._compose_trade_key(token_id=trade.token_id, condition_id=trade.condition_id, outcome_side=trade.outcome_side, market=trade.market)
    manager.active_trades[key] = trade

    closed = manager.process_exits(
        datetime.now(timezone.utc),
        persist_closed=False,
        technical_context={
            "alligator_alignment": "BEARISH",
            "price_above_anchored_vwap": False,
            "price_below_anchored_vwap": True,
            "adx_value": 24.0,
            "adx_threshold": 18.0,
        },
    )

    assert len(closed) == 1
    assert closed[0].close_reason == "technical_alligator_reversal"


def test_process_exits_closes_short_on_vwap_recross(tmp_path):
    manager = TradeManager(logs_dir=str(tmp_path))
    trade = _open_trade("NO", entry_price=0.45, current_price=0.448)
    key = manager._compose_trade_key(token_id=trade.token_id, condition_id=trade.condition_id, outcome_side=trade.outcome_side, market=trade.market)
    manager.active_trades[key] = trade

    closed = manager.process_exits(
        datetime.now(timezone.utc),
        persist_closed=False,
        technical_context={
            "alligator_alignment": "BEARISH",
            "price_above_anchored_vwap": True,
            "price_below_anchored_vwap": False,
            "adx_value": 21.0,
            "adx_threshold": 18.0,
        },
    )

    assert len(closed) == 1
    assert closed[0].close_reason == "technical_vwap_recross"


def test_process_exits_closes_on_adx_weakening_before_time_stop(tmp_path):
    manager = TradeManager(logs_dir=str(tmp_path))
    trade = _open_trade("YES", entry_price=0.50, current_price=0.506)
    key = manager._compose_trade_key(token_id=trade.token_id, condition_id=trade.condition_id, outcome_side=trade.outcome_side, market=trade.market)
    manager.active_trades[key] = trade

    closed = manager.process_exits(
        datetime.now(timezone.utc),
        persist_closed=False,
        technical_context={
            "alligator_alignment": "BULLISH",
            "price_above_anchored_vwap": True,
            "price_below_anchored_vwap": False,
            "adx_value": 17.0,
            "adx_threshold": 18.0,
        },
    )

    assert len(closed) == 1
    assert closed[0].close_reason == "technical_adx_weakening"
