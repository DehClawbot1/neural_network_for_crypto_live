from types import SimpleNamespace

import pandas as pd

from signal_engine import SignalEngine
from wallet_state_engine import (
    WalletStateEngine,
    resolve_source_wallet_reduce_fraction,
    should_convert_reduce_to_exit,
    source_wallet_signal_matches_trade,
)


def test_wallet_state_engine_emits_new_entry_scale_and_full_exit():
    engine = WalletStateEngine()
    raw = pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "trader_wallet": "0xabc",
                "condition_id": "cond-1",
                "market_slug": "btc-above-60k",
                "market_title": "BTC Above 60k",
                "order_side": "BUY",
                "outcome_side": "YES",
                "price": 0.52,
                "size": 100,
                "timestamp": "2026-04-08T10:00:00Z",
                "wallet_quality_score": 0.8,
                "wallet_watchlist_approved": True,
            },
            {
                "trade_id": "t2",
                "trader_wallet": "0xabc",
                "condition_id": "cond-1",
                "market_slug": "btc-above-60k",
                "market_title": "BTC Above 60k",
                "order_side": "BUY",
                "outcome_side": "YES",
                "price": 0.55,
                "size": 50,
                "timestamp": "2026-04-08T10:05:00Z",
                "wallet_quality_score": 0.8,
                "wallet_watchlist_approved": True,
            },
            {
                "trade_id": "t3",
                "trader_wallet": "0xabc",
                "condition_id": "cond-1",
                "market_slug": "btc-above-60k",
                "market_title": "BTC Above 60k",
                "order_side": "SELL",
                "outcome_side": "YES",
                "price": 0.60,
                "size": 150,
                "timestamp": "2026-04-08T10:10:00Z",
                "wallet_quality_score": 0.8,
                "wallet_watchlist_approved": True,
            },
        ]
    )

    signals = engine.build_state_signals(raw)

    assert list(signals["source_wallet_position_event"]) == ["FULL_EXIT", "SCALE_IN", "NEW_ENTRY"]
    assert signals.iloc[0]["entry_intent"] == "CLOSE_LONG"
    assert signals.iloc[1]["entry_intent"] == "OPEN_LONG"
    assert bool(signals.iloc[1]["source_wallet_net_position_increased"]) is True


def test_wallet_state_engine_emits_reversal_exit_and_entry():
    engine = WalletStateEngine()
    raw = pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "trader_wallet": "0xabc",
                "condition_id": "cond-2",
                "market_slug": "btc-above-65k",
                "market_title": "BTC Above 65k",
                "order_side": "BUY",
                "outcome_side": "YES",
                "price": 0.44,
                "size": 100,
                "timestamp": "2026-04-08T11:00:00Z",
                "wallet_quality_score": 0.75,
                "wallet_watchlist_approved": True,
            },
            {
                "trade_id": "t2",
                "trader_wallet": "0xabc",
                "condition_id": "cond-2",
                "market_slug": "btc-above-65k",
                "market_title": "BTC Above 65k",
                "order_side": "BUY",
                "outcome_side": "NO",
                "price": 0.58,
                "size": 120,
                "timestamp": "2026-04-08T11:04:00Z",
                "wallet_quality_score": 0.75,
                "wallet_watchlist_approved": True,
            },
        ]
    )

    signals = engine.build_state_signals(raw)
    events = set(signals["source_wallet_position_event"].tolist())

    assert "REVERSAL_EXIT" in events
    assert "REVERSAL_ENTRY" in events
    reversal_exit = signals.loc[signals["source_wallet_position_event"] == "REVERSAL_EXIT"].iloc[0]
    reversal_entry = signals.loc[signals["source_wallet_position_event"] == "REVERSAL_ENTRY"].iloc[0]
    assert reversal_exit["entry_intent"] == "CLOSE_LONG"
    assert reversal_exit["outcome_side"] == "YES"
    assert reversal_entry["entry_intent"] == "OPEN_LONG"
    assert reversal_entry["outcome_side"] == "NO"


def test_signal_engine_marks_wallet_gate_failure_without_owning_hard_veto():
    engine = SignalEngine()
    base_row = {
        "entry_intent": "OPEN_LONG",
        "outcome_side": "YES",
        "wallet_watchlist_approved": True,
        "wallet_state_gate_pass": True,
        "wallet_quality_score": 0.30,
        "wallet_state_freshness_score": 0.10,
        "source_wallet_net_position_increased": False,
        "wallet_conflict_with_stronger": True,
        "wallet_state_confidence": 0.20,
        "wallet_size_change_score": 0.10,
        "wallet_agreement_score": 0.10,
        "wallet_distance_score": 0.90,
        "whale_pressure": 0.9,
        "market_structure_score": 0.8,
        "volatility_risk": 0.2,
        "time_decay_score": 0.1,
        "btc_network_activity_score": 0.6,
        "btc_network_stress_score": 0.4,
        "p_tp_before_sl": 0.65,
        "expected_return": 0.03,
        "edge_score": 0.02,
    }
    scored_ok = engine.score_row(base_row)
    scored_fail = engine.score_row({**base_row, "wallet_state_gate_pass": False})

    assert bool(scored_ok["wallet_entry_gate_fail"]) is False
    assert bool(scored_fail["wallet_entry_gate_fail"]) is True
    assert scored_ok["confidence"] == scored_fail["confidence"]
    assert scored_ok["action_code"] == scored_fail["action_code"]


def test_source_wallet_signal_matches_trade_wallet_and_side():
    trade = SimpleNamespace(
        source_wallet="0xabc",
        token_id="tok-1",
        condition_id="cond-3",
        outcome_side="YES",
    )
    matching_signal = {
        "trader_wallet": "0xabc",
        "token_id": "tok-1",
        "condition_id": "cond-3",
        "outcome_side": "YES",
    }
    conflicting_signal = {
        "trader_wallet": "0xdef",
        "token_id": "tok-1",
        "condition_id": "cond-3",
        "outcome_side": "YES",
    }

    assert source_wallet_signal_matches_trade(matching_signal, trade) is True
    assert source_wallet_signal_matches_trade(conflicting_signal, trade) is False


def test_wallet_state_engine_conflict_updates_gate_owner_state():
    engine = WalletStateEngine()
    raw = pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "trader_wallet": "0xstrong",
                "condition_id": "cond-4",
                "market_slug": "btc-above-70k",
                "market_title": "BTC Above 70k",
                "order_side": "BUY",
                "outcome_side": "YES",
                "price": 0.57,
                "size": 100,
                "timestamp": "2026-04-08T12:00:00Z",
                "wallet_quality_score": 0.95,
                "wallet_watchlist_approved": True,
            },
            {
                "trade_id": "t2",
                "trader_wallet": "0xweak",
                "condition_id": "cond-4",
                "market_slug": "btc-above-70k",
                "market_title": "BTC Above 70k",
                "order_side": "BUY",
                "outcome_side": "NO",
                "price": 0.43,
                "size": 15,
                "timestamp": "2026-04-08T12:01:00Z",
                "wallet_quality_score": 0.35,
                "wallet_watchlist_approved": True,
            },
        ]
    )

    signals = engine.build_state_signals(raw)
    weak_row = signals.loc[signals["trader_wallet"] == "0xweak"].iloc[0]

    assert bool(weak_row["wallet_conflict_with_stronger"]) is True
    assert bool(weak_row["wallet_state_gate_pass"]) is False
    assert "conflict_with_stronger_wallet" in str(weak_row["wallet_state_gate_reason"])


def test_resolve_source_wallet_reduce_fraction_clips_reasonably():
    assert resolve_source_wallet_reduce_fraction({"source_wallet_reduce_fraction": 0.8}) == 0.8
    assert resolve_source_wallet_reduce_fraction({"source_wallet_reduce_fraction": 0.0}) == 0.5
    assert resolve_source_wallet_reduce_fraction({"source_wallet_reduce_fraction": 1.7}) == 1.0


def test_should_convert_reduce_to_exit_blocks_micro_reductions_and_dust_remainders():
    assert (
        should_convert_reduce_to_exit(
            total_shares=100.0,
            reduce_fraction=0.5,
            reference_price=0.55,
            min_reduce_notional=2.5,
            min_remainder_notional=2.5,
        )
        is False
    )
    assert (
        should_convert_reduce_to_exit(
            total_shares=4.0,
            reduce_fraction=0.25,
            reference_price=0.40,
            min_reduce_notional=2.5,
            min_remainder_notional=2.5,
        )
        is True
    )
    assert (
        should_convert_reduce_to_exit(
            total_shares=10.0,
            reduce_fraction=0.9,
            reference_price=0.50,
            min_reduce_notional=2.5,
            min_remainder_notional=2.5,
        )
        is True
    )
