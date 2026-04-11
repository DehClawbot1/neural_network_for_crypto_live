from datetime import datetime, timedelta, timezone

import pandas as pd

from position_telemetry import PositionTelemetry


def _recent_ts(minutes_ago: int) -> str:
    """Return an ISO timestamp *minutes_ago* minutes before now (UTC)."""
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def test_position_telemetry_flags_reversal_after_runup(tmp_path):
    telemetry = PositionTelemetry(logs_dir=tmp_path)
    rows = [
        {"timestamp": _recent_ts(5), "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.50},
        {"timestamp": _recent_ts(4), "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.525},
        {"timestamp": _recent_ts(3), "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.535},
        {"timestamp": _recent_ts(2), "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.522},
        {"timestamp": _recent_ts(1), "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.515},
    ]
    pd.DataFrame(rows).to_csv(telemetry.snapshots_file, index=False)

    metrics = telemetry.build_trajectory_metrics(hours=48)
    state = metrics["t|c|YES"]

    assert state["reversal_exit_signal"] is True
    assert state["trajectory_state"] == "reversal_exit"


def test_position_telemetry_capture_writes_portfolio_curve(tmp_path):
    telemetry = PositionTelemetry(logs_dir=tmp_path)
    positions = pd.DataFrame(
        [
            {
                "token_id": "t",
                "condition_id": "c",
                "outcome_side": "YES",
                "market": "BTC",
                "entry_price": 0.50,
                "current_price": 0.53,
                "shares": 10,
                "market_value": 5.3,
                "unrealized_pnl": 0.3,
                "realized_pnl": 0.0,
                "status": "OPEN",
            }
        ]
    )

    telemetry.capture_positions(positions)

    assert telemetry.snapshots_file.exists()
    assert telemetry.portfolio_file.exists()
    portfolio = pd.read_csv(telemetry.portfolio_file)
    assert float(portfolio.iloc[-1]["unrealized_pnl"]) == 0.3


def test_position_telemetry_flags_liquidity_stress_from_fallback_marks(tmp_path):
    telemetry = PositionTelemetry(logs_dir=tmp_path)
    rows = [
        {"timestamp": _recent_ts(4), "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.505, "spread_pct": 0.01, "mark_source": "orderbook_best_bid"},
        {"timestamp": _recent_ts(3), "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.506, "spread_pct": 0.01, "mark_source": "scored_fallback_price"},
        {"timestamp": _recent_ts(2), "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.507, "spread_pct": 0.01, "mark_source": "scored_fallback_price"},
        {"timestamp": _recent_ts(1), "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.508, "spread_pct": 0.01, "mark_source": "scored_fallback_price"},
    ]
    pd.DataFrame(rows).to_csv(telemetry.snapshots_file, index=False)

    metrics = telemetry.build_trajectory_metrics(hours=48)
    state = metrics["t|c|YES"]

    assert state["liquidity_stress_signal"] is True
    assert state["trajectory_state"] == "liquidity_stress"


def test_position_telemetry_uses_current_positions_without_duplicate_persist(tmp_path):
    telemetry = PositionTelemetry(logs_dir=tmp_path)
    seed = pd.DataFrame(
        [
            {"timestamp": _recent_ts(4), "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.50, "mark_source": "orderbook_best_bid"},
            {"timestamp": _recent_ts(3), "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.525, "mark_source": "orderbook_best_bid"},
            {"timestamp": _recent_ts(2), "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.535, "mark_source": "orderbook_best_bid"},
            {"timestamp": _recent_ts(1), "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.522, "mark_source": "orderbook_best_bid"},
        ]
    )
    seed.to_csv(telemetry.snapshots_file, index=False)

    current = pd.DataFrame(
        [
            {"token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.515, "current_price": 0.515, "mark_source": "orderbook_best_bid"}
        ]
    )
    metrics = telemetry.build_trajectory_metrics(current, hours=48)
    state = metrics["t|c|YES"]

    assert state["reversal_exit_signal"] is True
    assert len(pd.read_csv(telemetry.snapshots_file)) == 4


def test_position_telemetry_capture_persists_enriched_fields(tmp_path):
    telemetry = PositionTelemetry(logs_dir=tmp_path)
    positions = pd.DataFrame(
        [
            {
                "token_id": "t",
                "condition_id": "c",
                "outcome_side": "YES",
                "market": "BTC",
                "entry_price": 0.50,
                "current_price": 0.53,
                "mark_price": 0.53,
                "shares": 10,
                "market_value": 5.3,
                "unrealized_pnl": 0.3,
                "realized_pnl": 0.0,
                "status": "OPEN",
                "trajectory_state": "profit_lock",
                "drawdown_from_peak": 0.02,
                "recent_return_3": -0.01,
                "runup_from_entry": 0.06,
                "volatility_short": 0.02,
                "fallback_ratio": 0.25,
                "spread_pct": 0.01,
            }
        ]
    )

    telemetry.capture_positions(positions)
    snapshots = pd.read_csv(telemetry.snapshots_file)

    assert snapshots.iloc[-1]["trajectory_state"] == "profit_lock"
    assert float(snapshots.iloc[-1]["volatility_short"]) == 0.02
