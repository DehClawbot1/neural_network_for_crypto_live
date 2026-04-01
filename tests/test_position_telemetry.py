import pandas as pd

from position_telemetry import PositionTelemetry


def test_position_telemetry_flags_reversal_after_runup(tmp_path):
    telemetry = PositionTelemetry(logs_dir=tmp_path)
    rows = [
        {"timestamp": "2026-04-01T10:00:00Z", "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.50},
        {"timestamp": "2026-04-01T10:01:00Z", "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.525},
        {"timestamp": "2026-04-01T10:02:00Z", "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.535},
        {"timestamp": "2026-04-01T10:03:00Z", "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.522},
        {"timestamp": "2026-04-01T10:04:00Z", "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.515},
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
        {"timestamp": "2026-04-01T10:00:00Z", "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.505, "spread_pct": 0.01, "mark_source": "orderbook_best_bid"},
        {"timestamp": "2026-04-01T10:01:00Z", "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.506, "spread_pct": 0.01, "mark_source": "scored_fallback_price"},
        {"timestamp": "2026-04-01T10:02:00Z", "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.507, "spread_pct": 0.01, "mark_source": "scored_fallback_price"},
        {"timestamp": "2026-04-01T10:03:00Z", "position_key": "t|c|YES", "token_id": "t", "condition_id": "c", "outcome_side": "YES", "market": "BTC", "entry_price": 0.50, "mark_price": 0.508, "spread_pct": 0.01, "mark_source": "scored_fallback_price"},
    ]
    pd.DataFrame(rows).to_csv(telemetry.snapshots_file, index=False)

    metrics = telemetry.build_trajectory_metrics(hours=48)
    state = metrics["t|c|YES"]

    assert state["liquidity_stress_signal"] is True
    assert state["trajectory_state"] == "liquidity_stress"
