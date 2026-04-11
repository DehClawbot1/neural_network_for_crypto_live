from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

import pandas as pd
import logging

from leaderboard_service import PolymarketLeaderboardService
from weather_temperature_strategy import WeatherTemperatureStrategy


class _FakeTrade:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.state = "OPEN"
        self.close_reason = None

    def close(self, exit_price, reason):
        self.current_price = exit_price
        self.close_reason = reason
        self.state = "CLOSED"


def test_build_event_rows_emits_reversal_exit_and_entry(tmp_path):
    strategy = WeatherTemperatureStrategy(logs_dir=str(tmp_path))
    previous_snapshot = pd.DataFrame(
        [
            {
                "trader_wallet": "0xabc",
                "market_title": "Will the highest temperature in NYC be 64°F or higher on March 10?",
                "market_slug": "nyc-64f",
                "condition_id": "cond-1",
                "outcome_side": "YES",
                "source_wallet_position_size": 12.0,
                "source_wallet_reference_ts": "2026-03-09T12:00:00+00:00",
                "wallet_watchlist_approved": True,
                "wallet_quality_score": 0.9,
                "weather_location": "NYC",
                "weather_event_date_local": "2026-03-10",
                "market_family": "weather_temperature_threshold",
            }
        ]
    )
    current_snapshot = pd.DataFrame(
        [
            {
                "trader_wallet": "0xabc",
                "market_title": "Will the highest temperature in NYC be 64°F or higher on March 10?",
                "market_slug": "nyc-64f",
                "condition_id": "cond-1",
                "outcome_side": "NO",
                "source_wallet_position_size": 9.0,
                "source_wallet_reference_ts": "2026-03-09T12:10:00+00:00",
                "wallet_watchlist_approved": True,
                "wallet_quality_score": 0.9,
                "weather_location": "NYC",
                "weather_event_date_local": "2026-03-10",
                "market_family": "weather_temperature_threshold",
            }
        ]
    )

    events = strategy._build_event_rows(current_snapshot, previous_snapshot)

    assert set(events["source_wallet_position_event"]) == {"REVERSAL_EXIT", "REVERSAL_ENTRY"}
    assert set(events["entry_intent"]) == {"CLOSE_LONG", "OPEN_LONG"}


def test_score_candidates_generates_weather_specific_scored_frame(tmp_path):
    strategy = WeatherTemperatureStrategy(logs_dir=str(tmp_path))
    signals_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-03-09T12:00:00+00:00",
                "market_title": "Will the highest temperature in NYC be 64°F or higher on March 10?",
                "market_slug": "nyc-64f",
                "condition_id": "cond-1",
                "token_id": "tok-1",
                "entry_intent": "OPEN_LONG",
                "outcome_side": "YES",
                "market_family": "weather_temperature_threshold",
                "weather_question_type": "threshold",
                "weather_parseable": True,
                "forecast_ready": True,
                "forecast_stale": False,
                "forecast_p_hit_interval": 0.74,
                "forecast_margin_to_lower_c": 2.1,
                "forecast_margin_to_upper_c": None,
                "forecast_uncertainty_c": 1.3,
                "forecast_drift_c": 0.1,
                "weather_market_probability": 0.51,
                "wallet_quality_score": 0.88,
                "source_wallet_direction_confidence": 0.83,
                "wallet_agreement_score": 0.67,
                "source_wallet_size_delta_ratio": 0.42,
                "wallet_state_gate_pass": True,
                "liquidity": 30000.0,
                "volume": 9000.0,
                "liquidity_score": 0.45,
                "best_bid": 0.50,
                "best_ask": 0.54,
                "current_price": 0.51,
                "end_date": "2026-03-10T23:59:00+00:00",
            }
        ]
    )

    scored = strategy.score_candidates(signals_df)

    assert len(scored) == 1
    row = scored.iloc[0]
    assert row["entry_model_family"] == "weather_temperature_hybrid"
    assert bool(row["weather_entry_allowed_by_forecast"]) is True
    assert row["weather_min_forecast_edge"] == strategy.min_forecast_edge
    assert row["weather_max_spread"] == strategy.max_spread
    assert row["confidence"] > 0.0


def test_apply_active_exit_rules_closes_weather_trade_when_forecast_turns_stale(tmp_path):
    strategy = WeatherTemperatureStrategy(logs_dir=str(tmp_path))
    strategy.forecast_service.fetch_market_forecast = lambda row: {
        "forecast_ready": False,
        "forecast_stale": True,
        "forecast_p_hit_interval": 0.5,
    }
    trade = _FakeTrade(
        token_id="tok-1",
        condition_id="cond-1",
        outcome_side="YES",
        market="Will the highest temperature in London be 10°C or higher on January 28?",
        market_family="weather_temperature_threshold",
        current_price=0.61,
        entry_price=0.45,
        weather_location="London",
        weather_event_date_local="2026-01-28",
    )
    trade_manager = SimpleNamespace(active_trades={"tok-1": trade})

    exit_events = strategy.apply_active_exit_rules(trade_manager, pd.DataFrame())

    assert len(exit_events) == 1
    assert trade.state == "CLOSED"
    assert trade.close_reason == "weather_forecast_stale"


def test_build_event_signal_prefers_current_reference_ts_over_old_last_close(tmp_path):
    strategy = WeatherTemperatureStrategy(logs_dir=str(tmp_path))
    now_utc = datetime.now(timezone.utc)
    signal = strategy._build_event_signal(
        {
            "wallet_watchlist_approved": True,
            "wallet_quality_score": 0.9,
            "source_wallet_position_size": 10.0,
            "source_wallet_reference_ts": now_utc.isoformat(),
            "source_wallet_last_close": (now_utc - timedelta(days=30)).isoformat(),
        },
        entry_intent="OPEN_LONG",
        position_event="NEW_ENTRY",
        net_increase=True,
        size_delta=10.0,
    )

    assert bool(signal["wallet_state_gate_pass"]) is True
    assert signal["wallet_state_gate_reason"] == ""


def test_load_watchlist_uses_env_fallback_when_csv_is_empty(tmp_path, monkeypatch):
    watchlist_path = tmp_path / "weather_wallet_watchlist.csv"
    watchlist_path.write_text("wallet,label,enabled,min_wallet_score,region_scope\n", encoding="utf-8")
    monkeypatch.setenv(
        "WEATHER_APPROVED_WALLETS",
        "0xabc123|1pixel|true|0.72|nyc\n0xdef456|london-pro|true|0.65|london",
    )
    empty_service = PolymarketLeaderboardService(logs_dir=str(tmp_path))
    empty_service.fetch_leaderboard = lambda **kwargs: pd.DataFrame()
    strategy = WeatherTemperatureStrategy(
        logs_dir=str(tmp_path),
        watchlist_path=str(watchlist_path),
        leaderboard_service=empty_service,
    )
    watchlist = strategy.load_watchlist()

    assert len(watchlist.index) == 2
    assert set(watchlist["wallet"].astype(str)) == {"0xabc123", "0xdef456"}
    assert round(float(watchlist.loc[watchlist["wallet"] == "0xabc123", "min_wallet_score"].iloc[0]), 2) == 0.72
    assert str(watchlist.loc[watchlist["wallet"] == "0xdef456", "region_scope"].iloc[0]) == "london"


def test_load_watchlist_prefers_dynamic_weather_leaderboard_and_keeps_overrides(tmp_path):
    watchlist_path = tmp_path / "weather_wallet_watchlist.csv"
    watchlist_path.write_text(
        "wallet,label,enabled,min_wallet_score,region_scope\n0xmanual,manual,true,0.77,nyc\n",
        encoding="utf-8",
    )
    service = PolymarketLeaderboardService(logs_dir=str(tmp_path))
    service.fetch_leaderboard = lambda **kwargs: pd.DataFrame(
        [
            {
                "wallet": "0xleader",
                "label": "leader",
                "enabled": True,
                "min_wallet_score": 0.6,
                "region_scope": "",
                "approved": True,
                "source": "leaderboard_api",
            }
        ]
    )
    strategy = WeatherTemperatureStrategy(
        logs_dir=str(tmp_path),
        watchlist_path=str(watchlist_path),
        leaderboard_service=service,
    )

    watchlist = strategy.load_watchlist()

    assert set(watchlist["wallet"].astype(str)) == {"0xleader", "0xmanual"}
    assert "leaderboard_api" in set(watchlist["source"].astype(str))
    assert "manual_override_csv" in set(watchlist["source"].astype(str))


def test_load_watchlist_logs_summary_only_when_changed(tmp_path, caplog):
    service = PolymarketLeaderboardService(logs_dir=str(tmp_path))
    service.fetch_leaderboard = lambda **kwargs: pd.DataFrame(
        [
            {"wallet": "0xleader1", "approved": True, "source": "leaderboard_api"},
            {"wallet": "0xleader2", "approved": True, "source": "leaderboard_api"},
        ]
    )
    strategy = WeatherTemperatureStrategy(
        logs_dir=str(tmp_path),
        watchlist_path=str(tmp_path / "missing.csv"),
        leaderboard_service=service,
    )

    with caplog.at_level(logging.INFO):
        strategy.load_watchlist()
        strategy.load_watchlist()

    messages = [record.message for record in caplog.records if "Weather wallet source loaded" in record.message]
    assert len(messages) == 1
