import tempfile
import warnings
from pathlib import Path

import pandas as pd
from trade_lifecycle import serialize_signal_snapshot

from entry_snapshot_enrichment import enrich_frame_with_entry_snapshots
from historical_dataset_builder import HistoricalDatasetBuilder


def test_historical_dataset_builder_backfills_portfolio_context_from_equity_curve():
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        pd.DataFrame(
            [
                {
                    "market": "BTC Test",
                    "timestamp": "2026-04-09T05:00:00Z",
                    "trader_wallet": "0xabc",
                    "confidence": 0.42,
                }
            ]
        ).to_csv(logs / "signals.csv", index=False)
        pd.DataFrame(
            [
                {
                    "timestamp": "2026-04-09T04:59:00Z",
                    "open_positions": 2,
                    "gross_market_value": 5.42,
                    "entry_notional": 5.70,
                    "unrealized_pnl": -0.28,
                }
            ]
        ).to_csv(logs / "portfolio_equity_curve.csv", index=False)

        df = HistoricalDatasetBuilder(logs_dir=logs).build()

        assert not df.empty
        assert int(df.iloc[0]["open_positions_count"]) == 2
        assert round(float(df.iloc[0]["open_positions_negotiated_value_total"]), 2) == 5.70
        assert round(float(df.iloc[0]["open_positions_current_value_total"]), 2) == 5.42
        assert round(float(df.iloc[0]["open_positions_unrealized_pnl_total"]), 2) == -0.28


def test_historical_dataset_builder_merges_kalman_and_regime_context():
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        pd.DataFrame(
            [
                {
                    "market": "BTC Test",
                    "timestamp": "2026-04-09T05:00:00Z",
                    "trader_wallet": "0xabc",
                    "confidence": 0.42,
                }
            ]
        ).to_csv(logs / "signals.csv", index=False)
        pd.DataFrame(
            [
                {
                    "btc_live_timestamp": "2026-04-09T04:59:30Z",
                    "btc_live_mark_price_kalman": 68201.4,
                    "btc_live_return_15m_kalman": 0.0062,
                    "btc_live_confluence_kalman": 0.67,
                }
            ]
        ).to_csv(logs / "btc_live_snapshot.csv", index=False)
        pd.DataFrame(
            [
                {
                    "technical_timestamp": "2026-04-09T04:59:45Z",
                    "btc_market_regime_label": "trend",
                    "btc_market_regime_score": 0.74,
                    "btc_market_regime_trend_score": 0.81,
                    "btc_market_regime_confidence_multiplier": 1.08,
                }
            ]
        ).to_csv(logs / "technical_regime_snapshot.csv", index=False)

        df = HistoricalDatasetBuilder(logs_dir=logs).build()

        assert not df.empty
        assert round(float(df.iloc[0]["btc_live_mark_price_kalman"]), 1) == 68201.4
        assert round(float(df.iloc[0]["btc_live_return_15m_kalman"]), 4) == 0.0062
        assert df.iloc[0]["btc_market_regime_label"] == "trend"
        assert round(float(df.iloc[0]["btc_market_regime_confidence_multiplier"]), 2) == 1.08


def test_historical_dataset_builder_coalesces_suffix_columns_from_live_and_regime_merges():
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        pd.DataFrame(
            [
                {
                    "market": "BTC Test",
                    "timestamp": "2026-04-09T05:00:00Z",
                    "trader_wallet": "0xabc",
                    "confidence": 0.42,
                    "btc_live_index_price": None,
                    "btc_market_regime_score": None,
                }
            ]
        ).to_csv(logs / "signals.csv", index=False)
        pd.DataFrame(
            [
                {
                    "btc_live_timestamp": "2026-04-09T04:59:30Z",
                    "btc_live_index_price": 68195.2,
                    "btc_live_source_quality_score": 0.87,
                }
            ]
        ).to_csv(logs / "btc_live_snapshot.csv", index=False)
        pd.DataFrame(
            [
                {
                    "technical_timestamp": "2026-04-09T04:59:45Z",
                    "btc_market_regime_score": 0.74,
                }
            ]
        ).to_csv(logs / "technical_regime_snapshot.csv", index=False)

        df = HistoricalDatasetBuilder(logs_dir=logs).build()

        assert not df.empty
        assert "btc_live_index_price_x" not in df.columns
        assert "btc_live_index_price_y" not in df.columns
        assert "btc_market_regime_score_x" not in df.columns
        assert "btc_market_regime_score_y" not in df.columns
        assert round(float(df.iloc[0]["btc_live_index_price"]), 1) == 68195.2
        assert round(float(df.iloc[0]["btc_market_regime_score"]), 2) == 0.74


def test_historical_dataset_builder_applies_numeric_feature_priors():
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        pd.DataFrame(
            [
                {
                    "market": "BTC Test",
                    "timestamp": "2026-04-09T05:00:00Z",
                    "trader_wallet": "0xabc",
                    "confidence": 0.42,
                    "current_price": 0.57,
                    "btc_price": 68123.4,
                    "wallet_trade_count_30d": None,
                    "wallet_alpha_30d": None,
                    "wallet_signal_precision_tp": None,
                    "btc_live_index_price": None,
                    "btc_market_regime_score": None,
                    "open_positions_count": None,
                    "spread": None,
                }
            ]
        ).to_csv(logs / "signals.csv", index=False)

        df = HistoricalDatasetBuilder(logs_dir=logs).build()

        assert not df.empty
        row = df.iloc[0]
        assert float(row["wallet_trade_count_30d"]) == 0.0
        assert float(row["wallet_alpha_30d"]) == 0.0
        assert float(row["wallet_signal_precision_tp"]) == 0.5
        assert round(float(row["btc_live_index_price"]), 1) == 68123.4
        assert float(row["btc_market_regime_score"]) == 0.5
        assert float(row["open_positions_count"]) == 0.0
        assert float(row["spread"]) == 0.0


def test_historical_dataset_builder_applies_weather_feature_priors():
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        pd.DataFrame(
            [
                {
                    "market": "Will the highest temperature in NYC be 64F or higher?",
                    "timestamp": "2026-04-09T05:00:00Z",
                    "trader_wallet": "0xabc",
                    "market_family": "weather_temperature_threshold",
                    "current_price": None,
                    "wallet_temp_hit_rate_90d": None,
                    "wallet_quality_score": None,
                    "source_wallet_size_delta_ratio": None,
                    "wallet_watchlist_approved": None,
                    "wallet_state_gate_pass": None,
                    "weather_parseable": None,
                    "forecast_ready": None,
                    "forecast_stale": None,
                    "weather_forecast_confirms_direction": None,
                    "weather_threshold_conflict": None,
                    "forecast_p_hit_interval": None,
                    "weather_fair_probability_side": None,
                    "weather_market_probability": None,
                    "weather_forecast_edge": None,
                    "weather_forecast_margin_score": None,
                    "weather_forecast_stability_score": None,
                    "spread": None,
                }
            ]
        ).to_csv(logs / "signals.csv", index=False)

        df = HistoricalDatasetBuilder(logs_dir=logs).build()

        assert not df.empty
        row = df.iloc[0]
        assert float(row["current_price"]) == 0.5
        assert float(row["wallet_temp_hit_rate_90d"]) == 0.5
        assert float(row["wallet_quality_score"]) == 0.5
        assert float(row["source_wallet_size_delta_ratio"]) == 0.0
        assert float(row["wallet_watchlist_approved"]) == 0.0
        assert float(row["wallet_state_gate_pass"]) == 0.0
        assert float(row["weather_parseable"]) == 0.0
        assert float(row["forecast_ready"]) == 0.0
        assert float(row["forecast_stale"]) == 0.0
        assert float(row["weather_forecast_confirms_direction"]) == 0.0
        assert float(row["weather_threshold_conflict"]) == 0.0
        assert float(row["forecast_p_hit_interval"]) == 0.5
        assert float(row["weather_fair_probability_side"]) == 0.5
        assert float(row["weather_market_probability"]) == 0.5
        assert float(row["weather_forecast_edge"]) == 0.0
        assert float(row["weather_forecast_margin_score"]) == 0.5
        assert float(row["weather_forecast_stability_score"]) == 0.5
        assert float(row["spread"]) == 0.0


def test_historical_dataset_builder_backfills_missing_signal_columns_from_entry_snapshots():
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        pd.DataFrame(
            [
                {
                    "market": "BTC Test",
                    "timestamp": "2026-04-09T05:00:00Z",
                    "trader_wallet": "0xabc",
                    "token_id": "tok-1",
                    "condition_id": "cond-1",
                    "outcome_side": "YES",
                }
            ]
        ).to_csv(logs / "signals.csv", index=False)
        snapshot_json, feature_count = serialize_signal_snapshot(
            {
                "timestamp": "2026-04-09T05:00:00Z",
                "market": "BTC Test",
                "trader_wallet": "0xabc",
                "token_id": "tok-1",
                "condition_id": "cond-1",
                "outcome_side": "YES",
                "wallet_quality_score": 0.81,
                "btc_live_mark_price_kalman": 68222.4,
            }
        )
        pd.DataFrame(
            [
                {
                    "position_id": "tok-1|cond-1|YES",
                    "market": "BTC Test",
                    "market_title": "BTC Test",
                    "token_id": "tok-1",
                    "condition_id": "cond-1",
                    "outcome_side": "YES",
                    "opened_at": "2026-04-09T05:00:00Z",
                    "status": "OPEN",
                    "entry_signal_snapshot_json": snapshot_json,
                    "entry_signal_snapshot_feature_count": feature_count,
                    "entry_signal_snapshot_version": 1,
                }
            ]
        ).to_csv(logs / "positions.csv", index=False)

        df = HistoricalDatasetBuilder(logs_dir=logs).build()

        assert not df.empty
        assert round(float(df.iloc[0]["wallet_quality_score"]), 2) == 0.81
        assert round(float(df.iloc[0]["btc_live_mark_price_kalman"]), 1) == 68222.4
        assert bool(df.iloc[0]["entry_snapshot_backfilled"]) is True


def test_historical_dataset_builder_merges_local_weather_wallet_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        weather_logs = logs / "weather_temperature"
        weather_logs.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "market": "Will the highest temperature in NYC be 64F or higher?",
                    "market_title": "Will the highest temperature in NYC be 64F or higher?",
                    "timestamp": "2026-04-09T05:00:00Z",
                    "trader_wallet": "0xabc",
                    "token_id": "tok-1",
                    "condition_id": "cond-1",
                    "outcome_side": "YES",
                    "market_family": "weather_temperature_threshold",
                }
            ]
        ).to_csv(logs / "signals.csv", index=False)
        pd.DataFrame(
            [
                {
                    "timestamp": "2026-04-09T05:05:00Z",
                    "market_title": "Will the highest temperature in NYC be 64F or higher?",
                    "trader_wallet": "0xabc",
                    "token_id": "tok-1",
                    "condition_id": "cond-1",
                    "outcome_side": "YES",
                    "market_family": "weather_temperature_threshold",
                    "wallet_temp_hit_rate_90d": 0.82,
                    "wallet_quality_score": 0.76,
                    "source_wallet_size_delta_ratio": 0.31,
                    "wallet_watchlist_approved": True,
                    "weather_parseable": True,
                }
            ]
        ).to_csv(weather_logs / "weather_wallet_state_snapshot.csv", index=False)

        df = HistoricalDatasetBuilder(
            logs_dir=logs,
            market_family="weather_temperature",
            shared_logs_dir=logs,
        ).build()

        assert not df.empty
        row = df.iloc[0]
        assert round(float(row["wallet_temp_hit_rate_90d"]), 2) == 0.82
        assert round(float(row["wallet_quality_score"]), 2) == 0.76
        assert round(float(row["source_wallet_size_delta_ratio"]), 2) == 0.31
        assert float(row["wallet_watchlist_approved"]) == 1.0
        assert float(row["weather_parseable"]) == 1.0


def test_entry_snapshot_enrichment_avoids_futurewarning_on_bool_fill_and_append():
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        base_df = pd.DataFrame(
            [
                {
                    "market": "BTC Test",
                    "market_title": "BTC Test",
                    "timestamp": "2026-04-09T05:00:00Z",
                    "trader_wallet": "0xabc",
                    "token_id": "tok-1",
                    "condition_id": "cond-1",
                    "outcome_side": "YES",
                    "entry_snapshot_backfilled": None,
                }
            ]
        )
        snapshot_json, feature_count = serialize_signal_snapshot(
            {
                "timestamp": "2026-04-09T05:05:00Z",
                "market": "BTC Test 2",
                "trader_wallet": "0xdef",
                "token_id": "tok-2",
                "condition_id": "cond-2",
                "outcome_side": "NO",
                "wallet_quality_score": 0.55,
            }
        )
        pd.DataFrame(
            [
                {
                    "position_id": "tok-2|cond-2|NO",
                    "market": "BTC Test 2",
                    "market_title": "BTC Test 2",
                    "token_id": "tok-2",
                    "condition_id": "cond-2",
                    "outcome_side": "NO",
                    "opened_at": "2026-04-09T05:05:00Z",
                    "status": "OPEN",
                    "entry_signal_snapshot_json": snapshot_json,
                    "entry_signal_snapshot_feature_count": feature_count,
                    "entry_signal_snapshot_version": 1,
                }
            ]
        ).to_csv(logs / "positions.csv", index=False)

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            enriched = enrich_frame_with_entry_snapshots(base_df, logs_dir=logs)

        assert len(enriched.index) == 2
        assert enriched["entry_snapshot_backfilled"].dtype == bool
