import tempfile
from pathlib import Path

import pandas as pd

from task_training_table_builder import TaskTrainingTableBuilder


def test_task_training_table_builder_writes_filtered_tables():
    with tempfile.TemporaryDirectory() as tmp:
        logs_dir = Path(tmp)
        btc_dir = logs_dir / "btc"
        btc_dir.mkdir(parents=True, exist_ok=True)

        canonical = pd.DataFrame(
            [
                {
                    "token_id": "tok1",
                    "market_family": "btc_directional_intraday",
                    "learning_eligible": True,
                    "entry_quality": 1,
                    "entry_edge_realized": 0.12,
                    "trade_outcome_label": "win",
                    "execution_quality": 1,
                    "slippage_error": -0.001,
                    "exit_quality": 1,
                    "exit_regret": 0.0,
                    "technical_regime_bucket": "trend",
                    "actual_execution_path": "live_exit_filled",
                },
                {
                    "token_id": "tok2",
                    "market_family": "btc_other",
                    "learning_eligible": False,
                    "entry_quality": 0,
                    "entry_edge_realized": -0.05,
                    "trade_outcome_label": "loss",
                    "execution_quality": 0,
                    "slippage_error": 0.01,
                    "exit_quality": 0,
                    "exit_regret": 0.03,
                    "technical_regime_bucket": "chaotic",
                    "actual_execution_path": "external_manual_close",
                },
            ]
        )
        canonical.to_csv(btc_dir / "canonical_dataset.csv", index=False)

        TaskTrainingTableBuilder(logs_dir=logs_dir).write()

        entry_df = pd.read_csv(btc_dir / "task_training" / "entry_edge.csv")
        fill_df = pd.read_csv(btc_dir / "task_training" / "fill_probability.csv")

        assert len(entry_df) == 1
        assert entry_df.iloc[0]["token_id"] == "tok1"
        assert len(fill_df) == 2
        assert "fill_success" in fill_df.columns


def test_task_training_table_builder_bootstraps_weather_from_resolved_contracts():
    with tempfile.TemporaryDirectory() as tmp:
        logs_dir = Path(tmp)
        weather_dir = logs_dir / "weather_temperature"
        weather_dir.mkdir(parents=True, exist_ok=True)

        contract_targets = pd.DataFrame(
            [
                {
                    "timestamp": "2026-04-11T12:00:00Z",
                    "market_family": "weather_temperature",
                    "market_title": "Will the highest temperature in Lisbon be 20C on April 12?",
                    "token_id": "w1",
                    "condition_id": "c1",
                    "outcome_side": "YES",
                    "target_up": 1,
                    "weather_market_probability": 0.40,
                    "forecast_uncertainty_c": 1.0,
                    "weather_forecast_edge": 0.20,
                },
                {
                    "timestamp": "2026-04-12T12:00:00Z",
                    "market_family": "weather_temperature",
                    "market_title": "Will the highest temperature in Porto be 18C on April 13?",
                    "token_id": "w2",
                    "condition_id": "c2",
                    "outcome_side": "YES",
                    "target_up": 0,
                    "weather_market_probability": 0.70,
                    "forecast_uncertainty_c": 2.0,
                    "weather_forecast_edge": -0.10,
                },
                {
                    "timestamp": "2026-04-13T12:00:00Z",
                    "market_family": "weather_temperature",
                    "market_title": "Will the highest temperature in Madrid be 24C on April 14?",
                    "token_id": "w3",
                    "condition_id": "c3",
                    "outcome_side": "YES",
                    "target_up": 1,
                    "weather_market_probability": 0.55,
                    "forecast_uncertainty_c": 3.0,
                    "weather_forecast_edge": 0.08,
                },
            ]
        )
        contract_targets.to_csv(weather_dir / "contract_targets.csv", index=False)

        historical = pd.DataFrame(
            [
                {"timestamp": "2026-04-11T11:00:00Z", "token_id": "w1", "condition_id": "c1", "outcome_side": "YES", "weather_location": "Lisbon", "weather_question_type": "temp", "liquidity_score": 10, "volume_score": 100, "open_positions_count": 1},
                {"timestamp": "2026-04-12T11:00:00Z", "token_id": "w2", "condition_id": "c2", "outcome_side": "YES", "weather_location": "Porto", "weather_question_type": "temp", "liquidity_score": 20, "volume_score": 200, "open_positions_count": 2},
                {"timestamp": "2026-04-13T11:00:00Z", "token_id": "w3", "condition_id": "c3", "outcome_side": "YES", "weather_location": "Madrid", "weather_question_type": "temp", "liquidity_score": 30, "volume_score": 300, "open_positions_count": 3},
            ]
        )
        historical.to_csv(weather_dir / "historical_dataset.csv", index=False)

        decisions = pd.DataFrame(
            [
                {"created_at": "2026-04-11T11:30:00Z", "token_id": "w1", "condition_id": "c1", "outcome_side": "YES", "confidence": 0.65, "p_tp_before_sl": 0.8, "expected_return": 0.12, "final_decision": "ENTRY", "market": "Lisbon 20C"},
                {"created_at": "2026-04-12T11:30:00Z", "token_id": "w2", "condition_id": "c2", "outcome_side": "YES", "confidence": 0.35, "p_tp_before_sl": 0.4, "expected_return": -0.05, "final_decision": "REJECTED", "market": "Porto 18C"},
                {"created_at": "2026-04-13T11:30:00Z", "token_id": "w3", "condition_id": "c3", "outcome_side": "YES", "confidence": 0.72, "p_tp_before_sl": 0.9, "expected_return": 0.09, "final_decision": "ENTRY", "market": "Madrid 24C"},
            ]
        )
        decisions.to_csv(weather_dir / "candidate_decisions.csv", index=False)

        TaskTrainingTableBuilder(logs_dir=logs_dir).write()

        canonical_df = pd.read_csv(weather_dir / "canonical_dataset.csv")
        entry_df = pd.read_csv(weather_dir / "task_training" / "entry_edge.csv")
        exit_df = pd.read_csv(weather_dir / "task_training" / "exit_quality.csv")
        regime_df = pd.read_csv(weather_dir / "task_training" / "regime_calibration.csv")

        assert len(canonical_df) == 3
        assert canonical_df["weather_contract_resolved_yes"].tolist() == [1, 0, 1]
        assert "technical_regime_bucket" in canonical_df.columns
        assert len(entry_df) == 3
        assert len(exit_df) == 3
        assert len(regime_df) == 3
