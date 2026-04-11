from pathlib import Path

import pandas as pd

from brain_data_migration import migrate_legacy_mixed_training_data


def test_migrate_legacy_mixed_training_data_splits_family_outputs(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    weights_dir = tmp_path / "weights"
    logs_dir.mkdir()
    weights_dir.mkdir()

    mixed_frame = pd.DataFrame(
        [
            {"market_family": "btc", "token_id": "btc-1", "value": 1},
            {"market_family": "weather_temperature", "token_id": "w-1", "value": 2},
        ]
    )
    mixed_frame.to_csv(logs_dir / "historical_dataset.csv", index=False)
    mixed_frame.to_csv(logs_dir / "contract_targets.csv", index=False)
    mixed_frame.to_csv(logs_dir / "sequence_dataset.csv", index=False)
    mixed_frame.to_csv(logs_dir / "baseline_eval.csv", index=False)
    mixed_frame.to_csv(logs_dir / "model_registry_comparison.csv", index=False)
    mixed_frame.to_csv(logs_dir / "regime_model_comparison.csv", index=False)

    (weights_dir / "tp_classifier.joblib").write_text("btc", encoding="utf-8")
    (weights_dir / "weather_temperature_model.joblib").write_text("weather", encoding="utf-8")

    summary = migrate_legacy_mixed_training_data(shared_logs_dir=logs_dir, shared_weights_dir=weights_dir)

    archive_dir = Path(summary["archive_dir"])
    assert archive_dir.exists()
    assert (archive_dir / "historical_dataset.csv").exists()

    btc_df = pd.read_csv(logs_dir / "btc" / "historical_dataset.csv")
    weather_df = pd.read_csv(logs_dir / "weather_temperature" / "historical_dataset.csv")

    assert len(btc_df.index) == 1
    assert btc_df.iloc[0]["market_family"] == "btc"
    assert btc_df.iloc[0]["brain_id"] == "btc_brain"

    assert len(weather_df.index) == 1
    assert weather_df.iloc[0]["market_family"] == "weather_temperature"
    assert weather_df.iloc[0]["brain_id"] == "weather_temperature_brain"

    assert (weights_dir / "btc" / "tp_classifier.joblib").exists()
    assert (weights_dir / "weather_temperature" / "weather_temperature_model.joblib").exists()
