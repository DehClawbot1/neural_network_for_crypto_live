from pathlib import Path

import pandas as pd

from supervised_trainer import SupervisedTrainer


def test_supervised_trainer_coerces_schema_drift_strings(tmp_path):
    logs_dir = tmp_path / "logs"
    weights_dir = tmp_path / "weights"
    logs_dir.mkdir()
    weights_dir.mkdir()

    df = pd.DataFrame(
        [
            {
                "liquidity_score": "Neutral",
                "probability_momentum": "False",
                "volume_score": 0.7,
                "volatility_score": 0.2,
                "target_up": 0,
            },
            {
                "liquidity_score": 0.4,
                "probability_momentum": 0.3,
                "volume_score": 0.5,
                "volatility_score": 0.4,
                "target_up": 1,
            },
            {
                "liquidity_score": 0.6,
                "probability_momentum": 0.9,
                "volume_score": 0.8,
                "volatility_score": 0.1,
                "target_up": 1,
            },
        ]
    )
    df.to_csv(logs_dir / "aligned_dataset.csv", index=False)

    model, features = SupervisedTrainer(logs_dir=logs_dir, weights_dir=weights_dir).train()

    assert model is not None
    assert features is not None
    assert "volume_score" in features
    assert (weights_dir / "btc_direction_model.joblib").exists()
