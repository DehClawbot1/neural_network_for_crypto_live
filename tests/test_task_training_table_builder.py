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
