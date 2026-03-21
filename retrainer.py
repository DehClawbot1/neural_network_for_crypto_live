import logging
from pathlib import Path

import pandas as pd

from rl_trainer import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Retrainer:
    """
    Paper-mode retraining trigger based on accumulated historical dataset size.
    Research/paper-trading only.
    """

    def __init__(self, logs_dir="logs", retrain_threshold=200):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_file = self.logs_dir / "historical_dataset.csv"
        self.status_file = self.logs_dir / "retrainer_status.txt"
        self.status_csv = self.logs_dir / "model_status.csv"
        self.retrain_threshold = retrain_threshold

    def _write_status(self, dataset_rows: int, action: str):
        self.status_file.write_text(action + "\n", encoding="utf-8")
        pd.DataFrame(
            [
                {
                    "dataset_rows": dataset_rows,
                    "retrain_threshold": self.retrain_threshold,
                    "progress_ratio": round(dataset_rows / self.retrain_threshold, 4) if self.retrain_threshold else 0,
                    "last_action": action,
                }
            ]
        ).to_csv(self.status_csv, index=False)

    def maybe_retrain(self):
        if not self.dataset_file.exists():
            self._write_status(0, "No historical dataset yet.")
            return False

        try:
            dataset = pd.read_csv(self.dataset_file)
        except Exception:
            self._write_status(0, "Historical dataset unreadable.")
            return False

        if len(dataset) < self.retrain_threshold:
            self._write_status(len(dataset), f"Not enough rows for retraining yet: {len(dataset)} / {self.retrain_threshold}")
            return False

        logging.info("Historical dataset reached retraining threshold. Starting paper-mode retraining...")
        train_model(timesteps=5000)
        self._write_status(len(dataset), f"Retraining triggered successfully with {len(dataset)} rows.")
        return True
