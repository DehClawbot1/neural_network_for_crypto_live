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
        self.retrain_threshold = retrain_threshold

    def maybe_retrain(self):
        if not self.dataset_file.exists():
            return False

        try:
            dataset = pd.read_csv(self.dataset_file)
        except Exception:
            return False

        if len(dataset) < self.retrain_threshold:
            self.status_file.write_text(
                f"Not enough rows for retraining yet: {len(dataset)} / {self.retrain_threshold}\n",
                encoding="utf-8",
            )
            return False

        logging.info("Historical dataset reached retraining threshold. Starting paper-mode retraining...")
        train_model(timesteps=5000)
        self.status_file.write_text(
            f"Retraining triggered successfully with {len(dataset)} rows.\n",
            encoding="utf-8",
        )
        return True
