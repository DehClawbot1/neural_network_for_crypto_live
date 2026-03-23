from pathlib import Path

import pandas as pd


class DatasetAligner:
    """
    Align historical project snapshots with BTC future-return targets.
    """

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.dataset_file = self.logs_dir / "historical_dataset.csv"
        self.targets_file = self.logs_dir / "btc_targets.csv"
        self.output_file = self.logs_dir / "aligned_dataset.csv"

    def _safe_read(self, path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def build(self):
        dataset = self._safe_read(self.dataset_file)
        targets = self._safe_read(self.targets_file)

        if dataset.empty or targets.empty or "timestamp" not in dataset.columns:
            return pd.DataFrame()

        dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True, errors="coerce")
        targets["timestamp"] = pd.to_datetime(targets["timestamp"], utc=True, errors="coerce")
        dataset = dataset.dropna(subset=["timestamp"]).sort_values("timestamp")
        targets = targets.dropna(subset=["timestamp"]).sort_values("timestamp")

        aligned = pd.merge_asof(dataset, targets, on="timestamp", direction="backward")
        aligned = aligned.dropna(subset=["target_up", "future_return"]).reset_index(drop=True)
        return aligned

    def write(self):
        aligned = self.build()
        if not aligned.empty:
            aligned.to_csv(self.output_file, index=False)
        return aligned

