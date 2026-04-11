from pathlib import Path

import pandas as pd


def _safe_merge_asof(left, right, on, **kwargs):
    """merge_asof that tolerates NaT/null in the left merge key."""
    if left.empty or right.empty or on not in left.columns or on not in right.columns:
        return left
    mask = left[on].notna()
    if not mask.any():
        return left
    valid = left[mask].copy().sort_values(on)
    work_right = right.copy().sort_values(on)
    merged = pd.merge_asof(valid, work_right, on=on, **kwargs)
    if mask.all():
        return merged
    return pd.concat([merged, left[~mask]], ignore_index=True)


class DatasetAligner:
    """
    Align historical project snapshots with BTC future-return targets.
    """

    def __init__(self, logs_dir="logs", *, shared_logs_dir=None):
        self.logs_dir = Path(logs_dir)
        self.shared_logs_dir = Path(shared_logs_dir or logs_dir)
        self.dataset_file = self.logs_dir / "historical_dataset.csv"
        self.targets_file = self.shared_logs_dir / "btc_targets.csv"
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

        aligned = _safe_merge_asof(dataset, targets, on="timestamp", direction="backward")
        aligned = aligned.dropna(subset=["target_up", "future_return"]).reset_index(drop=True)
        return aligned

    def write(self):
        aligned = self.build()
        if not aligned.empty:
            aligned.to_csv(self.output_file, index=False)
        return aligned

