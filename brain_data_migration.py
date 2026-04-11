from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from brain_paths import filter_frame_for_brain, list_brain_contexts


LEGACY_TRAINING_FILES = [
    "historical_dataset.csv",
    "contract_targets.csv",
    "sequence_dataset.csv",
    "baseline_eval.csv",
    "model_registry_comparison.csv",
    "regime_model_comparison.csv",
]

BTC_WEIGHT_FILES = [
    "tp_classifier.joblib",
    "return_regressor.joblib",
    "stage1_tp_classifier.joblib",
    "stage1_return_regressor.joblib",
    "stage2_temporal_classifier.joblib",
    "stage2_temporal_regressor.joblib",
]

WEATHER_WEIGHT_FILES = [
    "weather_temperature_model.joblib",
]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _archive_move(src: Path, archive_dir: Path) -> Path | None:
    if not src.exists():
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / src.name
    shutil.move(str(src), str(dest))
    return dest


def migrate_legacy_mixed_training_data(
    *,
    shared_logs_dir: str | Path = "logs",
    shared_weights_dir: str | Path = "weights",
) -> dict:
    shared_logs_path = Path(shared_logs_dir)
    shared_weights_path = Path(shared_weights_dir)
    contexts = list_brain_contexts(shared_logs_dir=shared_logs_path, shared_weights_dir=shared_weights_path)
    btc_context = next(context for context in contexts if context.market_family == "btc")
    weather_context = next(context for context in contexts if context.market_family == "weather_temperature")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_dir = shared_logs_path / "legacy_mixed" / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "archive_dir": str(archive_dir),
        "archived_files": [],
        "brain_files": {},
        "seeded_weights": [],
    }

    for filename in LEGACY_TRAINING_FILES:
        src = shared_logs_path / filename
        if not src.exists():
            continue
        frame = _safe_read_csv(src)
        if not frame.empty:
            for context in contexts:
                filtered = filter_frame_for_brain(frame, context)
                out_path = context.logs_dir / filename
                filtered.to_csv(out_path, index=False)
                summary["brain_files"][str(out_path)] = int(len(filtered.index))
        archived = _archive_move(src, archive_dir)
        if archived is not None:
            summary["archived_files"].append(str(archived))

    for filename in BTC_WEIGHT_FILES:
        src = shared_weights_path / filename
        if not src.exists():
            continue
        dest = btc_context.weights_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        summary["seeded_weights"].append(str(dest))

    for filename in WEATHER_WEIGHT_FILES:
        src = shared_weights_path / filename
        if not src.exists():
            continue
        dest = weather_context.weights_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        summary["seeded_weights"].append(str(dest))

    return summary
