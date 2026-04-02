from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


INFERENCE_MODEL_FILENAMES = (
    "tp_classifier.joblib",
    "return_regressor.joblib",
    "stage1_tp_classifier.joblib",
    "stage1_return_regressor.joblib",
    "stage2_temporal_classifier.joblib",
    "stage2_temporal_regressor.joblib",
)

LEGACY_RL_FILENAMES = (
    "ppo_polytrader.zip",
    "ppo_polytrader_vecnormalize.pkl",
)

PROMOTABLE_MODEL_FILENAMES = INFERENCE_MODEL_FILENAMES + LEGACY_RL_FILENAMES


def _timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def build_candidate_weights_dir(weights_dir: str | Path, prefix: str = "candidate") -> Path:
    root = Path(weights_dir)
    candidate_dir = root / "_candidates" / f"{prefix}_{_timestamp_token()}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    return candidate_dir


def snapshot_artifact_state(
    weights_dir: str | Path,
    filenames: tuple[str, ...] = PROMOTABLE_MODEL_FILENAMES,
) -> dict:
    root = Path(weights_dir)
    state = {}
    for name in filenames:
        path = root / name
        if not path.exists():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        state[name] = {
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }
    return state


def promote_candidate_artifacts(
    candidate_dir: str | Path,
    active_weights_dir: str | Path,
    filenames: tuple[str, ...] = PROMOTABLE_MODEL_FILENAMES,
    backup_label: str = "promotion",
) -> dict:
    candidate_root = Path(candidate_dir)
    active_root = Path(active_weights_dir)
    active_root.mkdir(parents=True, exist_ok=True)

    present = [name for name in filenames if (candidate_root / name).exists()]
    if not present:
        return {"promoted_files": [], "rollback_dir": None}

    rollback_dir = active_root / "_rollback" / f"{backup_label}_{_timestamp_token()}"
    rollback_dir.mkdir(parents=True, exist_ok=True)

    for name in present:
        active_path = active_root / name
        if active_path.exists():
            shutil.copy2(active_path, rollback_dir / name)

    for name in present:
        staged_path = candidate_root / name
        active_path = active_root / name
        os.replace(str(staged_path), str(active_path))

    return {
        "promoted_files": present,
        "rollback_dir": str(rollback_dir),
    }
