"""
services/calibration_store.py
──────────────────────────────
Persistence for conformal calibration residuals.

One file per market family, appended whenever a trade closes. Stored as a
plain `.npy` float64 array so that misses/corruption are loud (np.load
raises), not silent.

Layout
──────
    logs/<family>/conformal_calibration.npy      # residuals = |y_true - y_pred|

Rules
─────
- `append_residual()` NEVER overwrites; it appends one scalar per call.
- `load_residuals()` returns an empty array on missing file (first run) but
  raises on corruption — callers decide whether to proceed without sizing.
- Residuals older than `max_residuals` are trimmed on append (FIFO window).
  Default window = 2000, large enough for stable 80% quantile, small enough
  to adapt to regime change within a few weeks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from services.types import DataFaultError

_DEFAULT_MAX_RESIDUALS = 2000


def _resolve_path(family: str, logs_dir: str | Path) -> Path:
    fam = str(family or "").strip().lower()
    if not fam:
        raise DataFaultError("calibration_store: family must be non-empty")
    root = Path(logs_dir) / fam
    root.mkdir(parents=True, exist_ok=True)
    return root / "conformal_calibration.npy"


def load_residuals(family: str, logs_dir: str | Path = "logs") -> np.ndarray:
    """
    Return the stored residual array for `family`. Empty array if none yet.
    Raises DataFaultError if the file exists but is unreadable/corrupt.
    """
    path = _resolve_path(family, logs_dir)
    if not path.exists():
        return np.array([], dtype=float)
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise DataFaultError(
            f"calibration_store: cannot load {path}: {exc}",
            context={"family": family, "path": str(path)},
        ) from exc
    if arr.ndim != 1 or arr.dtype.kind not in {"f", "i"}:
        raise DataFaultError(
            f"calibration_store: bad shape/dtype at {path} "
            f"(shape={arr.shape}, dtype={arr.dtype})"
        )
    return np.asarray(arr, dtype=float)


def append_residual(
    family: str,
    residual: float,
    *,
    logs_dir: str | Path = "logs",
    max_residuals: int = _DEFAULT_MAX_RESIDUALS,
) -> int:
    """
    Append a single absolute residual. Returns the new total length.
    FIFO-trims to `max_residuals`.
    """
    try:
        r = abs(float(residual))
    except (TypeError, ValueError) as exc:
        raise DataFaultError(
            f"calibration_store: residual={residual!r} not coercible to float"
        ) from exc
    if not np.isfinite(r):
        raise DataFaultError(f"calibration_store: residual={residual!r} is non-finite")
    path = _resolve_path(family, logs_dir)
    existing = load_residuals(family, logs_dir) if path.exists() else np.array([], dtype=float)
    updated = np.concatenate([existing, np.array([r], dtype=float)])
    if updated.size > max_residuals:
        updated = updated[-max_residuals:]
    # Atomic write: write to tmp then replace. np.save auto-appends `.npy` when
    # given a path-like, so write via an open file handle to control the name.
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, updated, allow_pickle=False)
    tmp.replace(path)
    return int(updated.size)


def residual_from_outcome(
    predicted_probability: float,
    outcome_binary: int,
) -> float:
    """
    Conformal nonconformity score for a binary classifier probability:

        residual = |outcome - p|

    where outcome ∈ {0, 1}. This is the standard Brier-style residual used
    by split-conformal on binary probability outputs.
    """
    if outcome_binary not in (0, 1):
        raise DataFaultError(f"outcome_binary must be 0 or 1, got {outcome_binary!r}")
    try:
        p = float(predicted_probability)
    except (TypeError, ValueError) as exc:
        raise DataFaultError(
            f"predicted_probability={predicted_probability!r} not numeric"
        ) from exc
    if not np.isfinite(p) or p < 0.0 or p > 1.0:
        raise DataFaultError(
            f"predicted_probability={p!r} outside [0, 1]"
        )
    return abs(float(outcome_binary) - p)


__all__ = [
    "load_residuals",
    "append_residual",
    "residual_from_outcome",
]
