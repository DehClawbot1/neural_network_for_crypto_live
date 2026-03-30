from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd

_BASE_DIR = Path(__file__).resolve().parent


def _load_legacy_module(module_name: str, filename: str):
    path = _BASE_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy_stage1 = _load_legacy_module("legacy_stage1_models", "stage1_models.py")
_legacy_stage2 = _load_legacy_module("legacy_stage2_temporal_models", "stage2_temporal_models.py")


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except Exception:
        return int(default)


class WindowedStage1Models(_legacy_stage1.Stage1Models):
    def __init__(self, *args, max_rows: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_rows = int(max_rows or _env_int("STAGE1_MAX_ROWS", 50000))

    def _safe_read(self):
        df = super()._safe_read()
        if df.empty:
            return df
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.sort_values("timestamp")
        return df.tail(self.max_rows).copy()


class WindowedStage2TemporalModels(_legacy_stage2.Stage2TemporalModels):
    def __init__(self, *args, max_rows: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_rows = int(max_rows or _env_int("STAGE2_MAX_ROWS", 20000))

    def _safe_read(self):
        df = super()._safe_read()
        if df.empty:
            return df
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.sort_values("timestamp")
        return df.tail(self.max_rows).copy()
