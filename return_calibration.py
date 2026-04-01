from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except Exception:
        return float(default)


def _finite_series(values) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)


def default_return_calibration() -> dict:
    abs_clip = max(0.05, _env_float("EXPECTED_RETURN_ABS_CLIP", 0.60))
    scale = max(abs_clip * 1.10, abs_clip + 0.01)
    return {
        "version": 1,
        "transform": "identity",
        "clip_lower": -abs_clip,
        "clip_upper": abs_clip,
        "scale": scale,
        "uncertainty_floor": max(0.01, abs_clip * 0.08),
        "train_rows": 0,
        "train_median": 0.0,
        "reliability": 1.0,
    }


def fit_return_calibration(values) -> dict:
    series = _finite_series(values)
    fallback = default_return_calibration()
    if series.empty:
        return fallback

    abs_clip_ceiling = max(0.10, _env_float("EXPECTED_RETURN_ABS_CLIP", 0.60))
    min_abs_bound = min(0.10, abs_clip_ceiling)
    tail_headroom = max(1.0, _env_float("EXPECTED_RETURN_TAIL_HEADROOM", 1.10))
    full_confidence_rows = max(25.0, _env_float("EXPECTED_RETURN_FULL_CONFIDENCE_ROWS", 200.0))
    min_reliability = min(1.0, max(0.05, _env_float("EXPECTED_RETURN_MIN_RELIABILITY", 0.20)))

    q01 = float(series.quantile(0.01))
    q99 = float(series.quantile(0.99))
    observed_min = float(series.min())
    observed_max = float(series.max())
    observed_median = float(series.quantile(0.50))
    train_rows = int(len(series))
    reliability = float(np.clip(train_rows / full_confidence_rows, min_reliability, 1.0))

    clip_lower = min(observed_min * tail_headroom, q01 * tail_headroom, -min_abs_bound)
    clip_upper = max(observed_max * tail_headroom, q99 * tail_headroom, min_abs_bound)
    clip_lower = max(clip_lower, -abs_clip_ceiling)
    clip_upper = min(clip_upper, abs_clip_ceiling)

    if clip_lower >= -1e-9:
        clip_lower = -min_abs_bound
    if clip_upper <= 1e-9:
        clip_upper = min_abs_bound

    scale = max(abs(clip_lower), abs(clip_upper), min_abs_bound) * 1.10
    scale = max(scale, min_abs_bound + 0.01)

    return {
        "version": 1,
        "transform": "signed_tanh",
        "clip_lower": float(clip_lower),
        "clip_upper": float(clip_upper),
        "scale": float(scale),
        "uncertainty_floor": float(max(0.01, min(abs_clip_ceiling * 0.20, series.std(ddof=0) * 0.35 if len(series) > 1 else 0.02))),
        "train_rows": train_rows,
        "train_median": observed_median,
        "reliability": reliability,
        "train_min": observed_min,
        "train_max": observed_max,
        "train_q01": q01,
        "train_q99": q99,
    }


def transform_return_targets(values, calibration: dict | None):
    series = _finite_series(values)
    if series.empty:
        return series
    calibration = calibration or default_return_calibration()
    clip_lower = float(calibration.get("clip_lower", default_return_calibration()["clip_lower"]))
    clip_upper = float(calibration.get("clip_upper", default_return_calibration()["clip_upper"]))
    scale = max(0.01, float(calibration.get("scale", default_return_calibration()["scale"])))

    clipped = series.clip(lower=clip_lower, upper=clip_upper)
    if calibration.get("transform") == "signed_tanh":
        normalized = np.clip(clipped / scale, -0.999999, 0.999999)
        return pd.Series(np.arctanh(normalized), index=clipped.index, dtype=float)
    return clipped.astype(float)


def calibrate_return_predictions(predictions, calibration: dict | None = None, index=None) -> pd.Series:
    fallback = default_return_calibration()
    calibration = calibration or fallback
    clip_lower = float(calibration.get("clip_lower", fallback["clip_lower"]))
    clip_upper = float(calibration.get("clip_upper", fallback["clip_upper"]))
    scale = max(0.01, float(calibration.get("scale", fallback["scale"])))
    train_median = float(calibration.get("train_median", fallback["train_median"]))
    reliability = float(calibration.get("reliability", fallback["reliability"]))
    reliability = float(np.clip(reliability, 0.0, 1.0))

    series = predictions if isinstance(predictions, pd.Series) else pd.Series(np.asarray(predictions).ravel(), index=index)
    if index is not None and len(series.index) != len(index):
        series = pd.Series(np.asarray(predictions).ravel(), index=index)
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)

    if calibration.get("transform") == "signed_tanh":
        series = pd.Series(np.tanh(np.clip(series, -10.0, 10.0)) * scale, index=series.index, dtype=float)

    # Small datasets can produce extreme but unstable regressors.
    # Revert partially toward the training-median ROI until sample size grows.
    series = train_median + ((series - train_median) * reliability)
    return series.clip(lower=clip_lower, upper=clip_upper)


def clip_expected_return_series(values, calibration: dict | None = None) -> pd.Series:
    fallback = default_return_calibration()
    calibration = calibration or fallback
    clip_lower = float(calibration.get("clip_lower", fallback["clip_lower"]))
    clip_upper = float(calibration.get("clip_upper", fallback["clip_upper"]))
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
    return series.clip(lower=clip_lower, upper=clip_upper)
