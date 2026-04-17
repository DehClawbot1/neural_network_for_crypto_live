"""
services/uncertainty_service.py
───────────────────────────────
Split-conformal prediction for position sizing.

Given a held-out calibration set of (prediction, realised_outcome) pairs,
emit a coverage-guaranteed prediction interval for new predictions. Use the
interval width to scale position size: wider interval → smaller position.

References
----------
Vovk et al. 2005; Angelopoulos & Bates 2023 ("A Gentle Intro to Conformal
Prediction"). This is the vanilla split-conformal variant, which assumes
exchangeability of calibration and test samples — a reasonable approximation
when we refresh the calibration set on every purged walk-forward refit.

Usage
-----
    cal_residuals = np.abs(cal_y - cal_pred)   # absolute residuals
    service = ConformalIntervalService(alpha=0.2)
    service.fit_from_residuals(cal_residuals)
    lower, upper = service.interval(point_prediction=0.62)
    size_scale = service.size_multiplier(point_prediction=0.62)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from services.types import DataFaultError


@dataclass
class ConformalBand:
    alpha: float           # miscoverage rate (target: 1-alpha coverage)
    quantile: float        # empirical (1-alpha) quantile of residuals
    n_calibration: int
    residual_median: float

    @property
    def coverage_target(self) -> float:
        return 1.0 - self.alpha


class ConformalIntervalService:
    """
    Split-conformal interval estimator for scalar regression / probability outputs.

    Contract
    --------
    - `fit_from_residuals(residuals)` : residuals = |y_true - y_pred| on a
      held-out calibration set. Does NOT touch any training split.
    - `interval(point_prediction)` : returns (lower, upper) clamped to
      [0, 1] when `clamp_unit_interval=True` (default — suited for
      classifier probabilities).
    - `size_multiplier(...)` : deterministic mapping from interval width to
      a [0, 1] size scale. Narrow interval (confident) → 1.0; interval as
      wide as `max_reasonable_width` → `min_size_floor`.

    Strictness
    ----------
    Calls before `fit_from_residuals` raise DataFaultError. Calibration
    sets with < `min_calibration_size` samples raise DataFaultError — we do
    not silently fall back to a normal approximation.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        *,
        min_calibration_size: int = 50,
        clamp_unit_interval: bool = True,
        max_reasonable_width: float = 0.40,
        min_size_floor: float = 0.15,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if min_calibration_size < 20:
            raise ValueError("min_calibration_size must be >= 20 for meaningful coverage")
        self.alpha = float(alpha)
        self.min_calibration_size = int(min_calibration_size)
        self.clamp_unit_interval = bool(clamp_unit_interval)
        self.max_reasonable_width = float(max_reasonable_width)
        self.min_size_floor = float(min_size_floor)
        self._band: ConformalBand | None = None

    # ----------------------------- fitting -----------------------------
    def fit_from_residuals(self, residuals: Sequence[float]) -> ConformalBand:
        arr = np.asarray(residuals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < self.min_calibration_size:
            raise DataFaultError(
                f"ConformalIntervalService: only {arr.size} finite calibration "
                f"residuals (need >= {self.min_calibration_size})"
            )
        arr = np.abs(arr)
        # Finite-sample correction: use (n+1)(1-alpha)/n quantile level.
        n = arr.size
        level = min((n + 1) * (1.0 - self.alpha) / n, 1.0)
        q = float(np.quantile(arr, level, method="higher"))
        band = ConformalBand(
            alpha=self.alpha,
            quantile=q,
            n_calibration=n,
            residual_median=float(np.median(arr)),
        )
        self._band = band
        return band

    def fit_from_pairs(self, y_true: Sequence[float], y_pred: Sequence[float]) -> ConformalBand:
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        if yt.shape != yp.shape:
            raise DataFaultError("fit_from_pairs: y_true and y_pred must share shape")
        return self.fit_from_residuals(np.abs(yt - yp))

    @property
    def is_fitted(self) -> bool:
        return self._band is not None

    @property
    def band(self) -> ConformalBand:
        if self._band is None:
            raise DataFaultError("ConformalIntervalService used before fit_from_residuals()")
        return self._band

    # --------------------------- inference -----------------------------
    def interval(self, point_prediction: float) -> tuple[float, float]:
        q = self.band.quantile
        lo = float(point_prediction) - q
        hi = float(point_prediction) + q
        if self.clamp_unit_interval:
            lo = max(0.0, lo)
            hi = min(1.0, hi)
        return lo, hi

    def half_width(self) -> float:
        return self.band.quantile

    def size_multiplier(self, point_prediction: float | None = None) -> float:
        """
        Map interval width to a size scale in [min_size_floor, 1.0].
        Ignores `point_prediction` unless clamp_unit_interval=True (in which
        case saturation near 0/1 yields a narrower effective width).
        """
        if self.clamp_unit_interval and point_prediction is not None:
            lo, hi = self.interval(point_prediction)
            width = hi - lo
        else:
            width = 2.0 * self.band.quantile
        if width <= 0.0:
            return 1.0
        frac = min(width / (2.0 * self.max_reasonable_width), 1.0)
        # Linear interpolation: width=0 → 1.0; width=max → min_size_floor
        return float(1.0 - frac * (1.0 - self.min_size_floor))

    # ----------------------------- diag --------------------------------
    def summary(self) -> dict:
        b = self._band
        if b is None:
            return {"fitted": False}
        return {
            "fitted": True,
            "alpha": b.alpha,
            "coverage_target": b.coverage_target,
            "half_width": b.quantile,
            "residual_median": b.residual_median,
            "n_calibration": b.n_calibration,
        }


class MondrianConformalService:
    """
    Mondrian (per-regime) conformal prediction.

    Vanilla split-conformal assumes one population for calibration. In live
    trading, that assumption is violated by regime change: high-volatility
    regimes produce wider residuals than quiet regimes, so a pooled interval
    systematically *under-covers* in vol spikes and *over-covers* in quiet
    periods.

    Mondrian conformal fits a separate quantile per regime bucket. The
    `regime_key` is any deterministic label the caller attaches to each
    residual — for this codebase it should be
    `f"{volatility_bucket}|{horizon_bucket}"` or similar.

    Guarantees
    ──────────
    If residuals within each bucket are exchangeable, the Mondrian interval
    achieves ≥ (1-alpha) marginal coverage *within each bucket*. The pooled
    rule only guarantees unconditional coverage, which is too weak for
    regime-conditional risk.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        *,
        min_bucket_size: int = 30,
        fallback_service: ConformalIntervalService | None = None,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.min_bucket_size = int(min_bucket_size)
        self._per_bucket: dict[str, ConformalIntervalService] = {}
        self._fallback = fallback_service

    def fit_from_labelled_residuals(self, pairs: list[tuple[str, float]]) -> dict[str, int]:
        """
        Input: list of (regime_key, residual) tuples.
        Any bucket with < min_bucket_size samples is discarded — callers
        fall through to `fallback_service`.

        Returns a {bucket: n_samples} summary.
        """
        by_bucket: dict[str, list[float]] = {}
        for key, r in pairs:
            k = str(key or "unknown")
            by_bucket.setdefault(k, []).append(float(r))
        summary: dict[str, int] = {}
        self._per_bucket = {}
        for key, residuals in by_bucket.items():
            arr = np.asarray(residuals, dtype=float)
            if arr.size < self.min_bucket_size:
                summary[key] = 0
                continue
            svc = ConformalIntervalService(alpha=self.alpha, min_calibration_size=self.min_bucket_size)
            svc.fit_from_residuals(arr)
            self._per_bucket[key] = svc
            summary[key] = arr.size
        return summary

    def size_multiplier(self, point_prediction: float, regime_key: str) -> float:
        svc = self._per_bucket.get(str(regime_key)) or self._fallback
        if svc is None or not svc.is_fitted:
            return 1.0
        return svc.size_multiplier(point_prediction=point_prediction)

    def interval(self, point_prediction: float, regime_key: str) -> tuple[float, float]:
        svc = self._per_bucket.get(str(regime_key)) or self._fallback
        if svc is None or not svc.is_fitted:
            return (0.0, 1.0)
        return svc.interval(point_prediction)

    def coverage_audit(self) -> dict:
        return {
            key: svc.summary() for key, svc in self._per_bucket.items()
        } | ({"fallback": self._fallback.summary()} if self._fallback else {})


def build_family_conformal_service(
    family: str,
    *,
    logs_dir: str = "logs",
    alpha: float = 0.2,
    min_calibration_size: int = 50,
) -> ConformalIntervalService | None:
    """
    Construct and fit a ConformalIntervalService from persisted residuals
    for `family` (e.g. "btc", "weather_temperature"). Returns None if there
    are too few residuals yet (caller then skips uncertainty sizing).

    Never raises on 'not enough data yet' — that's a normal early state.
    DOES raise on corrupt calibration files (see calibration_store).
    """
    from services.calibration_store import load_residuals
    residuals = load_residuals(family, logs_dir=logs_dir)
    if residuals.size < min_calibration_size:
        return None
    svc = ConformalIntervalService(
        alpha=alpha,
        min_calibration_size=min_calibration_size,
    )
    svc.fit_from_residuals(residuals)
    return svc


__all__ = [
    "ConformalIntervalService",
    "ConformalBand",
    "MondrianConformalService",
    "build_family_conformal_service",
]
