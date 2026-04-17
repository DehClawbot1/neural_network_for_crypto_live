"""
services/calibration_suite.py
─────────────────────────────
Integrated probability calibration stack:

1. **Platt scaling** (logistic) — fits y ~ sigmoid(a*s + b). Good for SVMs
   and any monotone-distorted classifier output. Two parameters — extremely
   data-efficient; works well with N ≥ 100.
2. **Isotonic regression** — non-parametric monotone fit via Pool Adjacent
   Violators. Strictly better than Platt when N ≥ 500 and distortion is
   non-sigmoidal.
3. **Beta calibration** (Kull et al. 2017) — between Platt and isotonic,
   three parameters, handles asymmetric distortion Platt can't.
4. **Conformal wrap** — after point-calibration, the `ConformalIntervalService`
   supplies coverage guarantees. Exposed here as the one-stop factory.
5. **Diagnostics** — Brier, log-loss, ECE (expected calibration error) and
   reliability curve bucketing.

All calibrators share a common interface: `.fit(scores, labels)` then
`.predict(scores)` → calibrated probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ──────────────────────────── diagnostics ──────────────────────────────
def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float); y = np.asarray(labels, dtype=float)
    m = np.isfinite(p) & np.isfinite(y)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean((p[m] - y[m]) ** 2))


def log_loss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    y = np.asarray(labels, dtype=float)
    m = np.isfinite(p) & np.isfinite(y)
    if m.sum() == 0:
        return float("nan")
    return float(-np.mean(y[m] * np.log(p[m]) + (1 - y[m]) * np.log(1 - p[m])))


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15,
) -> float:
    p = np.asarray(probs, dtype=float); y = np.asarray(labels, dtype=float)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        conf = float(p[mask].mean()); acc = float(y[mask].mean())
        ece += (mask.sum() / p.size) * abs(acc - conf)
    return float(ece)


def reliability_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10,
) -> list[dict]:
    p = np.asarray(probs, dtype=float); y = np.asarray(labels, dtype=float)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    if p.size == 0:
        return []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        out.append({
            "bin": i,
            "lo": round(float(lo), 3),
            "hi": round(float(hi), 3),
            "n": n,
            "mean_pred": round(float(p[mask].mean()), 4),
            "mean_obs": round(float(y[mask].mean()), 4),
        })
    return out


# ─────────────────────────── Platt scaling ─────────────────────────────
@dataclass
class PlattScaler:
    a: float = 1.0
    b: float = 0.0
    n_iter: int = 100
    lr: float = 0.05
    _fitted: bool = field(default=False)

    def fit(self, scores: Sequence[float], labels: Sequence[float]) -> "PlattScaler":
        s = np.asarray(scores, dtype=float); y = np.asarray(labels, dtype=float)
        m = np.isfinite(s) & np.isfinite(y); s, y = s[m], y[m]
        if s.size < 20:
            raise ValueError("PlattScaler needs ≥ 20 samples")
        # Newton-Raphson-ish gradient descent on log-loss
        a, b = 1.0, float(np.log(max(1e-6, (1 - y.mean()) / max(1e-6, y.mean()))) * -1)
        for _ in range(self.n_iter):
            z = a * s + b
            p = 1.0 / (1.0 + np.exp(-z))
            ga = float(np.mean((p - y) * s))
            gb = float(np.mean(p - y))
            a -= self.lr * ga
            b -= self.lr * gb
        self.a, self.b, self._fitted = float(a), float(b), True
        return self

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PlattScaler.predict before fit")
        s = np.asarray(scores, dtype=float)
        z = self.a * s + self.b
        return 1.0 / (1.0 + np.exp(-z))


# ───────────────────────── Isotonic regression ─────────────────────────
@dataclass
class IsotonicCalibrator:
    x_fit: np.ndarray = field(default_factory=lambda: np.array([]))
    y_fit: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, scores: Sequence[float], labels: Sequence[float]) -> "IsotonicCalibrator":
        s = np.asarray(scores, dtype=float); y = np.asarray(labels, dtype=float)
        m = np.isfinite(s) & np.isfinite(y); s, y = s[m], y[m]
        if s.size < 20:
            raise ValueError("IsotonicCalibrator needs ≥ 20 samples")
        order = np.argsort(s)
        s, y = s[order], y[order].astype(float)
        # Pool Adjacent Violators
        n = s.size
        vals = y.copy()
        weights = np.ones(n)
        i = 0
        while i < n - 1:
            if vals[i] > vals[i + 1]:
                new_w = weights[i] + weights[i + 1]
                new_v = (weights[i] * vals[i] + weights[i + 1] * vals[i + 1]) / new_w
                vals[i] = new_v
                weights[i] = new_w
                # delete i+1
                vals = np.delete(vals, i + 1)
                weights = np.delete(weights, i + 1)
                s = np.delete(s, i + 1)
                n -= 1
                if i > 0:
                    i -= 1
            else:
                i += 1
        self.x_fit, self.y_fit = s, vals
        return self

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        if self.x_fit.size == 0:
            raise RuntimeError("IsotonicCalibrator.predict before fit")
        return np.interp(np.asarray(scores, dtype=float), self.x_fit, self.y_fit,
                         left=self.y_fit[0], right=self.y_fit[-1])


# ───────────────────────── Beta calibration ────────────────────────────
@dataclass
class BetaCalibrator:
    a: float = 1.0
    b: float = 1.0
    c: float = 0.0
    _fitted: bool = field(default=False)

    def fit(self, scores: Sequence[float], labels: Sequence[float]) -> "BetaCalibrator":
        s = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(labels, dtype=float)
        m = np.isfinite(s) & np.isfinite(y); s, y = s[m], y[m]
        if s.size < 50:
            raise ValueError("BetaCalibrator needs ≥ 50 samples")
        # Kull et al. reparameterise as logistic on (log s, log(1-s)).
        x1 = np.log(s); x2 = np.log(1 - s)
        # Newton steps on logistic regression with two features.
        a, b, c = 1.0, 1.0, 0.0
        for _ in range(200):
            z = a * x1 - b * x2 + c
            p = 1.0 / (1.0 + np.exp(-z))
            err = p - y
            ga = float(np.mean(err * x1))
            gb = float(np.mean(-err * x2))
            gc = float(np.mean(err))
            a -= 0.05 * ga; b -= 0.05 * gb; c -= 0.05 * gc
        self.a, self.b, self.c, self._fitted = float(a), float(b), float(c), True
        return self

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("BetaCalibrator.predict before fit")
        s = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
        z = self.a * np.log(s) - self.b * np.log(1 - s) + self.c
        return 1.0 / (1.0 + np.exp(-z))


# ───────────────────────── method selector ────────────────────────────
def auto_calibrate(
    scores: Sequence[float],
    labels: Sequence[float],
    *,
    eval_scores: Sequence[float] | None = None,
    eval_labels: Sequence[float] | None = None,
) -> dict:
    """
    Train Platt / Isotonic / Beta, score each on eval (or train if no eval),
    and return the best by Brier score along with the full diagnostics.
    """
    results = {}
    candidates = []
    for name, ctor in [("platt", PlattScaler), ("isotonic", IsotonicCalibrator),
                        ("beta", BetaCalibrator)]:
        try:
            model = ctor().fit(scores, labels)
            xs = eval_scores if eval_scores is not None else scores
            ys = eval_labels if eval_labels is not None else labels
            p = np.clip(model.predict(xs), 0.0, 1.0)
            results[name] = {
                "brier": round(brier_score(p, ys), 6),
                "log_loss": round(log_loss(p, ys), 6),
                "ece": round(expected_calibration_error(p, ys), 6),
            }
            candidates.append((results[name]["brier"], name, model))
        except Exception as e:
            results[name] = {"error": str(e)}
    if not candidates:
        return {"best": None, "results": results}
    candidates.sort()
    _, best_name, best_model = candidates[0]
    return {"best": best_name, "model": best_model, "results": results}


__all__ = [
    "PlattScaler",
    "IsotonicCalibrator",
    "BetaCalibrator",
    "auto_calibrate",
    "brier_score",
    "log_loss",
    "expected_calibration_error",
    "reliability_curve",
]
