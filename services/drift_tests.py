"""
services/drift_tests.py
───────────────────────
Formal statistical drift detectors — upgrade path beyond PSI + heuristics.

Included
--------
1. **CUSUM (Cumulative Sum)** — Page, 1954. Detects mean shift in a
   residual stream with controllable false-alarm rate. Classical sequential
   change-point detector used in industrial process control and quant ops.
2. **Page-Hinkley** — online variant of CUSUM with a forgetting factor;
   better-suited for non-stationary distributions.
3. **Kolmogorov-Smirnov two-sample test** — non-parametric distributional
   equality test. Returns D-statistic + p-value; flag when p < alpha.
4. **Welch's t-test** — mean-equality test robust to unequal variance.

All detectors are state-capable: persist the `state` dict, rehydrate next
cycle, call `update(x)` with new observations. No lookback window needed.

References
----------
- Page (1954), "Continuous Inspection Schemes"
- Mouss et al. (2004), "Test of Page-Hinkley …"
- Gama et al. (2014), "A Survey on Concept Drift Adaptation"
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# ─────────────────────────────── CUSUM ──────────────────────────────────
@dataclass
class CUSUMState:
    """Two-sided CUSUM for detecting positive / negative mean shifts."""
    mean: float = 0.0           # reference mean (EWMA-updated during warm-up)
    std: float = 1.0            # reference std
    g_pos: float = 0.0          # upward cumulative sum
    g_neg: float = 0.0          # downward cumulative sum
    n: int = 0
    warmed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "CUSUMState":
        return cls(**(d or {}))


class CUSUM:
    """
    Two-sided CUSUM. Fires an alarm when g_pos > h*std or g_neg < -h*std.

    Parameters
    ----------
    k : float (default 0.5)  — reference value / slack (in std units). Typical
                               choice is half the smallest shift you want to
                               detect. k=0.5 detects a 1-std shift quickly.
    h : float (default 5.0)  — decision threshold (in std units). Higher h →
                               fewer false alarms, slower detection. h=5
                               roughly corresponds to ARL₀ ≈ 465 under N(0,1).
    warmup : int (default 50) — observations to estimate reference mean/std
                                before arming the detector.
    """

    def __init__(self, k: float = 0.5, h: float = 5.0, warmup: int = 50):
        if k <= 0 or h <= 0:
            raise ValueError("k and h must be positive")
        self.k = float(k)
        self.h = float(h)
        self.warmup = int(warmup)
        self.state = CUSUMState()
        self._warmup_buf: list[float] = []

    def update(self, x: float) -> dict:
        s = self.state
        s.n += 1
        if not s.warmed:
            self._warmup_buf.append(float(x))
            if len(self._warmup_buf) >= self.warmup:
                arr = np.asarray(self._warmup_buf, dtype=float)
                s.mean = float(arr.mean())
                s.std = max(float(arr.std(ddof=1)), 1e-9)
                s.warmed = True
            return {"alarm": False, "g_pos": s.g_pos, "g_neg": s.g_neg, "warmed": s.warmed}

        z = (float(x) - s.mean) / s.std
        s.g_pos = max(0.0, s.g_pos + z - self.k)
        s.g_neg = min(0.0, s.g_neg + z + self.k)
        alarm_up = s.g_pos > self.h
        alarm_dn = s.g_neg < -self.h
        return {
            "alarm": bool(alarm_up or alarm_dn),
            "direction": "up" if alarm_up else ("down" if alarm_dn else "none"),
            "g_pos": round(s.g_pos, 4),
            "g_neg": round(s.g_neg, 4),
            "z": round(z, 4),
            "warmed": True,
        }

    def reset(self) -> None:
        self.state = CUSUMState()
        self._warmup_buf = []


# ──────────────────────────── Page-Hinkley ──────────────────────────────
@dataclass
class PageHinkleyState:
    cum: float = 0.0
    min_cum: float = 0.0
    max_cum: float = 0.0
    mean: float = 0.0
    n: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class PageHinkley:
    """
    Online Page-Hinkley test. Detects a change in the mean of a stream.

    Parameters
    ----------
    delta : magnitude of allowable change (absorbs small drift noise).
    lambda_ : detection threshold. Alarm when |cum - extremum| > lambda_.
    alpha : forgetting factor for the running mean (0 < alpha <= 1).
            alpha=1 is a plain running mean; alpha=0.999 forgets slowly.
    """

    def __init__(self, delta: float = 0.005, lambda_: float = 50.0, alpha: float = 0.9999):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self.delta = float(delta)
        self.lambda_ = float(lambda_)
        self.alpha = float(alpha)
        self.state = PageHinkleyState()

    def update(self, x: float) -> dict:
        s = self.state
        s.n += 1
        if s.n == 1:
            s.mean = float(x)
        else:
            s.mean = self.alpha * s.mean + (1.0 - self.alpha) * float(x)
        dev = float(x) - s.mean - self.delta
        s.cum += dev
        s.min_cum = min(s.min_cum, s.cum)
        s.max_cum = max(s.max_cum, s.cum)
        ph_up = s.cum - s.min_cum
        ph_dn = s.max_cum - s.cum
        return {
            "alarm": bool(ph_up > self.lambda_ or ph_dn > self.lambda_),
            "direction": "up" if ph_up > self.lambda_ else ("down" if ph_dn > self.lambda_ else "none"),
            "ph_up": round(ph_up, 4),
            "ph_dn": round(ph_dn, 4),
            "mean": round(s.mean, 6),
        }

    def reset(self) -> None:
        self.state = PageHinkleyState()


# ──────────────────────────── KS two-sample ─────────────────────────────
def ks_two_sample(reference: np.ndarray, current: np.ndarray) -> dict:
    """
    Kolmogorov-Smirnov two-sample test. Returns D-statistic and approximate
    p-value (Smirnov asymptotic formula). Use to compare a calibration
    window vs a recent production window.
    """
    a = np.asarray(reference, dtype=float)
    b = np.asarray(current, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 20 or b.size < 20:
        return {"n_ref": int(a.size), "n_cur": int(b.size), "D": None, "p": None,
                "status": "insufficient"}
    data_all = np.concatenate([a, b])
    cdf_a = np.searchsorted(np.sort(a), data_all, side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), data_all, side="right") / b.size
    D = float(np.max(np.abs(cdf_a - cdf_b)))
    # Smirnov asymptotic p-value
    en = np.sqrt(a.size * b.size / (a.size + b.size))
    x = (en + 0.12 + 0.11 / en) * D
    # Kolmogorov distribution tail sum
    p = 0.0
    for j in range(1, 101):
        term = 2.0 * ((-1) ** (j - 1)) * np.exp(-2.0 * j * j * x * x)
        p += term
        if abs(term) < 1e-8:
            break
    p = float(max(0.0, min(1.0, p)))
    return {"n_ref": int(a.size), "n_cur": int(b.size), "D": round(D, 4),
            "p": round(p, 6), "status": "ok"}


# ──────────────────────────── Welch's t-test ────────────────────────────
def welch_t_test(reference: np.ndarray, current: np.ndarray) -> dict:
    a = np.asarray(reference, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(current, dtype=float);   b = b[np.isfinite(b)]
    if a.size < 20 or b.size < 20:
        return {"t": None, "df": None, "p_approx": None, "status": "insufficient"}
    ma, mb = float(a.mean()), float(b.mean())
    va = float(a.var(ddof=1)); vb = float(b.var(ddof=1))
    se = np.sqrt(va / a.size + vb / b.size)
    if se <= 0:
        return {"t": 0.0, "df": None, "p_approx": 1.0, "status": "ok"}
    t = (ma - mb) / se
    # Welch-Satterthwaite df
    df = (va / a.size + vb / b.size) ** 2 / (
        (va / a.size) ** 2 / (a.size - 1) + (vb / b.size) ** 2 / (b.size - 1)
    )
    # Normal-tail approximation (df > 30 usual)
    p_approx = 2.0 * (1.0 - _norm_cdf(abs(t)))
    return {"t": round(float(t), 4), "df": round(float(df), 2),
            "p_approx": round(float(p_approx), 6), "mean_ref": ma, "mean_cur": mb,
            "status": "ok"}


def _norm_cdf(z: float) -> float:
    # Abramowitz & Stegun 7.1.26 approximation
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))


def _erf(x: float) -> float:
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


__all__ = ["CUSUM", "PageHinkley", "CUSUMState", "PageHinkleyState",
           "ks_two_sample", "welch_t_test"]
