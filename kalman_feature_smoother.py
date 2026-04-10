from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _safe_float(value, default=None):
    try:
        num = float(value)
    except Exception:
        return default
    if not np.isfinite(num):
        return default
    return float(num)


@dataclass
class AdaptiveScalarKalmanFilter:
    """
    Lightweight scalar Kalman filter with scale-aware noise terms.

    The noise scales with the current signal magnitude so the same class can
    smooth both price-level and basis/return-style features.
    """

    process_noise_ratio: float = 0.0008
    measurement_noise_ratio: float = 0.0035
    min_scale: float = 1.0
    estimate: float | None = None
    error_covariance: float = 1.0

    def update(self, measurement):
        value = _safe_float(measurement, default=None)
        if value is None:
            return self.estimate

        scale = max(abs(value), abs(self.estimate or 0.0), float(self.min_scale))
        measurement_variance = max((scale * self.measurement_noise_ratio) ** 2, 1e-12)
        process_variance = max((scale * self.process_noise_ratio) ** 2, 1e-12)

        if self.estimate is None:
            self.estimate = value
            self.error_covariance = measurement_variance
            return float(self.estimate)

        prior_estimate = float(self.estimate)
        prior_error = float(self.error_covariance) + process_variance
        kalman_gain = prior_error / (prior_error + measurement_variance)
        innovation = value - prior_estimate

        self.estimate = prior_estimate + (kalman_gain * innovation)
        self.error_covariance = max((1.0 - kalman_gain) * prior_error, 1e-12)
        return float(self.estimate)
