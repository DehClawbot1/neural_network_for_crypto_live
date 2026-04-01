import pandas as pd

from return_calibration import (
    calibrate_return_predictions,
    clip_expected_return_series,
    fit_return_calibration,
    transform_return_targets,
)


def test_return_calibration_round_trip_stays_bounded():
    raw = pd.Series([-0.22, -0.08, 0.0, 0.04, 0.19, 0.41])
    calibration = fit_return_calibration(raw)
    calibration["reliability"] = 1.0
    transformed = transform_return_targets(raw, calibration)
    restored = calibrate_return_predictions(transformed, calibration, index=raw.index)

    assert restored.min() >= calibration["clip_lower"] - 1e-9
    assert restored.max() <= calibration["clip_upper"] + 1e-9
    assert abs(float(restored.iloc[0]) - float(raw.iloc[0])) < 1e-6
    assert abs(float(restored.iloc[-1]) - float(raw.iloc[-1])) < 1e-6


def test_legacy_predictions_are_hard_clipped_to_sane_bounds():
    restored = calibrate_return_predictions([7.5, -9.0, 0.12], None)

    assert float(restored.iloc[0]) == 0.60
    assert float(restored.iloc[1]) == -0.60
    assert float(restored.iloc[2]) == 0.12


def test_clip_expected_return_series_handles_non_finite_values():
    clipped = clip_expected_return_series(pd.Series([float("inf"), float("-inf"), 0.33, -0.91]))

    assert list(clipped.round(2)) == [0.0, 0.0, 0.33, -0.60]


def test_small_sample_reliability_reverts_toward_training_median():
    raw = pd.Series([-0.04, 0.01, 0.03, 0.05])
    calibration = fit_return_calibration(raw)
    calibration["reliability"] = 0.2
    calibration["train_median"] = 0.01

    restored = calibrate_return_predictions([0.40], calibration)

    assert float(restored.iloc[0]) < 0.10
    assert float(restored.iloc[0]) > 0.01
