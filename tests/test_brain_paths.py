from pathlib import Path

import pandas as pd

from brain_paths import (
    BTC_BRAIN_ID,
    WEATHER_BRAIN_ID,
    filter_frame_for_brain,
    infer_market_family_from_row,
    resolve_brain_context,
)


def test_resolve_brain_context_uses_family_specific_paths(tmp_path: Path):
    btc_context = resolve_brain_context("btc", shared_logs_dir=tmp_path / "logs", shared_weights_dir=tmp_path / "weights")
    weather_context = resolve_brain_context(
        "weather_temperature",
        shared_logs_dir=tmp_path / "logs",
        shared_weights_dir=tmp_path / "weights",
    )

    assert btc_context.brain_id == BTC_BRAIN_ID
    assert btc_context.logs_dir == tmp_path / "logs" / "btc"
    assert btc_context.weights_dir == tmp_path / "weights" / "btc"

    assert weather_context.brain_id == WEATHER_BRAIN_ID
    assert weather_context.logs_dir == tmp_path / "logs" / "weather_temperature"
    assert weather_context.weights_dir == tmp_path / "weights" / "weather_temperature"


def test_infer_market_family_handles_pandas_series():
    weather_row = pd.Series({"market_title": "Will the highest temperature in NYC be 64F or higher?"})
    btc_row = pd.Series({"market_title": "Will the price of Bitcoin be above $90,000?"})

    assert infer_market_family_from_row(weather_row) == "weather_temperature"
    assert infer_market_family_from_row(btc_row) == "btc"


def test_filter_frame_for_brain_sets_brain_identity(tmp_path: Path):
    frame = pd.DataFrame(
        [
            {"market_title": "Will the price of Bitcoin be above $90,000?", "market_family": "btc"},
            {"market_title": "Will the highest temperature in NYC be 64F or higher?", "market_family": "weather_temperature"},
        ]
    )

    btc_context = resolve_brain_context("btc", shared_logs_dir=tmp_path / "logs", shared_weights_dir=tmp_path / "weights")
    weather_context = resolve_brain_context(
        "weather_temperature",
        shared_logs_dir=tmp_path / "logs",
        shared_weights_dir=tmp_path / "weights",
    )

    btc_only = filter_frame_for_brain(frame, btc_context)
    weather_only = filter_frame_for_brain(frame, weather_context)

    assert len(btc_only.index) == 1
    assert btc_only.iloc[0]["brain_id"] == BTC_BRAIN_ID
    assert btc_only.iloc[0]["market_family"] == "btc"

    assert len(weather_only.index) == 1
    assert weather_only.iloc[0]["brain_id"] == WEATHER_BRAIN_ID
    assert weather_only.iloc[0]["market_family"] == "weather_temperature"
