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


def test_resolve_brain_context_btc_returns_correct_paths(tmp_path: Path):
    ctx = resolve_brain_context("btc", shared_logs_dir=tmp_path / "logs", shared_weights_dir=tmp_path / "weights")
    assert ctx.logs_dir == tmp_path / "logs" / "btc"
    assert ctx.weights_dir == tmp_path / "weights" / "btc"


def test_resolve_brain_context_weather_returns_correct_paths(tmp_path: Path):
    ctx = resolve_brain_context(
        "weather_temperature", shared_logs_dir=tmp_path / "logs", shared_weights_dir=tmp_path / "weights"
    )
    assert ctx.logs_dir == tmp_path / "logs" / "weather_temperature"
    assert ctx.weights_dir == tmp_path / "weights" / "weather_temperature"


def test_infer_market_family_from_row_btc_dict():
    row = {"market_title": "Will Bitcoin exceed $100k?", "market_family": "btc"}
    assert infer_market_family_from_row(row) == "btc"


def test_infer_market_family_from_row_weather_highest_temperature():
    row = {"question": "Will the highest temperature in NYC be 64F or higher?"}
    assert infer_market_family_from_row(row) == "weather_temperature"


def test_filter_frame_for_brain_splits_mixed_dataframe(tmp_path: Path):
    frame = pd.DataFrame([
        {"market_title": "BTC price question", "market_family": "btc"},
        {"market_title": "Will the highest temperature in Dallas be 70F?", "market_family": "weather_temperature"},
        {"market_title": "Another crypto market", "market_family": "btc"},
    ])
    btc_ctx = resolve_brain_context("btc", shared_logs_dir=tmp_path / "logs", shared_weights_dir=tmp_path / "weights")
    weather_ctx = resolve_brain_context(
        "weather_temperature", shared_logs_dir=tmp_path / "logs", shared_weights_dir=tmp_path / "weights"
    )
    btc_only = filter_frame_for_brain(frame, btc_ctx)
    weather_only = filter_frame_for_brain(frame, weather_ctx)
    assert len(btc_only) == 2
    assert len(weather_only) == 1
