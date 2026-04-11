from __future__ import annotations

from pathlib import Path

import pandas as pd

from brain_log_routing import append_csv_with_brain_mirrors, overwrite_csv_with_brain_mirrors


def test_append_csv_with_brain_mirrors_splits_shared_operational_log(tmp_path: Path):
    shared_logs = tmp_path / "logs"
    shared_weights = tmp_path / "weights"
    path = shared_logs / "signals.csv"
    frame = pd.DataFrame(
        [
            {"market_title": "Will the price of Bitcoin be above $90,000?", "market_family": "btc", "brain_id": "btc_brain"},
            {"market_title": "Will the highest temperature in NYC be 64F or higher?", "market_family": "weather_temperature", "brain_id": "weather_temperature_brain"},
        ]
    )

    append_csv_with_brain_mirrors(path, frame, shared_logs_dir=shared_logs, shared_weights_dir=shared_weights)

    shared_df = pd.read_csv(path)
    btc_df = pd.read_csv(shared_logs / "btc" / "signals.csv")
    weather_df = pd.read_csv(shared_logs / "weather_temperature" / "signals.csv")

    assert len(shared_df.index) == 2
    assert len(btc_df.index) == 1
    assert btc_df.iloc[0]["market_family"] == "btc"
    assert len(weather_df.index) == 1
    assert weather_df.iloc[0]["market_family"] == "weather_temperature"


def test_overwrite_csv_with_brain_mirrors_rewrites_per_family_views(tmp_path: Path):
    shared_logs = tmp_path / "logs"
    shared_weights = tmp_path / "weights"
    path = shared_logs / "positions.csv"
    frame = pd.DataFrame(
        [
            {"market": "BTC market", "market_family": "btc", "brain_id": "btc_brain"},
            {"market": "Weather market", "market_family": "weather_temperature", "brain_id": "weather_temperature_brain"},
        ]
    )

    overwrite_csv_with_brain_mirrors(path, frame, shared_logs_dir=shared_logs, shared_weights_dir=shared_weights)
    overwrite_csv_with_brain_mirrors(path, frame.iloc[:1].copy(), shared_logs_dir=shared_logs, shared_weights_dir=shared_weights)

    shared_df = pd.read_csv(path)
    btc_df = pd.read_csv(shared_logs / "btc" / "positions.csv")
    weather_df = pd.read_csv(shared_logs / "weather_temperature" / "positions.csv")

    assert len(shared_df.index) == 1
    assert len(btc_df.index) == 1
    assert btc_df.iloc[0]["market_family"] == "btc"
    assert weather_df.empty

