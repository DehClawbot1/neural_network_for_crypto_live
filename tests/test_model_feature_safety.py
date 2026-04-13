import numpy as np
import pandas as pd

from model_feature_safety import clean_dataframe_for_training, drop_all_nan_features


def test_all_nan_columns_are_dropped():
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, 2.0], "c": [np.nan, np.nan]})
    usable, dropped = drop_all_nan_features(df, ["a", "b", "c"])
    assert usable == ["b"]
    assert set(dropped) == {"a", "c"}


def test_columns_with_some_valid_data_are_kept():
    df = pd.DataFrame({"x": [np.nan, 1.0], "y": [2.0, 3.0]})
    usable, dropped = drop_all_nan_features(df, ["x", "y"])
    assert usable == ["x", "y"]
    assert dropped == []


def test_empty_dataframe_returns_empty_lists():
    df = pd.DataFrame()
    usable, dropped = drop_all_nan_features(df, ["a", "b"])
    assert usable == []
    assert dropped == []


def test_clean_dataframe_for_training_projects_historical_dataset_columns():
    df = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-12T10:00:00Z",
                "market_family": "btc",
                "brain_id": "btc_brain",
                "token_id": "tok-1",
                "entry_price": 0.42,
                "current_price": 0.45,
                "wallet_alpha_30d": 0.12,
                "spread": 0.03,
                "debug_runtime_only": "drop-me",
                "all_nan_runtime": np.nan,
            }
        ]
    )

    cleaned = clean_dataframe_for_training(df, context="historical_dataset")

    assert "wallet_alpha_30d" in cleaned.columns
    assert "spread" in cleaned.columns
    assert "debug_runtime_only" not in cleaned.columns
    assert "all_nan_runtime" not in cleaned.columns


def test_clean_dataframe_for_training_projects_sequence_dataset_columns():
    df = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-12T10:00:00Z",
                "market_family": "btc",
                "brain_id": "btc_brain",
                "token_id": "tok-1",
                "entry_price": 0.42,
                "recent_token_activity_5": 3,
                "wallet_alpha_30d": 0.12,
                "wallet_alpha_30d_lag_1": 0.10,
                "tp_before_sl_60m": 1,
                "target_up": 1,
                "debug_runtime_only": "drop-me",
            }
        ]
    )

    cleaned = clean_dataframe_for_training(df, context="sequence_dataset")

    assert "wallet_alpha_30d" in cleaned.columns
    assert "wallet_alpha_30d_lag_1" in cleaned.columns
    assert "tp_before_sl_60m" in cleaned.columns
    assert "debug_runtime_only" not in cleaned.columns


def test_clean_dataframe_for_training_preserves_identity_columns_as_strings():
    df = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-12T10:00:00Z",
                "token_id": "101146002438600581234567890123456789012345678901234567890123456789012345678901",
                "condition_id": "0xabc123",
                "trader_wallet": "0xwallet",
                "market_family": "btc",
                "wallet_alpha_30d": "0.12",
            }
        ]
    )

    cleaned = clean_dataframe_for_training(df, context="historical_dataset")

    assert str(cleaned.loc[0, "token_id"]).startswith("10114600243860058")
    assert cleaned["token_id"].dtype.name.startswith("string")
    assert cleaned["condition_id"].dtype.name.startswith("string")
    assert cleaned["trader_wallet"].dtype.name.startswith("string")
