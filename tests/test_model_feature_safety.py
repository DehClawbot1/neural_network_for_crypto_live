import numpy as np
import pandas as pd

from model_feature_safety import drop_all_nan_features


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
