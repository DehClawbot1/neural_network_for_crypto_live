from collections import Counter

from model_feature_catalog import DEFAULT_TABULAR_FEATURE_COLUMNS, SEQUENCE_BASE_COLUMNS


def test_default_tabular_feature_columns_are_unique():
    counts = Counter(DEFAULT_TABULAR_FEATURE_COLUMNS)
    duplicates = {name: count for name, count in counts.items() if count > 1}
    assert duplicates == {}


def test_sequence_base_columns_are_unique():
    counts = Counter(SEQUENCE_BASE_COLUMNS)
    duplicates = {name: count for name, count in counts.items() if count > 1}
    assert duplicates == {}
