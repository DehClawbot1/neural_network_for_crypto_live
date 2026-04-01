from balance_normalization import extract_balance_value, normalize_allowance_balance


def test_normalize_allowance_balance_scales_integer_like_units():
    assert normalize_allowance_balance("31005342", asset_type="CONDITIONAL") == 31.005342
    assert normalize_allowance_balance("251860", asset_type="CONDITIONAL") == 0.25186


def test_normalize_allowance_balance_keeps_decimal_values():
    assert normalize_allowance_balance("0.25186", asset_type="CONDITIONAL") == 0.25186
    assert normalize_allowance_balance(31.005342, asset_type="CONDITIONAL") == 31.005342


def test_extract_balance_value_prefers_balance_fields():
    key, value = extract_balance_value({"balance": "123", "available": "456"})
    assert key == "balance"
    assert value == "123"
