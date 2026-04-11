import numpy as np


def _sanitize_macro_context(macro_context: dict) -> dict:
    """Mimics the scalar sanitizer pattern from supervisor.py."""
    _scalar_types = (int, float, str, bool, type(None))
    safe = {}
    for k, v in macro_context.items():
        if isinstance(v, _scalar_types) or (
            hasattr(np, "integer") and isinstance(v, (np.integer, np.floating, np.bool_))
        ):
            safe[k] = v
    return safe


def test_all_scalar_values_kept():
    ctx = {"a": 1, "b": 2.5, "c": "hello", "d": True}
    result = _sanitize_macro_context(ctx)
    assert result == ctx


def test_list_value_filtered_out():
    ctx = {"a": 1, "bad": [1, 2, 3]}
    result = _sanitize_macro_context(ctx)
    assert "a" in result
    assert "bad" not in result


def test_numpy_array_filtered_out():
    ctx = {"a": 1, "arr": np.array([1, 2, 3])}
    result = _sanitize_macro_context(ctx)
    assert "a" in result
    assert "arr" not in result


def test_none_value_kept():
    ctx = {"a": None, "b": 1}
    result = _sanitize_macro_context(ctx)
    assert "a" in result
    assert result["a"] is None


def test_numpy_integer_and_float_kept():
    ctx = {"i": np.int64(42), "f": np.float64(3.14)}
    result = _sanitize_macro_context(ctx)
    assert result["i"] == 42
    assert abs(result["f"] - 3.14) < 1e-9
