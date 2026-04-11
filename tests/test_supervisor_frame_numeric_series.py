import ast
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd


def _load_supervisor_numeric_helpers():
    source_path = Path(__file__).resolve().parent.parent / "supervisor.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    helper_names = {"_frame_column_as_series", "_frame_numeric_series"}
    snippets = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in helper_names:
            snippets.append(dedent(ast.get_source_segment(source, node)))
    namespace = {"pd": pd, "np": np}
    exec("\n\n".join(snippets), namespace)
    return namespace["_frame_numeric_series"]


def test_frame_numeric_series_collapses_duplicate_columns_to_1d_series():
    frame_numeric_series = _load_supervisor_numeric_helpers()
    df = pd.DataFrame(
        [
            [0.10, None],
            [None, 0.25],
            [0.30, 0.35],
        ],
        columns=["expected_return", "expected_return"],
    )

    series = frame_numeric_series(df, "expected_return", 0.0)

    assert isinstance(series, pd.Series)
    assert list(series.round(4)) == [0.10, 0.25, 0.30]
