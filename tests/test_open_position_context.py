import ast
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd


def _load_summarize_open_position_context():
    source_path = Path(__file__).resolve().parent.parent / "supervisor.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    func_node = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "summarize_open_position_context"
    )
    func_source = dedent(ast.get_source_segment(source, func_node))
    namespace = {
        "pd": pd,
        "np": np,
        "_active_trades_to_positions_frame": lambda active_trades: pd.DataFrame(),
    }
    exec(func_source, namespace)
    return namespace["summarize_open_position_context"]


def test_summarize_open_position_context_handles_duplicate_numeric_columns():
    summarize_open_position_context = _load_summarize_open_position_context()
    positions_df = pd.DataFrame(
        [
            [None, 2.0, 0.40, None, 0.50],
            [3.0, None, None, 0.20, 0.25],
        ],
        columns=[
            "shares",
            "shares",
            "entry_price",
            "avg_entry_price",
            "current_price",
        ],
    )

    summary = summarize_open_position_context(positions_df=positions_df)

    assert summary["open_positions_count"] == 2
    assert round(summary["open_positions_negotiated_value_total"], 4) == 1.4
    assert round(summary["open_positions_current_value_total"], 4) == 1.75
    assert round(summary["open_positions_unrealized_pnl_total"], 4) == 0.35
    assert summary["open_positions_winner_count"] == 2
    assert summary["open_positions_loser_count"] == 0
