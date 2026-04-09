import ast
from pathlib import Path
from textwrap import dedent

import pandas as pd


def _load_split_entry_pipeline_signals():
    source_path = Path(__file__).resolve().parent.parent / "supervisor.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    func_node = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "split_entry_pipeline_signals"
    )
    func_source = dedent(ast.get_source_segment(source, func_node))
    namespace = {"pd": pd}
    exec(func_source, namespace)
    return namespace["split_entry_pipeline_signals"]


def test_split_entry_pipeline_signals_drops_global_scan_from_entry_path():
    split_entry_pipeline_signals = _load_split_entry_pipeline_signals()
    signals_df = pd.DataFrame(
        [
            {"signal_source": "leaderboard_wallet", "market_slug": "m1", "entry_intent": "OPEN_LONG"},
            {"signal_source": "global_btc_scan", "market_slug": "m2", "entry_intent": "OPEN_LONG"},
            {"signal_source": "always_on_market", "market_slug": "m3", "entry_intent": "OPEN_LONG"},
        ]
    )

    filtered, stats = split_entry_pipeline_signals(signals_df)

    assert len(filtered) == 2
    assert set(filtered["signal_source"]) == {"leaderboard_wallet", "always_on_market"}
    assert stats["dropped_rows"] == 1
    assert stats["dropped_global_btc_scan"] == 1


def test_split_entry_pipeline_signals_drops_stale_open_long_but_keeps_stale_closes():
    split_entry_pipeline_signals = _load_split_entry_pipeline_signals()
    signals_df = pd.DataFrame(
        [
            {
                "signal_source": "leaderboard_wallet",
                "market_slug": "m1",
                "entry_intent": "OPEN_LONG",
                "source_wallet_fresh": False,
            },
            {
                "signal_source": "leaderboard_wallet",
                "market_slug": "m2",
                "entry_intent": "OPEN_LONG",
                "source_wallet_fresh": True,
            },
            {
                "signal_source": "leaderboard_wallet",
                "market_slug": "m3",
                "entry_intent": "CLOSE_LONG",
                "source_wallet_fresh": False,
            },
        ]
    )

    filtered, stats = split_entry_pipeline_signals(signals_df)

    assert len(filtered) == 2
    assert set(filtered["market_slug"]) == {"m2", "m3"}
    assert stats["dropped_rows"] == 1
    assert stats["dropped_global_btc_scan"] == 0
    assert stats["dropped_stale_wallet_entries"] == 1
