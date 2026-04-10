import ast
import warnings
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


def test_split_entry_pipeline_signals_handles_duplicate_source_wallet_fresh_columns():
    split_entry_pipeline_signals = _load_split_entry_pipeline_signals()
    signals_df = pd.DataFrame(
        [
            ["leaderboard_wallet", "OPEN_LONG", False, True, "m1"],
            ["leaderboard_wallet", "OPEN_LONG", True, True, "m2"],
            ["leaderboard_wallet", "CLOSE_LONG", False, False, "m3"],
        ],
        columns=[
            "signal_source",
            "entry_intent",
            "source_wallet_fresh",
            "source_wallet_fresh",
            "market_slug",
        ],
    )

    filtered, stats = split_entry_pipeline_signals(signals_df)

    assert set(filtered["market_slug"]) == {"m2", "m3"}
    assert stats["dropped_rows"] == 1
    assert stats["dropped_stale_wallet_entries"] == 1


def test_split_entry_pipeline_signals_avoids_fillna_downcast_futurewarning():
    split_entry_pipeline_signals = _load_split_entry_pipeline_signals()
    signals_df = pd.DataFrame(
        [
            ["leaderboard_wallet", "OPEN_LONG", None, False, "m1"],
            ["leaderboard_wallet", None, "OPEN_LONG", True, "m2"],
        ],
        columns=[
            "signal_source",
            "entry_intent",
            "entry_intent",
            "source_wallet_fresh",
            "market_slug",
        ],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        filtered, stats = split_entry_pipeline_signals(signals_df)

    assert set(filtered["market_slug"]) == {"m2"}
    assert stats["dropped_rows"] == 1


def test_split_entry_pipeline_signals_drops_generic_analytics_only_rows():
    split_entry_pipeline_signals = _load_split_entry_pipeline_signals()
    signals_df = pd.DataFrame(
        [
            {"signal_source": "weather_temperature_wallet", "market_slug": "w1", "entry_intent": "OPEN_LONG", "analytics_only": True},
            {"signal_source": "weather_temperature_wallet", "market_slug": "w2", "entry_intent": "OPEN_LONG", "analytics_only": False},
            {"signal_source": "weather_temperature_wallet", "market_slug": "w3", "entry_intent": "CLOSE_LONG", "analytics_only": False},
        ]
    )

    filtered, stats = split_entry_pipeline_signals(signals_df)

    assert set(filtered["market_slug"]) == {"w2", "w3"}
    assert stats["dropped_rows"] == 1
    assert stats["dropped_analytics_only"] == 1
