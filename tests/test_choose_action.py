import ast
import logging
import os
from pathlib import Path
from textwrap import dedent

import numpy as np


def _load_choose_action():
    source_path = Path(__file__).resolve().parent.parent / "supervisor.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    func_node = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "choose_action")
    func_source = dedent(ast.get_source_segment(source, func_node))

    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default

    class _TradingConfig:
        ENTRY_INACTIVITY_EXPECTED_RETURN_FLOOR = -0.002
        ENTRY_INACTIVITY_CONFIDENCE_BOOST = 0.08

    class _PredictionLayer:
        @staticmethod
        def select_signal_score(row):
            return float(row.get("confidence", 0.0) or 0.0)

    namespace = {
        "os": os,
        "np": np,
        "logging": logging,
        "EntryRuleLayer": object,
        "TradingConfig": _TradingConfig,
        "PredictionLayer": _PredictionLayer,
        "prepare_observation": lambda feature_row, legacy=False: feature_row,
        "_safe_float": _safe_float,
    }
    exec(func_source, namespace)
    return namespace["choose_action"]


class _RuleShouldNotBeCalled:
    def __init__(self):
        self.should_enter_calls = 0

    def should_enter(self, row):
        self.should_enter_calls += 1
        raise AssertionError("should_enter should not be called when rule_allows_entry is already known")


class _HoldBrain:
    def predict(self, signal_row):
        return 0


def test_choose_action_force_candidate_uses_precomputed_rule_result():
    choose_action = _load_choose_action()
    entry_rule = _RuleShouldNotBeCalled()

    action = choose_action(
        {"force_candidate": True, "edge_score": 0.01},
        entry_rule,
        precomputed_rule_eval={"allow": True},
        precomputed_rule_allows=True,
    )

    assert action == 1
    assert entry_rule.should_enter_calls == 0


def test_choose_action_rl_hold_fallback_does_not_requery_rule():
    choose_action = _load_choose_action()
    entry_rule = _RuleShouldNotBeCalled()

    action = choose_action(
        {"confidence": 0.25, "expected_return": 0.01, "edge_score": 0.05},
        entry_rule,
        entry_brain=_HoldBrain(),
        precomputed_rule_eval={"allow": True},
        precomputed_rule_allows=True,
    )

    assert action == 2
    assert entry_rule.should_enter_calls == 0


def test_choose_action_no_rl_fallback_uses_precomputed_rule_result():
    choose_action = _load_choose_action()
    entry_rule = _RuleShouldNotBeCalled()

    action = choose_action(
        {"edge_score": 0.05},
        entry_rule,
        precomputed_rule_eval={"allow": True},
        precomputed_rule_allows=True,
    )

    assert action == 2
    assert entry_rule.should_enter_calls == 0


def test_choose_action_weather_market_bypasses_rl_and_uses_rules_only():
    choose_action = _load_choose_action()
    entry_rule = _RuleShouldNotBeCalled()

    action = choose_action(
        {
            "market_family": "weather_temperature_threshold",
            "entry_intent": "OPEN_LONG",
            "confidence": 0.99,
            "edge_score": 0.99,
        },
        entry_rule,
        entry_brain=_HoldBrain(),
        precomputed_rule_eval={"allow": True},
        precomputed_rule_allows=True,
    )

    assert action == 1
    assert entry_rule.should_enter_calls == 0
