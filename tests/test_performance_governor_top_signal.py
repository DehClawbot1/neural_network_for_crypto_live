import ast
from pathlib import Path
from textwrap import dedent


def _load_helpers():
    source_path = Path(__file__).resolve().parent.parent / "supervisor.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    wanted = {
        "_safe_int",
        "performance_governor_top_signal_decision",
        "performance_governor_consume_top_signal_slot",
    }
    func_nodes = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    func_source = "\n\n".join(dedent(ast.get_source_segment(source, node)) for node in func_nodes)
    import numpy as np
    namespace = {"np": np}
    exec(func_source, namespace)
    return (
        namespace["performance_governor_top_signal_decision"],
        namespace["performance_governor_consume_top_signal_slot"],
    )


def test_top_signal_slot_is_not_consumed_until_entry_opens():
    allow, consume = _load_helpers()
    governor_state = {"top_signal_only": True}

    assert allow(governor_state, consumed_count=0) is True
    assert allow(governor_state, consumed_count=0) is True

    consumed_count = consume(governor_state, consumed_count=0)
    assert consumed_count == 1
    assert allow(governor_state, consumed_count=consumed_count) is False


def test_top_signal_helpers_are_noops_when_disabled():
    allow, consume = _load_helpers()
    governor_state = {"top_signal_only": False}

    assert allow(governor_state, consumed_count=5) is True
    assert consume(governor_state, consumed_count=5) == 5
