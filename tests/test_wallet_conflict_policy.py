import ast
from pathlib import Path
from textwrap import dedent


def _load_should_soften_wallet_state_conflict():
    source_path = Path(__file__).resolve().parent.parent / "supervisor.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    func_node = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "should_soften_wallet_state_conflict"
    )
    func_source = dedent(ast.get_source_segment(source, func_node))
    namespace = {}
    exec(func_source, namespace)
    return namespace["should_soften_wallet_state_conflict"]


def test_softens_only_scale_in_conflict_as_sole_reason():
    fn = _load_should_soften_wallet_state_conflict()
    row = {
        "entry_intent": "OPEN_LONG",
        "wallet_conflict_with_stronger": True,
        "source_wallet_position_event": "SCALE_IN",
        "wallet_state_gate_reason": "conflict_with_stronger_wallet",
    }
    assert fn(row) is True


def test_does_not_soften_new_entry_conflict():
    fn = _load_should_soften_wallet_state_conflict()
    row = {
        "entry_intent": "OPEN_LONG",
        "wallet_conflict_with_stronger": True,
        "source_wallet_position_event": "NEW_ENTRY",
        "wallet_state_gate_reason": "conflict_with_stronger_wallet",
    }
    assert fn(row) is False


def test_does_not_soften_when_other_wallet_failures_are_present():
    fn = _load_should_soften_wallet_state_conflict()
    row = {
        "entry_intent": "OPEN_LONG",
        "wallet_conflict_with_stronger": True,
        "source_wallet_position_event": "SCALE_IN",
        "wallet_state_gate_reason": "wallet_state_stale,conflict_with_stronger_wallet",
    }
    assert fn(row) is False
