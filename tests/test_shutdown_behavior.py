import ast
from pathlib import Path
from textwrap import dedent


def _load_shutdown_helpers():
    source_path = Path(__file__).resolve().parent.parent / "supervisor.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)

    segments = []
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_shutdown_requested":
                    segments.append(dedent(ast.get_source_segment(source, node)))
                    break
        if isinstance(node, ast.FunctionDef) and node.name in {"_request_shutdown", "_sleep_until_shutdown_or_timeout"}:
            segments.append(dedent(ast.get_source_segment(source, node)))

    class _FakeTime:
        def __init__(self):
            self.sleeps = []
            self.raise_keyboard_interrupt = False
            self.calls = 0

        def sleep(self, seconds):
            self.calls += 1
            self.sleeps.append(float(seconds))
            if self.raise_keyboard_interrupt and self.calls == 1:
                raise KeyboardInterrupt

    class _SignalEnum:
        def __init__(self, signum):
            self.name = f"SIG{signum}"

    class _FakeSignal:
        SIGINT = 2
        Signals = _SignalEnum

    class _FakeLogging:
        messages = []

        @classmethod
        def info(cls, message, *args):
            cls.messages.append(message % args if args else message)

    fake_time = _FakeTime()
    namespace = {
        "time": fake_time,
        "signal": _FakeSignal,
        "logging": _FakeLogging,
    }
    exec("\n\n".join(segments), namespace)
    return namespace, fake_time, _FakeLogging


def test_request_shutdown_sets_flag():
    namespace, _, fake_logging = _load_shutdown_helpers()

    namespace["_request_shutdown"](2, None)

    assert namespace["_shutdown_requested"] is True
    assert fake_logging.messages
    assert "shutting down" in fake_logging.messages[-1].lower()


def test_interruptible_sleep_chunks_until_timeout():
    namespace, fake_time, _ = _load_shutdown_helpers()

    result = namespace["_sleep_until_shutdown_or_timeout"](0.6, step_seconds=0.25)

    assert result is False
    assert len(fake_time.sleeps) == 3
    assert round(sum(fake_time.sleeps), 2) == 0.6


def test_interruptible_sleep_turns_keyboard_interrupt_into_shutdown():
    namespace, fake_time, _ = _load_shutdown_helpers()
    fake_time.raise_keyboard_interrupt = True

    result = namespace["_sleep_until_shutdown_or_timeout"](5.0, step_seconds=1.0)

    assert result is True
    assert namespace["_shutdown_requested"] is True
    assert len(fake_time.sleeps) == 1
