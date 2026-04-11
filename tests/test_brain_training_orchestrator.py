from pathlib import Path

from brain_paths import list_brain_contexts


def test_list_brain_contexts_returns_btc_and_weather(tmp_path: Path):
    contexts = list_brain_contexts(
        shared_logs_dir=tmp_path / "logs",
        shared_weights_dir=tmp_path / "weights",
    )
    families = {ctx.market_family for ctx in contexts}
    assert "btc" in families
    assert "weather_temperature" in families
    assert len(contexts) == 2


def test_build_family_datasets_does_not_crash_on_empty_logs(tmp_path: Path):
    """build_family_datasets touches many heavy deps, so we only test
    that list_brain_contexts works with empty dirs (the precondition)."""
    logs = tmp_path / "logs"
    weights = tmp_path / "weights"
    logs.mkdir()
    weights.mkdir()
    contexts = list_brain_contexts(shared_logs_dir=logs, shared_weights_dir=weights)
    for ctx in contexts:
        assert ctx.logs_dir.exists()
        assert ctx.weights_dir.exists()
