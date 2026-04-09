import os

from trading_mode_preset import apply_preset


def test_apply_preset_keeps_explicit_governor_confidence_env(monkeypatch):
    monkeypatch.setenv("GOV_LEVEL1_MIN_ENTRY_CONFIDENCE", "0.15")
    monkeypatch.setenv("GOV_LEVEL2_MIN_ENTRY_CONFIDENCE", "0.10")

    apply_preset(2)

    assert os.getenv("GOV_LEVEL1_MIN_ENTRY_CONFIDENCE") == "0.15"
    assert os.getenv("GOV_LEVEL2_MIN_ENTRY_CONFIDENCE") == "0.10"


def test_apply_preset_mode_2_defaults_align_with_live_governor_thresholds(monkeypatch):
    monkeypatch.delenv("GOV_LEVEL1_MIN_ENTRY_CONFIDENCE", raising=False)
    monkeypatch.delenv("GOV_LEVEL2_MIN_ENTRY_CONFIDENCE", raising=False)

    apply_preset(2)

    assert os.getenv("GOV_LEVEL1_MIN_ENTRY_CONFIDENCE") == "0.15"
    assert os.getenv("GOV_LEVEL2_MIN_ENTRY_CONFIDENCE") == "0.10"
