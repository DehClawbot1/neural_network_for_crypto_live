import tempfile
from pathlib import Path

from model_registry import ModelRegistry
from model_registry_runtime import candidate_beats_champion, register_and_promote_rows


def test_candidate_beats_champion_for_higher_accuracy():
    assert candidate_beats_champion({"accuracy": 0.71}, {"accuracy": 0.63}) is True
    assert candidate_beats_champion({"accuracy": 0.51}, {"accuracy": 0.63}) is False


def test_register_and_promote_rows_marks_winners_without_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        logs_dir = Path(tmp) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        registry = ModelRegistry(logs_dir=str(logs_dir))

        rows = [
            {
                "run_id": "run-1",
                "model_kind": "lda",
                "artifact_group": "lda",
                "feature_set": "curated_standardized",
                "scaling": "standard",
                "regularization": "none",
                "market_family": "btc",
                "regime_slice": "all",
                "n_test_rows": 12,
                "accuracy": 0.63,
                "promotion_status": "evaluation_only",
            }
        ]

        registered = register_and_promote_rows(registry=registry, candidate_rows=rows)

        assert len(registered.index) == 1
        assert registered.iloc[0]["promotion_status"] == "evaluation_only"
        assert (logs_dir / "model_registry_comparison.csv").exists()
        assert (logs_dir / "decision_profit_audit.md").exists()
