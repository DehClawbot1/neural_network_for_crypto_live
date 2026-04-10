import tempfile
from pathlib import Path

from model_registry import ModelRegistry


class TestModelRegistry:
    def setup_method(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.logs_dir = Path(self.test_dir.name)

    def teardown_method(self):
        self.test_dir.cleanup()

    def test_register_and_read(self):
        registry = ModelRegistry(logs_dir=str(self.logs_dir))
        registry.register(
            model_kind="logistic_l1",
            regularization="l1",
            nonzero_feature_count=42,
            accuracy=0.72,
        )
        table = registry.comparison_table()
        assert len(table) == 1
        assert table.iloc[0]["model_kind"] == "logistic_l1"
        assert table.iloc[0]["accuracy"] == 0.72

    def test_append_only(self):
        registry = ModelRegistry(logs_dir=str(self.logs_dir))
        registry.register(model_kind="lda", accuracy=0.6)
        registry.register(model_kind="gaussian_nb", accuracy=0.55)
        table = registry.comparison_table()
        assert len(table) == 2

    def test_decision_profit_audit_md(self):
        registry = ModelRegistry(logs_dir=str(self.logs_dir))
        registry.register(model_kind="lda", accuracy=0.65)
        path = registry.write_decision_profit_audit()
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Decision / Profit Audit" in content
