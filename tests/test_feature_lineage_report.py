import tempfile
from pathlib import Path

from feature_lineage_report import build_lineage_report, write_lineage_report


class TestFeatureLineageReport:
    def test_build_report_not_empty(self):
        df = build_lineage_report()
        assert not df.empty
        assert "feature" in df.columns
        assert "source" in df.columns
        assert "runtime_use" in df.columns
        assert "training_use" in df.columns
        assert "target_use" in df.columns
        assert "treatment_kind" in df.columns
        assert "family" in df.columns

    def test_all_catalog_features_present(self):
        from model_feature_catalog import DEFAULT_TABULAR_FEATURE_COLUMNS
        df = build_lineage_report()
        report_features = set(df["feature"])
        for f in DEFAULT_TABULAR_FEATURE_COLUMNS:
            assert f in report_features, f"Feature {f} missing from lineage report"

    def test_write_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_lineage_report(logs_dir=tmpdir)
            assert path.exists()

    def test_treatment_kinds_match_policy(self):
        from feature_treatment_policy import get_treatment
        df = build_lineage_report()
        for _, row in df.iterrows():
            t = get_treatment(row["feature"])
            assert row["treatment_kind"] == t.kind
