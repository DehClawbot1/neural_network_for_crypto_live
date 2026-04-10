import pytest

from feature_treatment_policy import (
    FEATURE_TREATMENT,
    VALID_KINDS,
    VALID_SCOPES,
    FeatureTreatment,
    audit_schema,
    features_by_kind,
    features_for_scope,
    get_treatment,
)
from model_feature_catalog import DEFAULT_TABULAR_FEATURE_COLUMNS


class TestFeatureTreatment:
    def test_all_kinds_valid(self):
        for name, t in FEATURE_TREATMENT.items():
            assert t.kind in VALID_KINDS, f"{name} has invalid kind={t.kind}"

    def test_all_scopes_valid(self):
        for name, t in FEATURE_TREATMENT.items():
            assert t.scope in VALID_SCOPES, f"{name} has invalid scope={t.scope}"

    def test_get_treatment_known(self):
        t = get_treatment("trader_win_rate")
        assert t.kind == "clip01"

    def test_get_treatment_unknown_defaults_raw(self):
        t = get_treatment("__nonexistent_feature__")
        assert t.kind == "raw"
        assert t.scope == "all"

    def test_features_by_kind(self):
        bools = features_by_kind("boolean")
        assert "btc_market_regime_is_calm" in bools
        assert "trader_win_rate" not in bools

    def test_features_by_kind_with_subset(self):
        subset = ["trader_win_rate", "btc_market_regime_is_calm", "btc_rsi_14"]
        clip = features_by_kind("clip01", subset)
        assert clip == ["trader_win_rate"]

    def test_features_for_scope_all(self):
        # With default scope="all", all features should be included since
        # current policy has no nn-only or tree-only features.
        result = features_for_scope("all", ["trader_win_rate", "btc_rsi_14"])
        assert "trader_win_rate" in result
        assert "btc_rsi_14" in result

    def test_features_for_scope_tree(self):
        result = features_for_scope("tree", ["trader_win_rate"])
        assert "trader_win_rate" in result


class TestSchemaAudit:
    def test_audit_passes_for_catalog(self):
        result = audit_schema()
        assert result.ok, result.summary()
        assert result.missing_treatment == []
        assert result.invalid_kind == []
        assert result.invalid_scope == []

    def test_audit_detects_missing_treatment(self):
        result = audit_schema(["__fantasy_feature__"])
        assert not result.ok
        assert "__fantasy_feature__" in result.missing_treatment

    def test_audit_reports_orphans(self):
        # Passing a small subset means most policy entries become orphans
        result = audit_schema(["trader_win_rate"])
        assert len(result.orphan_treatment) > 0
        # orphans alone do NOT cause ok=False
        assert result.ok

    def test_catalog_fully_covered(self):
        """Every feature in the default catalog must have a treatment."""
        missing = set(DEFAULT_TABULAR_FEATURE_COLUMNS) - set(FEATURE_TREATMENT)
        assert missing == set(), f"Features without treatment: {missing}"
