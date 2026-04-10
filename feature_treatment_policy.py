"""Feature treatment policy and schema audit.

Every feature in the model catalog is assigned a *treatment* that declares
how it should be transformed before entering each model family:

Treatment kinds
---------------
- ``raw``         : no transform; already bounded or tree-models handle it.
- ``clip01``      : already clipped to [0, 1] during feature building.
- ``standardize`` : z-score (mean/std) before neural-net stages.
- ``log_scale``   : log1p before any scaling (heavy-tailed prices/counts).
- ``robust_scale``: median/IQR scaling (outlier-prone continuous values).
- ``boolean``     : binary 0/1 flag; no scaling needed.

Model scope
-----------
- ``tree``  : only used by tree-based models (Stage 1).
- ``nn``    : only used by neural-net models (Stage 2).
- ``all``   : used by every model family.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


# ── treatment descriptor ────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class FeatureTreatment:
    kind: str          # raw | clip01 | standardize | log_scale | robust_scale | boolean
    scope: str = "all" # all | tree | nn

VALID_KINDS = frozenset({"raw", "clip01", "standardize", "log_scale", "robust_scale", "boolean"})
VALID_SCOPES = frozenset({"all", "tree", "nn"})


# ── helpers for bulk assignment ─────────────────────────────────────

def _bulk(names: Sequence[str], kind: str, scope: str = "all") -> Dict[str, FeatureTreatment]:
    return {n: FeatureTreatment(kind=kind, scope=scope) for n in names}


# ── the policy registry ────────────────────────────────────────────

FEATURE_TREATMENT: Dict[str, FeatureTreatment] = {}

# ---- wallet_copy ----
FEATURE_TREATMENT.update(_bulk([
    "trader_win_rate",
    "normalized_trade_size",
    "whale_pressure",
], "clip01"))
FEATURE_TREATMENT.update(_bulk([
    "wallet_trade_count_30d",
], "log_scale"))
FEATURE_TREATMENT.update(_bulk([
    "wallet_alpha_30d",
    "wallet_avg_forward_return_15m",
    "wallet_signal_precision_tp",
], "standardize"))
FEATURE_TREATMENT["wallet_recent_streak"] = FeatureTreatment("raw")

# ---- market_microstructure ----
FEATURE_TREATMENT["current_price"] = FeatureTreatment("clip01")
FEATURE_TREATMENT["spread"] = FeatureTreatment("robust_scale")
FEATURE_TREATMENT.update(_bulk([
    "time_left",
    "liquidity_score",
    "volume_score",
    "probability_momentum",
    "volatility_score",
    "market_structure_score",
], "clip01"))

# ---- onchain_network ----
FEATURE_TREATMENT.update(_bulk([
    "btc_fee_pressure_score",
    "btc_mempool_congestion_score",
    "btc_network_activity_score",
    "btc_network_stress_score",
], "clip01"))

# ---- btc_spot_regime ----
FEATURE_TREATMENT["trend_score"] = FeatureTreatment("clip01")
FEATURE_TREATMENT.update(_bulk([
    "btc_spot_return_5m",
    "btc_spot_return_15m",
], "standardize"))
FEATURE_TREATMENT.update(_bulk([
    "btc_realized_vol_15m",
    "btc_volume_proxy",
], "robust_scale"))
FEATURE_TREATMENT["btc_atr_pct_15m"] = FeatureTreatment("robust_scale")
FEATURE_TREATMENT.update(_bulk([
    "btc_realized_vol_1h",
    "btc_realized_vol_4h",
], "robust_scale"))
FEATURE_TREATMENT["btc_volatility_regime_score"] = FeatureTreatment("clip01")
FEATURE_TREATMENT["btc_trend_persistence"] = FeatureTreatment("clip01")

# ---- btc_momentum_quality ----
FEATURE_TREATMENT["btc_rsi_14"] = FeatureTreatment("standardize")
FEATURE_TREATMENT["btc_rsi_distance_mid"] = FeatureTreatment("standardize")
FEATURE_TREATMENT["btc_rsi_divergence_score"] = FeatureTreatment("standardize")
FEATURE_TREATMENT.update(_bulk([
    "btc_macd",
    "btc_macd_signal",
    "btc_macd_hist",
    "btc_macd_hist_slope",
], "standardize"))
FEATURE_TREATMENT["btc_momentum_confluence"] = FeatureTreatment("standardize")

# ---- btc_live_index ----
FEATURE_TREATMENT.update(_bulk([
    "btc_live_price",
    "btc_live_spot_price",
    "btc_live_index_price",
    "btc_live_mark_price",
    "btc_live_price_kalman",
    "btc_live_spot_price_kalman",
    "btc_live_index_price_kalman",
    "btc_live_mark_price_kalman",
], "log_scale"))
FEATURE_TREATMENT["btc_live_funding_rate"] = FeatureTreatment("standardize")
FEATURE_TREATMENT["btc_live_source_quality_score"] = FeatureTreatment("clip01")
FEATURE_TREATMENT["btc_live_source_divergence_bps"] = FeatureTreatment("robust_scale")
FEATURE_TREATMENT.update(_bulk([
    "btc_live_spot_index_basis_bps",
    "btc_live_mark_index_basis_bps",
    "btc_live_mark_spot_basis_bps",
    "btc_live_spot_index_basis_bps_kalman",
    "btc_live_mark_index_basis_bps_kalman",
    "btc_live_mark_spot_basis_bps_kalman",
], "robust_scale"))
FEATURE_TREATMENT.update(_bulk([
    "btc_live_return_1m",
    "btc_live_return_5m",
    "btc_live_return_15m",
    "btc_live_return_1h",
    "btc_live_return_1m_kalman",
    "btc_live_return_5m_kalman",
    "btc_live_return_15m_kalman",
    "btc_live_return_1h_kalman",
], "standardize"))
FEATURE_TREATMENT["btc_live_volatility_proxy"] = FeatureTreatment("robust_scale")
FEATURE_TREATMENT["btc_live_confluence"] = FeatureTreatment("standardize")
FEATURE_TREATMENT["btc_live_confluence_kalman"] = FeatureTreatment("standardize")
FEATURE_TREATMENT["btc_live_index_ready"] = FeatureTreatment("boolean")
FEATURE_TREATMENT["btc_live_index_feed_available"] = FeatureTreatment("boolean")
FEATURE_TREATMENT["btc_live_mark_feed_available"] = FeatureTreatment("boolean")

# ---- btc_market_regime ----
FEATURE_TREATMENT.update(_bulk([
    "btc_market_regime_score",
    "btc_market_regime_trend_score",
    "btc_market_regime_volatility_score",
    "btc_market_regime_chaos_score",
    "btc_market_regime_stability_score",
], "clip01"))
FEATURE_TREATMENT.update(_bulk([
    "btc_market_regime_is_calm",
    "btc_market_regime_is_trend",
    "btc_market_regime_is_volatile",
    "btc_market_regime_is_chaotic",
], "boolean"))
FEATURE_TREATMENT["btc_market_regime_confidence_multiplier"] = FeatureTreatment("raw")
FEATURE_TREATMENT.update(_bulk([
    "btc_market_regime_weight_legacy",
    "btc_market_regime_weight_stage1",
    "btc_market_regime_weight_stage2",
], "raw"))

# ---- btc_sentiment ----
FEATURE_TREATMENT["sentiment_score"] = FeatureTreatment("standardize")
FEATURE_TREATMENT["btc_funding_rate"] = FeatureTreatment("standardize")
FEATURE_TREATMENT["is_overheated_long"] = FeatureTreatment("boolean")
FEATURE_TREATMENT["fgi_value"] = FeatureTreatment("standardize")
FEATURE_TREATMENT["fgi_normalized"] = FeatureTreatment("clip01")
FEATURE_TREATMENT.update(_bulk([
    "fgi_extreme_fear",
    "fgi_extreme_greed",
], "boolean"))
FEATURE_TREATMENT.update(_bulk([
    "fgi_contrarian",
    "fgi_momentum",
    "fgi_momentum_3d",
], "standardize"))
FEATURE_TREATMENT["gtrends_bitcoin"] = FeatureTreatment("standardize")
FEATURE_TREATMENT["gtrends_zscore"] = FeatureTreatment("raw")  # already a z-score
FEATURE_TREATMENT["gtrends_spike"] = FeatureTreatment("boolean")
FEATURE_TREATMENT["gtrends_momentum"] = FeatureTreatment("standardize")
FEATURE_TREATMENT.update(_bulk([
    "twitter_sentiment",
    "reddit_sentiment",
], "standardize"))
FEATURE_TREATMENT.update(_bulk([
    "twitter_post_count",
    "reddit_post_count",
], "log_scale"))
FEATURE_TREATMENT.update(_bulk([
    "twitter_sentiment_pos",
    "twitter_sentiment_neg",
    "reddit_sentiment_pos",
    "reddit_sentiment_neg",
], "standardize"))
FEATURE_TREATMENT.update(_bulk([
    "twitter_engagement_proxy",
], "log_scale"))
FEATURE_TREATMENT.update(_bulk([
    "twitter_sentiment_zscore",
    "reddit_sentiment_zscore",
], "raw"))  # already z-scores
FEATURE_TREATMENT.update(_bulk([
    "twitter_bullish",
    "twitter_bearish",
    "reddit_bullish",
    "reddit_bearish",
], "boolean"))
FEATURE_TREATMENT.update(_bulk([
    "twitter_sentiment_momentum",
    "reddit_sentiment_momentum",
], "standardize"))

# ---- portfolio_context ----
FEATURE_TREATMENT["open_positions_count"] = FeatureTreatment("raw")
FEATURE_TREATMENT.update(_bulk([
    "open_positions_negotiated_value_total",
    "open_positions_max_payout_total",
    "open_positions_current_value_total",
], "log_scale"))
FEATURE_TREATMENT["open_positions_unrealized_pnl_total"] = FeatureTreatment("standardize")
FEATURE_TREATMENT["open_positions_unrealized_pnl_pct_total"] = FeatureTreatment("standardize")
FEATURE_TREATMENT.update(_bulk([
    "open_positions_avg_to_now_price_change_pct_mean",
    "open_positions_avg_to_now_price_change_pct_min",
    "open_positions_avg_to_now_price_change_pct_max",
], "standardize"))
FEATURE_TREATMENT["open_positions_winner_count"] = FeatureTreatment("raw")
FEATURE_TREATMENT["open_positions_loser_count"] = FeatureTreatment("raw")

# ---- weather_wallet_copy ----
FEATURE_TREATMENT.update(_bulk([
    "wallet_temp_hit_rate_90d",
    "wallet_region_score",
    "wallet_temp_range_skill",
    "wallet_temp_threshold_skill",
    "wallet_quality_score",
    "wallet_state_confidence",
    "wallet_state_freshness_score",
    "wallet_size_change_score",
    "wallet_agreement_score",
], "clip01"))
FEATURE_TREATMENT["wallet_temp_realized_pnl_90d"] = FeatureTreatment("standardize")

# ---- weather_market_structure ----
# current_price, spread, time_left, liquidity_score, volume_score,
# market_structure_score already registered above from market_microstructure.
# Only execution_quality_score is new.
FEATURE_TREATMENT["execution_quality_score"] = FeatureTreatment("clip01")

# ---- weather_forecast_edge ----
FEATURE_TREATMENT.update(_bulk([
    "forecast_p_hit_interval",
    "weather_fair_probability_yes",
    "weather_fair_probability_side",
    "weather_market_probability",
], "clip01"))
FEATURE_TREATMENT.update(_bulk([
    "forecast_margin_to_lower_c",
    "forecast_margin_to_upper_c",
    "forecast_uncertainty_c",
    "forecast_drift_c",
], "standardize"))
FEATURE_TREATMENT.update(_bulk([
    "weather_forecast_edge",
    "weather_forecast_margin_score",
    "weather_forecast_stability_score",
], "standardize"))


# ── public query helpers ────────────────────────────────────────────

def get_treatment(feature: str) -> FeatureTreatment:
    """Return the treatment for *feature*, defaulting to ``raw``."""
    return FEATURE_TREATMENT.get(feature, FeatureTreatment("raw"))


def features_by_kind(kind: str, features: Sequence[str] | None = None) -> List[str]:
    """Return features matching a treatment *kind*.

    If *features* is given, filter within that list; otherwise scan the
    full policy registry.
    """
    pool = features if features is not None else list(FEATURE_TREATMENT)
    return [f for f in pool if get_treatment(f).kind == kind]


def features_for_scope(scope: str, features: Sequence[str] | None = None) -> List[str]:
    """Return features whose scope includes *scope* (``tree``, ``nn``, or ``all``)."""
    pool = features if features is not None else list(FEATURE_TREATMENT)
    return [f for f in pool if get_treatment(f).scope in (scope, "all")]


# ── schema audit ────────────────────────────────────────────────────

@dataclass
class SchemaAuditResult:
    ok: bool
    missing_treatment: List[str]   # in catalog but no treatment
    orphan_treatment: List[str]    # treatment exists but not in catalog
    invalid_kind: List[str]        # treatment.kind not in VALID_KINDS
    invalid_scope: List[str]       # treatment.scope not in VALID_SCOPES

    def summary(self) -> str:
        if self.ok:
            return "schema audit passed"
        parts = []
        if self.missing_treatment:
            parts.append(f"missing_treatment({len(self.missing_treatment)}): {', '.join(self.missing_treatment[:10])}")
        if self.orphan_treatment:
            parts.append(f"orphan_treatment({len(self.orphan_treatment)}): {', '.join(self.orphan_treatment[:10])}")
        if self.invalid_kind:
            parts.append(f"invalid_kind: {', '.join(self.invalid_kind[:10])}")
        if self.invalid_scope:
            parts.append(f"invalid_scope: {', '.join(self.invalid_scope[:10])}")
        return "; ".join(parts)


def audit_schema(catalog_features: Sequence[str] | None = None) -> SchemaAuditResult:
    """Compare the treatment registry against the model feature catalog.

    Parameters
    ----------
    catalog_features : list[str] | None
        Feature list to audit against.  When *None*, imports
        ``DEFAULT_TABULAR_FEATURE_COLUMNS`` from ``model_feature_catalog``.

    Returns
    -------
    SchemaAuditResult
    """
    if catalog_features is None:
        from model_feature_catalog import DEFAULT_TABULAR_FEATURE_COLUMNS
        catalog_features = DEFAULT_TABULAR_FEATURE_COLUMNS

    catalog_set: Set[str] = set(catalog_features)
    policy_set: Set[str] = set(FEATURE_TREATMENT)

    missing = sorted(catalog_set - policy_set)
    orphans = sorted(policy_set - catalog_set)
    bad_kind = sorted(f for f, t in FEATURE_TREATMENT.items() if t.kind not in VALID_KINDS)
    bad_scope = sorted(f for f, t in FEATURE_TREATMENT.items() if t.scope not in VALID_SCOPES)

    ok = not (missing or bad_kind or bad_scope)
    # orphans are acceptable (extra features like weather that may not be in
    # every catalog slice), so they don't fail the audit.

    if missing:
        logger.warning("Features without treatment policy: %s", ", ".join(missing))
    if orphans:
        logger.debug("Treatment entries not in catalog (likely weather/extended): %s", ", ".join(orphans))

    return SchemaAuditResult(
        ok=ok,
        missing_treatment=missing,
        orphan_treatment=orphans,
        invalid_kind=bad_kind,
        invalid_scope=bad_scope,
    )


def log_audit(catalog_features: Sequence[str] | None = None) -> SchemaAuditResult:
    """Run :func:`audit_schema` and log the result."""
    result = audit_schema(catalog_features)
    if result.ok:
        logger.info("Feature schema audit passed (%d features covered)", len(FEATURE_TREATMENT))
    else:
        logger.warning("Feature schema audit FAILED: %s", result.summary())
    return result


# ── CSV export ──────────────────────────────────────────────────────

def export_policy_csv(logs_dir: str = "logs") -> str:
    """Write ``logs/feature_treatment_policy.csv`` and return the path."""
    import pandas as pd
    from pathlib import Path

    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    out = logs_path / "feature_treatment_policy.csv"

    rows = []
    for feature, treatment in sorted(FEATURE_TREATMENT.items()):
        rows.append({
            "feature": feature,
            "treatment_kind": treatment.kind,
            "treatment_scope": treatment.scope,
        })
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Feature treatment policy exported to %s (%d features)", out, len(rows))
    return str(out)


# ── categorical label registry ──────────────────────────────────────
# Features that carry string/categorical labels (not numeric).
# These must be label-encoded or one-hot-encoded before model use.

CATEGORICAL_FEATURES: Dict[str, List[str]] = {
    "btc_market_regime_label": ["calm", "trend", "volatile", "chaotic"],
    "btc_volatility_regime": ["LOW", "NORMAL", "HIGH", "EXTREME"],
    "btc_momentum_regime": ["OVERSOLD_EXHAUSTION", "BEARISH", "NEUTRAL", "BULLISH", "OVERBOUGHT_EXHAUSTION"],
    "btc_trend_bias": ["BEARISH", "NEUTRAL", "BULLISH"],
    "btc_live_bias": ["BEARISH", "NEUTRAL", "BULLISH"],
    "alligator_alignment": ["BEARISH", "NEUTRAL", "BULLISH"],
    "market_structure": ["UNKNOWN", "BEARISH", "NEUTRAL", "BULLISH"],
    "onchain_network_health": ["UNKNOWN", "STRESSED", "NORMAL", "HEALTHY"],
    "market_family": ["crypto", "weather", "sports", "politics", "other"],
}
