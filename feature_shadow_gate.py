"""
feature_shadow_gate.py
──────────────────────
Hard rule: no new feature enters live use unless it has proven edge
offline (walk-forward evaluation) AND in shadow (live scoring without
real trades).

Usage
-----
From supervisor.py or real_pipeline.py, call:

    from feature_shadow_gate import FeatureShadowGate
    gate = FeatureShadowGate(logs_dir="logs")
    if gate.is_approved("btc_internal_trend_score"):
        # use it
    else:
        # silently drop or fill with prior

From the research pipeline, register a feature after it passes offline eval:

    gate.register_offline_approved("btc_internal_trend_score", metrics={
        "walk_forward_lift": 0.023,
        "brier_improvement": 0.004,
        "eval_rows": 1200,
    })

From the shadow runner, promote to live after shadow is clean:

    gate.promote_to_live("btc_internal_trend_score", shadow_rows=500)

Registry
--------
Stored in logs/feature_shadow_registry.csv — one row per feature, with:
  feature_name, status, offline_approved_at, shadow_approved_at,
  live_approved_at, offline_metrics_json, shadow_rows, notes
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Features that existed before this gate was introduced are grandfathered in.
# They do NOT need re-approval unless explicitly revoked.
_GRANDFATHERED_FEATURES: frozenset[str] = frozenset({
    # Wallet copy features
    "trader_win_rate", "wallet_trade_count_30d", "wallet_alpha_30d",
    "wallet_avg_forward_return_15m", "wallet_signal_precision_tp",
    "wallet_recent_streak", "normalized_trade_size", "whale_pressure",
    # Market microstructure
    "current_price", "spread", "time_left", "liquidity_score",
    "volume_score", "probability_momentum", "volatility_score",
    "market_structure_score",
    # On-chain
    "btc_fee_pressure_score", "btc_mempool_congestion_score",
    "btc_network_activity_score", "btc_network_stress_score",
    # BTC spot regime
    "btc_ret_5m", "btc_ret_15m", "btc_realized_vol_15m",
    "btc_realized_vol_1h", "btc_realized_vol_4h", "btc_trend_score",
    "btc_atr_pct", "btc_volume_proxy", "btc_volatility_regime_score",
    "btc_trend_persistence",
    # BTC momentum
    "btc_rsi_14", "btc_rsi_distance_mid", "btc_rsi_divergence_score",
    "btc_macd", "btc_macd_signal", "btc_macd_hist", "btc_macd_hist_slope",
    "btc_momentum_confluence",
    # BTC live index (core)
    "btc_live_price", "btc_spot_price", "btc_index_price", "btc_mark_price",
    "btc_funding_rate", "btc_basis", "btc_ret_1m", "btc_ret_1h",
    # Sentiment (pre-existing)
    "fgi_value", "twitter_sentiment", "reddit_sentiment",
    # BTC internal trend (added in session as Google Trends substitute)
    "btc_internal_ret_1h", "btc_internal_ret_4h", "btc_internal_vol_ratio",
    "btc_internal_price_vs_7d_ma", "btc_internal_trend_score", "btc_internal_trend_up",
    # Regime
    "active_regime", "btc_market_regime_weight_legacy",
    "btc_market_regime_weight_stage1", "btc_market_regime_weight_stage2",
    # Portfolio context
    "open_position_count", "portfolio_unrealized_pnl", "portfolio_exposure_usdc",
    # Model outputs used as features downstream
    "confidence", "edge_score", "expected_return", "calibrated_edge",
    "p_tp_before_sl", "hybrid_edge",
})

_REGISTRY_COLUMNS = [
    "feature_name", "status", "offline_approved_at", "shadow_approved_at",
    "live_approved_at", "offline_metrics_json", "shadow_rows", "notes",
]

_STATUS_ORDER = {"grandfathered": 0, "offline": 1, "shadow": 2, "live": 3, "revoked": -1}


class FeatureShadowGate:
    """
    Enforces the rule: no new feature enters live unless proven offline + shadow.

    Statuses
    --------
    grandfathered  Pre-existing feature; approved for live without re-evaluation.
    offline        Passed walk-forward offline evaluation.
    shadow         Passed shadow mode (scored live, not traded).
    live           Fully approved for live trading.
    revoked        Explicitly removed from approval; treated as unknown.
    unknown        Not in registry; blocked from live use by default.
    """

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.logs_dir / "feature_shadow_registry.csv"
        self._cache: dict[str, str] = {}  # feature_name → status
        self._load()

    # ── Public API ───────────────────────────────────────────────────────────

    def is_approved(self, feature_name: str) -> bool:
        """
        Returns True if the feature is approved for live use.
        Grandfathered, shadow-approved, and live-approved features all pass.
        Offline-only features are NOT approved for live yet.
        """
        status = self._status(feature_name)
        return status in ("grandfathered", "shadow", "live")

    def is_approved_for_training(self, feature_name: str) -> bool:
        """
        Returns True if the feature can be used in offline training.
        Offline-approved and above pass.
        """
        status = self._status(feature_name)
        return status in ("grandfathered", "offline", "shadow", "live")

    def filter_approved(self, feature_names: list[str]) -> list[str]:
        """Return only features approved for live use."""
        return [f for f in feature_names if self.is_approved(f)]

    def filter_approved_for_training(self, feature_names: list[str]) -> list[str]:
        """Return only features approved for offline training."""
        return [f for f in feature_names if self.is_approved_for_training(f)]

    def register_offline_approved(
        self,
        feature_name: str,
        metrics: dict[str, Any] | None = None,
        notes: str = "",
    ) -> None:
        """
        Call this after a feature passes walk-forward offline evaluation.
        It is still NOT approved for live — it needs shadow validation next.
        """
        self._upsert(
            feature_name,
            status="offline",
            offline_approved_at=_now_iso(),
            offline_metrics_json=json.dumps(metrics or {}),
            notes=notes,
        )
        logger.info(
            "Feature '%s' approved for offline training. Shadow validation required before live.",
            feature_name,
        )

    def promote_to_shadow(self, feature_name: str, notes: str = "") -> None:
        """
        Call this to allow the feature to be scored live (shadow mode).
        It is still NOT traded with — shadow results must be reviewed first.
        """
        current = self._status(feature_name)
        if current not in ("offline", "grandfathered"):
            logger.warning(
                "Cannot promote '%s' to shadow: current status is '%s'. Requires offline approval first.",
                feature_name, current,
            )
            return
        self._upsert(feature_name, status="shadow", shadow_approved_at=_now_iso(), notes=notes)
        logger.info("Feature '%s' promoted to shadow mode.", feature_name)

    def promote_to_live(
        self,
        feature_name: str,
        shadow_rows: int = 0,
        notes: str = "",
    ) -> None:
        """
        Call this after shadow validation passes.
        Only then is the feature approved for live trading.
        """
        current = self._status(feature_name)
        if current not in ("shadow", "grandfathered"):
            logger.warning(
                "Cannot promote '%s' to live: current status is '%s'. "
                "Requires shadow validation first.",
                feature_name, current,
            )
            return
        if shadow_rows < 100 and current != "grandfathered":
            logger.warning(
                "Feature '%s' has only %d shadow rows — minimum 100 required for live promotion.",
                feature_name, shadow_rows,
            )
            return
        self._upsert(
            feature_name,
            status="live",
            live_approved_at=_now_iso(),
            shadow_rows=shadow_rows,
            notes=notes,
        )
        logger.info(
            "Feature '%s' promoted to LIVE with %d shadow rows. Now approved for trading.",
            feature_name, shadow_rows,
        )

    def revoke(self, feature_name: str, reason: str = "") -> None:
        """Revoke a feature from all use. Use when a feature is found to be harmful."""
        self._upsert(feature_name, status="revoked", notes=f"REVOKED: {reason}")
        logger.warning("Feature '%s' REVOKED. Reason: %s", feature_name, reason)

    def status_report(self) -> pd.DataFrame:
        """Return the full registry as a DataFrame for inspection."""
        return self._load_df()

    def unknown_features(self, feature_names: list[str]) -> list[str]:
        """Return features that are not yet in the registry (unvetted)."""
        return [f for f in feature_names if self._status(f) == "unknown"]

    # ── Internal ─────────────────────────────────────────────────────────────

    def _status(self, feature_name: str) -> str:
        if feature_name in self._cache:
            return self._cache[feature_name]
        if feature_name in _GRANDFATHERED_FEATURES:
            self._cache[feature_name] = "grandfathered"
            return "grandfathered"
        return "unknown"

    def _load(self) -> None:
        df = self._load_df()
        for _, row in df.iterrows():
            name = str(row.get("feature_name") or "").strip()
            status = str(row.get("status") or "unknown").strip()
            if name:
                self._cache[name] = status
        # Ensure all grandfathered features are in cache
        for f in _GRANDFATHERED_FEATURES:
            if f not in self._cache:
                self._cache[f] = "grandfathered"

    def _load_df(self) -> pd.DataFrame:
        if not self._registry_path.exists():
            return pd.DataFrame(columns=_REGISTRY_COLUMNS)
        try:
            return pd.read_csv(self._registry_path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame(columns=_REGISTRY_COLUMNS)

    def _upsert(self, feature_name: str, **kwargs) -> None:
        df = self._load_df()
        status = kwargs.get("status", "unknown")

        if feature_name in df.get("feature_name", pd.Series(dtype=str)).values:
            for col, val in kwargs.items():
                if col in df.columns:
                    df.loc[df["feature_name"] == feature_name, col] = val
                else:
                    df[col] = None
                    df.loc[df["feature_name"] == feature_name, col] = val
        else:
            row = {col: None for col in _REGISTRY_COLUMNS}
            row["feature_name"] = feature_name
            row.update(kwargs)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        try:
            df.to_csv(self._registry_path, index=False)
        except Exception as exc:
            logger.warning("Failed to persist feature shadow registry: %s", exc)

        self._cache[feature_name] = status


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Singleton for import convenience ─────────────────────────────────────────
_default_gate: FeatureShadowGate | None = None


def get_feature_gate(logs_dir: str = "logs") -> FeatureShadowGate:
    """Return a cached FeatureShadowGate instance."""
    global _default_gate
    if _default_gate is None:
        _default_gate = FeatureShadowGate(logs_dir=logs_dir)
    return _default_gate


if __name__ == "__main__":
    gate = FeatureShadowGate(logs_dir="logs")

    # Test grandfathered
    assert gate.is_approved("current_price"), "grandfathered feature should be approved"
    assert gate.is_approved("btc_internal_trend_score"), "btc internal trend should be grandfathered"

    # Test unknown
    assert not gate.is_approved("some_new_experimental_feature"), "unknown feature should be blocked"

    # Test offline → shadow → live pipeline
    gate.register_offline_approved("test_new_feature_xyz", metrics={"lift": 0.01}, notes="test")
    assert gate.is_approved_for_training("test_new_feature_xyz"), "offline feature ok for training"
    assert not gate.is_approved("test_new_feature_xyz"), "offline feature NOT ok for live"

    gate.promote_to_shadow("test_new_feature_xyz")
    assert gate.is_approved("test_new_feature_xyz"), "shadow feature ok for live scoring"

    gate.promote_to_live("test_new_feature_xyz", shadow_rows=200)
    assert gate.is_approved("test_new_feature_xyz"), "live feature approved"

    gate.revoke("test_new_feature_xyz", reason="test cleanup")
    assert not gate.is_approved("test_new_feature_xyz"), "revoked feature blocked"

    print("FeatureShadowGate self-test PASSED.")
    print(f"  Grandfathered features: {len(_GRANDFATHERED_FEATURES)}")
    print(f"  Registry: {gate._registry_path}")
