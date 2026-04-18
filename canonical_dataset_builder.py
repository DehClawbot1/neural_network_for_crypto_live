"""Canonical learning dataset builder.

Produces ONE row per closed trade, joining every data domain needed for
supervised retraining.  This is the single source of truth for all model
training — do NOT train from raw source files directly.

Join pipeline (spine → enrichment layers):
  Spine    closed_positions.csv          — trade outcomes, fill mechanics, P&L
  Layer 1  btc_trade_attribution.csv     — BTC forecast attribution (exact position_id join)
  Layer 2  candidate_decisions.csv       — supervisor gate + wallet context (merge_asof)
  Layer 3  historical_dataset.csv        — signal-time regime / sentiment features (merge_asof)
  Layer 4  btc_forecast_eval.csv         — forecast accuracy feedback at trade open time (merge_asof)
  Layer 5  entry_signal_snapshot_json    — exploded live snapshot features (JSON → columns)

Output schema sections (preserved in column order):
  A  Identity          — row keys across all sources
  B  Market identity   — what market / family / brain
  C  Timestamps        — signal time, fill time, close time
  D  Microstructure    — spread, depth, liquidity, implied prob at signal time
  E  BTC regime        — live price, volatility, trend, momentum, sentiment (BTC only)
  F  Weather forecast  — probability, edge, uncertainty (weather only)
  G  Wallet / portfolio context
  H  Order decision    — supervisor output: gates, confidence, edge, size
  I  Fill result       — actual entry price, slippage, execution path
  J  Close result      — exit price, PnL, hold time, close reason
  K  Path attribution  — MFE/MAE, forecast accuracy (BTC), resolution (weather)
  L  Derived labels    — computed from realized outcomes (never from heuristics)
  M  Data quality      — artifact flags, eligibility, exclusion reason

Outputs:
  logs/btc/canonical_dataset.csv
  logs/weather_temperature/canonical_dataset.csv
  logs/canonical_dataset.csv  (union, backward compat only — do not train from this)

Family guarantee:
  BTC columns (E, path labels in K) are ONLY written to btc/ output.
  Weather columns (F, resolution labels in K) are ONLY written to weather/ output.
  The combined file carries both column sets but has NaN in the wrong-family slots.

Run:
  python canonical_dataset_builder.py [--logs-dir logs] [--validate]
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return pd.DataFrame()


def _parse_ts(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except Exception:
        return pd.to_datetime(series, utc=True, errors="coerce")


def _truthy(v) -> bool:
    if pd.isna(v):
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _coalesce(df: pd.DataFrame, *cols: str) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype=object)
    for c in cols:
        if c in df.columns:
            result = result.fillna(df[c])
    return result


def _merge_asof_safe(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    by: list[str] | None,
    tolerance: pd.Timedelta,
    suffixes: tuple[str, str] = ("", "_r"),
) -> pd.DataFrame:
    if left.empty or right.empty:
        return left
    if on not in left.columns or on not in right.columns:
        return left
    lc = left.copy()
    rc = right.copy()
    lc[on] = _parse_ts(lc[on]) if not pd.api.types.is_datetime64_any_dtype(lc[on]) else lc[on]
    rc[on] = _parse_ts(rc[on]) if not pd.api.types.is_datetime64_any_dtype(rc[on]) else rc[on]
    valid_by = [c for c in (by or []) if c in lc.columns and c in rc.columns]
    lc_valid = lc[lc[on].notna()].sort_values(on)
    rc_valid = rc[rc[on].notna()].sort_values(on)
    if lc_valid.empty or rc_valid.empty:
        return left
    merged = pd.merge_asof(
        lc_valid, rc_valid,
        on=on, by=valid_by or None,
        direction="backward",
        tolerance=tolerance,
        suffixes=suffixes,
    )
    if lc[on].isna().any():
        merged = pd.concat([merged, lc[lc[on].isna()]], ignore_index=True, sort=False)
    return merged


def _explode_json_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    def _parse(v):
        try:
            p = json.loads(str(v or "").strip())
            return p if isinstance(p, dict) else {}
        except Exception:
            return {}
    flat = pd.json_normalize(df[col].apply(_parse))
    if flat.empty:
        return df
    new_cols = [c for c in flat.columns if c not in df.columns]
    if not new_cols:
        return df
    return pd.concat([df.reset_index(drop=True), flat[new_cols].reset_index(drop=True)], axis=1)


def _coalesce_cols(df: pd.DataFrame, target: str, *candidates: str) -> pd.DataFrame:
    """Fill target column from candidates in order, then drop candidate duplicates."""
    if target not in df.columns:
        df[target] = pd.NA
    for c in candidates:
        if c in df.columns and c != target:
            df[target] = df[target].fillna(df[c])
    return df


# ── section column manifests ──────────────────────────────────────────────────
# These define what ends up in the final output — order is preserved.

SECTION_A_IDENTITY = [
    "candidate_id", "cycle_id", "position_id", "token_id",
    "condition_id", "outcome_side",
]

SECTION_B_MARKET = [
    "market", "market_title", "market_slug", "market_family", "brain_id",
]

SECTION_C_TIMESTAMPS = [
    "signal_timestamp",   # when supervisor generated the signal
    "fill_timestamp",     # when the entry order was filled (opened_at)
    "close_timestamp",    # when the position was closed (closed_at)
    "hold_time_minutes",
]

SECTION_D_MICROSTRUCTURE = [
    # Market state at signal time
    "market_implied_prob",    # CLOB mid-price = implied YES probability
    "spread_at_signal",       # bid-ask spread at entry
    "best_bid",
    "best_ask",
    "liquidity_score",
    "volume_score",
    "probability_momentum",
    "whale_pressure",
    "market_structure_score",
    "time_left",              # time to contract expiry
]

# BTC-ONLY: signal-time market regime and price features
SECTION_E_BTC_REGIME = [
    "btc_live_price",
    "btc_live_spot_price",
    "btc_live_funding_rate",
    "btc_live_return_1m",
    "btc_live_return_5m",
    "btc_live_return_15m",
    "btc_live_return_1h",
    "btc_live_confluence",
    "btc_live_source_quality_score",
    "btc_spot_return_5m",
    "btc_spot_return_15m",
    "btc_realized_vol_15m",
    "btc_realized_vol_1h",
    "btc_realized_vol_4h",
    "btc_volatility_regime",
    "btc_volatility_regime_score",
    "btc_trend_persistence",
    "btc_market_regime_label",
    "btc_market_regime_score",
    "btc_market_regime_trend_score",
    "btc_market_regime_volatility_score",
    "btc_market_regime_chaos_score",
    "btc_market_regime_stability_score",
    "btc_market_regime_is_calm",
    "btc_market_regime_is_trend",
    "btc_market_regime_is_volatile",
    "btc_market_regime_is_chaotic",
    "btc_market_regime_confidence_multiplier",
    "btc_momentum_confluence",
    "btc_rsi_14",
    "btc_rsi_distance_mid",
    "btc_macd",
    "btc_macd_hist",
    "btc_macd_hist_slope",
    "btc_atr_pct_15m",
    "fgi_value",
    "fgi_normalized",
    "fgi_contrarian",
    "fgi_momentum",
    "sentiment_score",
    "btc_funding_rate",
    "is_overheated_long",
]

# WEATHER-ONLY: signal-time forecast features
SECTION_F_WEATHER_FORECAST = [
    "weather_fair_probability_yes",
    "weather_fair_probability_side",
    "weather_market_probability",
    "weather_forecast_edge",
    "weather_forecast_margin_score",
    "weather_forecast_stability_score",
    "forecast_p_hit_interval",
    "forecast_margin_to_lower_c",
    "forecast_margin_to_upper_c",
    "forecast_uncertainty_c",
    "forecast_drift_c",
    "weather_parseable",
    "weather_location",
    "weather_event_date_local",
    "weather_question_type",
    "weather_temp_unit",
    "weather_lower_c",
    "weather_upper_c",
]

SECTION_G_WALLET = [
    "trader_wallet",
    "wallet_trade_count_30d",
    "wallet_alpha_30d",
    "wallet_avg_forward_return_15m",
    "wallet_signal_precision_tp",
    "wallet_recent_streak",
    "wallet_quality_score",
    "open_positions_count",
    "open_positions_unrealized_pnl_total",
    "open_positions_unrealized_pnl_pct_total",
    "open_positions_winner_count",
    "open_positions_loser_count",
    "performance_governor_level",
    "available_balance",
]

SECTION_H_ORDER_DECISION = [
    # Supervisor output at decision time
    "signal_label",
    "entry_intent",
    "model_action",
    "final_decision",
    "gate",
    "reject_reason",
    "reject_category",
    "entry_model_family",
    "entry_model_version",
    "active_model_group",
    "active_model_kind",
    "active_regime",
    "technical_regime_bucket",
    "horizon_bucket",
    "liquidity_bucket",
    "volatility_bucket",
    "confidence",
    "edge_score",
    "calibrated_edge",
    "calibrated_baseline",
    "expected_return",
    "p_tp_before_sl",
    "proposed_size_usdc",
    "final_size_usdc",
    "order_id",
]

SECTION_I_FILL = [
    # What actually happened at fill time
    "entry_price",
    "shares",
    "size_usdc",
    "actual_execution_path",
    "exit_fill_latency_seconds",
    "exit_partial_fill_ratio",
    "exit_realized_slippage_bps",
    "entry_signal_snapshot_feature_count",
    "entry_signal_snapshot_version",
]

SECTION_J_CLOSE = [
    "exit_price",
    "close_reason",
    "exit_reason_family",
    "intended_exit_reason",
    "realized_pnl",
    "net_realized_pnl",
    "price_return",
    "drawdown_from_peak",
    "runup_from_entry",
    "recent_return_3",
    "reconciliation_close_flag",
    "operational_close_flag",
    "is_reconciliation_close",
]

# BTC-ONLY: post-trade path and forecast attribution
SECTION_K_BTC_ATTRIBUTION = [
    "entry_btc_price",
    "exit_btc_price",
    "entry_btc_predicted_direction",
    "entry_btc_predicted_return",
    "entry_btc_forecast_confidence",
    "entry_btc_mtf_agreement",
    "btc_predicted_direction",     # from btc_trade_attribution (may be more complete)
    "btc_predicted_return",
    "btc_forecast_confidence",
    "btc_mtf_agreement",
    "max_adverse_excursion_pct",   # MAE during hold
    "max_favorable_excursion_pct", # MFE during hold
    "max_drawdown_from_peak_pct",
    # Forecast eval feedback joined at trade open time
    "forecast_eval_predicted_direction",
    "forecast_eval_predicted_return",
    "forecast_eval_confidence",
    "forecast_eval_actual_return",
    "forecast_eval_correct",
    "forecast_eval_mtf_agreement",
]

# WEATHER-ONLY: resolution attribution
SECTION_K_WEATHER_ATTRIBUTION = [
    "weather_contract_resolved_yes",  # 1 = YES resolved, 0 = NO (label)
    "weather_forecast_confirms_direction",
    "weather_entry_gate_fail",
    "weather_entry_allowed_by_forecast",
]

SECTION_L_DERIVED_LABELS = [
    # Computed from realized outcomes — never heuristic proxies
    "trade_outcome_label",     # win / loss / flat  (from realized_pnl sign)
    "entry_edge_realized",     # realized_pnl / size_usdc
    "entry_quality",           # 1 if entry_edge_realized > 0
    "exit_regret",             # BTC only: mfe - price_return (missed upside, return space)
    "exit_quality",            # BTC only: 1 if exit_regret <= 0.01
    "slippage_error",          # actual_slippage - predicted_slippage
    "execution_quality",       # 1 if slippage_error <= 0
    "strategy_quality",        # 1 if strategy group mean edge > 0
]

SECTION_M_QUALITY = [
    "learning_eligible",
    "learning_exclusion_reason",
    "contaminated_learning_row",
    "reconciliation_artifact_flag",
    "operational_close_flag",
    "dead_orderbook_artifact_flag",
    "stale_price_artifact_flag",
    "malformed_fill_flag",
    "missing_execution_path_flag",
    "actual_execution_path",   # kept here for audit trail
]

_ALL_SECTIONS = (
    SECTION_A_IDENTITY
    + SECTION_B_MARKET
    + SECTION_C_TIMESTAMPS
    + SECTION_D_MICROSTRUCTURE
    + SECTION_E_BTC_REGIME
    + SECTION_F_WEATHER_FORECAST
    + SECTION_G_WALLET
    + SECTION_H_ORDER_DECISION
    + SECTION_I_FILL
    + SECTION_J_CLOSE
    + SECTION_K_BTC_ATTRIBUTION
    + SECTION_K_WEATHER_ATTRIBUTION
    + SECTION_L_DERIVED_LABELS
    + SECTION_M_QUALITY
)

_LIVE_FILL_REASONS = {
    "take_profit_roi", "stop_loss", "hard_emergency_stop",
    "trajectory_panic_exit", "trajectory_liquidity_stress",
    "rl_exit", "time_stop", "policy_exit",
}

# ── builder ───────────────────────────────────────────────────────────────────

class CanonicalDatasetBuilder:
    """Build the canonical one-row-per-trade learning dataset."""

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.btc_out = self.logs_dir / "btc" / "canonical_dataset.csv"
        self.weather_out = self.logs_dir / "weather_temperature" / "canonical_dataset.csv"
        self.combined_out = self.logs_dir / "canonical_dataset.csv"

    # ── source loaders ────────────────────────────────────────────────────────

    def _load_spine(self) -> pd.DataFrame:
        """closed_positions is the authoritative trade record."""
        df = _safe_read(self.logs_dir / "closed_positions.csv")
        if df.empty:
            logger.warning("closed_positions.csv missing — no spine")
            return df
        df["_spine_ts"] = _parse_ts(_coalesce(df, "opened_at", "closed_at"))
        df["fill_timestamp"] = _parse_ts(_coalesce(df, "opened_at"))
        df["close_timestamp"] = _parse_ts(_coalesce(df, "closed_at"))
        df["signal_timestamp"] = df["fill_timestamp"]  # best proxy until decisions joined
        logger.info("Spine: %d closed trades", len(df))
        return df

    def _enrich_btc_attribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Layer 1: btc_trade_attribution — exact join on position_id."""
        attr = _safe_read(self.logs_dir / "btc_trade_attribution.csv")
        if attr.empty or "position_id" not in df.columns or "position_id" not in attr.columns:
            return df
        # Rename attribution columns that overlap with spine to avoid silent clobber
        rename = {
            "btc_predicted_direction": "btc_predicted_direction",
            "btc_predicted_return": "btc_predicted_return",
            "btc_forecast_confidence": "btc_forecast_confidence",
            "btc_mtf_agreement": "btc_mtf_agreement",
            "edge_score_at_entry": "edge_score_at_entry",
            "expected_return_at_entry": "expected_return_at_entry",
            "decision_score": "decision_score",
        }
        keep = ["position_id"] + [c for c in attr.columns if c not in df.columns or c in rename]
        attr_slim = attr[[c for c in keep if c in attr.columns]].drop_duplicates(
            subset=["position_id"], keep="last"
        )
        df = df.merge(attr_slim, on="position_id", how="left", suffixes=("", "_attr"))
        # Coalesce attribution columns into spine
        for src, dst in rename.items():
            if f"{src}_attr" in df.columns:
                df[dst] = df.get(dst, pd.Series(pd.NA, index=df.index)).fillna(df[f"{src}_attr"])
                df.drop(columns=[f"{src}_attr"], inplace=True, errors="ignore")
        logger.info("After BTC attribution join: %d rows", len(df))
        return df

    def _enrich_decisions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Layer 2: candidate_decisions — merge_asof on timestamp by [token_id, outcome_side].

        Brings in: cycle_id, candidate_id, trader_wallet, calibrated_edge,
        calibrated_baseline, reject_reason, reject_category, order_id.
        """
        decisions = _safe_read(self.logs_dir / "candidate_decisions.csv")
        if decisions.empty:
            return df
        # Only accepted entries match to a trade
        accepted = decisions[
            decisions.get("final_decision", pd.Series("", dtype=str))
            .astype(str).isin(["ENTRY_FILLED", "PAPER_OPENED"])
        ].copy()
        if accepted.empty:
            return df
        accepted["_dec_ts"] = _parse_ts(accepted.get("created_at", pd.Series(dtype=str)))
        keep_cols = [
            "token_id", "outcome_side", "_dec_ts",
            "cycle_id", "candidate_id", "trader_wallet",
            "calibrated_edge", "calibrated_baseline",
            "reject_reason", "reject_category", "order_id",
            "proposed_size_usdc", "available_balance",
        ]
        accepted_slim = accepted[[c for c in keep_cols if c in accepted.columns]].copy()
        accepted_slim = accepted_slim[accepted_slim["_dec_ts"].notna()].sort_values("_dec_ts")
        new_cols = [c for c in accepted_slim.columns if c not in df.columns and c != "_dec_ts"]
        if not new_cols:
            return df
        accepted_slim = accepted_slim[["token_id", "outcome_side", "_dec_ts"] + new_cols]
        df = _merge_asof_safe(
            df, accepted_slim,
            on="_spine_ts", by=["token_id", "outcome_side"],
            tolerance=pd.Timedelta("2h"),
        )
        # signal_timestamp = when the decision was made, if we found a match
        if "_dec_ts" in df.columns:
            df["signal_timestamp"] = df["signal_timestamp"].fillna(df["_dec_ts"])
            df.drop(columns=["_dec_ts"], inplace=True, errors="ignore")
        logger.info("After decisions join: %d rows, cycle_id coverage=%.1f%%",
                    len(df),
                    100 * df["cycle_id"].notna().mean() if "cycle_id" in df.columns else 0)
        return df

    def _enrich_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Layer 3: historical_dataset — signal-time regime and sentiment features.

        Joined merge_asof on signal_timestamp by [token_id, outcome_side], 12h tolerance.
        Brings in BTC regime, wallet stats, weather forecast, sentiment.
        """
        hist = _safe_read(self.logs_dir / "historical_dataset.csv")
        if hist.empty:
            return df
        hist["_hist_ts"] = _parse_ts(hist.get("timestamp", pd.Series(dtype=str)))
        hist = hist[hist["_hist_ts"].notna()].copy()

        # Only import columns not already in df (avoid clobbering spine data)
        skip = {"candidate_id", "cycle_id", "position_id", "entry_price",
                "exit_price", "realized_pnl", "size_usdc", "shares",
                "close_reason", "opened_at", "closed_at", "timestamp"}
        new_cols = [c for c in hist.columns if c not in df.columns and c not in skip]
        if not new_cols:
            return df

        hist_slim = hist[["token_id", "outcome_side", "_hist_ts"] + new_cols].copy()
        join_ts = "signal_timestamp" if "signal_timestamp" in df.columns else "_spine_ts"
        df = _merge_asof_safe(
            df, hist_slim,
            on=join_ts, by=["token_id", "outcome_side"],
            tolerance=pd.Timedelta("12h"),
        )
        logger.info("After historical features join: %d rows, %d total columns",
                    len(df), len(df.columns))
        return df

    def _enrich_forecast_eval(self, df: pd.DataFrame) -> pd.DataFrame:
        """Layer 4: btc_forecast_eval — model forecast accuracy at trade open time.

        Joined merge_asof on fill_timestamp ≈ predict_ts, 60m tolerance.
        Brings in: was the model's direction prediction correct for this trade's open?
        """
        feval = _safe_read(self.logs_dir / "btc_forecast_eval.csv")
        if feval.empty or "predict_ts" not in feval.columns:
            return df
        feval["_feval_ts"] = _parse_ts(feval["predict_ts"])
        feval = feval[feval["_feval_ts"].notna()].copy()

        rename_map = {
            "predicted_direction": "forecast_eval_predicted_direction",
            "predicted_return":    "forecast_eval_predicted_return",
            "confidence":          "forecast_eval_confidence",
            "actual_return":       "forecast_eval_actual_return",
            "actual_direction":    "forecast_eval_actual_direction",
            "correct":             "forecast_eval_correct",
            "mtf_agreement":       "forecast_eval_mtf_agreement",
        }
        feval_slim = feval[["_feval_ts"] + [c for c in rename_map if c in feval.columns]].copy()
        feval_slim = feval_slim.rename(columns=rename_map).sort_values("_feval_ts")

        join_ts = "fill_timestamp" if "fill_timestamp" in df.columns else "_spine_ts"
        df = _merge_asof_safe(
            df, feval_slim,
            on=join_ts, by=None,
            tolerance=pd.Timedelta("60min"),
        )
        logger.info("After forecast eval join: forecast_eval_correct coverage=%.1f%%",
                    100 * df.get("forecast_eval_correct", pd.Series()).notna().mean())
        return df

    def _enrich_snapshot_json(self, df: pd.DataFrame) -> pd.DataFrame:
        """Layer 5: expand entry_signal_snapshot_json into individual feature columns."""
        df = _explode_json_col(df, "entry_signal_snapshot_json")
        logger.info("After snapshot expansion: %d columns", len(df.columns))
        return df

    # ── derived labels ────────────────────────────────────────────────────────

    def _derive_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all derived labels from realized outcomes only — no heuristics."""
        out = df.copy()

        # Infer actual_execution_path from close_reason where missing
        if "actual_execution_path" not in out.columns:
            out["actual_execution_path"] = ""
        _raw_path = out["actual_execution_path"].astype(str).str.strip()
        _path_missing = _raw_path.isin(["", "nan", "None", "NaN"])
        _close_reason = _coalesce(out, "close_reason", "exit_reason_family").astype(str).str.lower()
        _inferred = _close_reason.map(lambda r: (
            "live_exit_filled" if r in _LIVE_FILL_REASONS
            else "external_manual_close" if "external_manual_close" in r
            else ""
        ))
        out.loc[_path_missing, "actual_execution_path"] = _inferred[_path_missing]

        # Artifact flags
        _exec_path = out["actual_execution_path"].astype(str).str.lower()
        out["reconciliation_artifact_flag"] = (
            _coalesce(out, "reconciliation_close_flag", "is_reconciliation_close")
            .apply(_truthy)
            | _close_reason.str.contains("reconciliation|external_manual_close", regex=True, na=False)
        )
        out["operational_close_flag"] = _coalesce(out, "operational_close_flag").apply(_truthy)
        out["dead_orderbook_artifact_flag"] = (
            _close_reason.str.contains("dead_orderbook", na=False)
            | _exec_path.str.contains("dead_orderbook", na=False)
        )
        out["stale_price_artifact_flag"] = _close_reason.str.contains("stale", na=False)
        out["missing_execution_path_flag"] = _exec_path.isin(["", "nan"])
        _size_usdc = pd.to_numeric(out.get("size_usdc", 0), errors="coerce").fillna(0)
        _entry_price = pd.to_numeric(out.get("entry_price", 0), errors="coerce").fillna(0)
        out["malformed_fill_flag"] = (_size_usdc <= 0) | (_entry_price <= 0)

        out["learning_eligible"] = ~(
            out["reconciliation_artifact_flag"]
            | out["operational_close_flag"]
            | out["dead_orderbook_artifact_flag"]
            | out["stale_price_artifact_flag"]
            | out["malformed_fill_flag"]
            | out["missing_execution_path_flag"]
        )
        out["contaminated_learning_row"] = ~out["learning_eligible"]
        out["learning_exclusion_reason"] = np.select(
            [out["reconciliation_artifact_flag"], out["operational_close_flag"],
             out["dead_orderbook_artifact_flag"], out["stale_price_artifact_flag"],
             out["malformed_fill_flag"], out["missing_execution_path_flag"]],
            ["reconciliation_artifact", "operational_close", "dead_orderbook",
             "stale_price", "malformed_fill", "missing_execution_path"],
            default="clean",
        )

        # price_return
        _ep = pd.to_numeric(out.get("entry_price", np.nan), errors="coerce")
        _xp = pd.to_numeric(out.get("exit_price", np.nan), errors="coerce")
        _pr = ((_xp - _ep) / _ep.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        if "price_return" not in out.columns:
            out["price_return"] = _pr
        else:
            out["price_return"] = pd.to_numeric(out["price_return"], errors="coerce").fillna(_pr)

        # hold_time_minutes
        if "fill_timestamp" in out.columns and "close_timestamp" in out.columns:
            _ft = out["fill_timestamp"] if pd.api.types.is_datetime64_any_dtype(out["fill_timestamp"]) else _parse_ts(out["fill_timestamp"])
            _ct = out["close_timestamp"] if pd.api.types.is_datetime64_any_dtype(out["close_timestamp"]) else _parse_ts(out["close_timestamp"])
            out["hold_time_minutes"] = (_ct - _ft).dt.total_seconds().div(60).where(
                _ft.notna() & _ct.notna(), other=np.nan
            )

        # entry_edge_realized = realized_pnl / size_usdc
        _pnl = pd.to_numeric(_coalesce(out, "net_realized_pnl", "realized_pnl"), errors="coerce").fillna(0)
        out["entry_edge_realized"] = (_pnl / _size_usdc.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        out["trade_outcome_label"] = np.where(_pnl > 0, "win", np.where(_pnl < 0, "loss", "flat"))
        out["entry_quality"] = np.where(out["entry_edge_realized"] > 0, 1, 0)

        # exit_regret: BTC only — mfe_60m is a price-space path metric, meaningless for weather
        _family = _coalesce(out, "market_family", "").astype(str).str.lower()
        _is_btc = ~_family.str.startswith("weather")
        _mfe_col = _coalesce(out, "max_favorable_excursion_pct", "mfe_60m")
        _mfe = pd.to_numeric(_mfe_col, errors="coerce")
        _exit_regret = (_mfe - out["price_return"]).replace([np.inf, -np.inf], np.nan)
        out["exit_regret"] = np.where(_is_btc, _exit_regret, np.nan)
        out["exit_regret"] = pd.to_numeric(out["exit_regret"], errors="coerce").fillna(0)
        out["exit_quality"] = np.where(_is_btc, np.where(out["exit_regret"] <= 0.01, 1, 0), np.nan)

        # slippage_error
        _pred_slip = pd.to_numeric(
            _coalesce(out, "exec_expected_slippage", "predicted_slippage"), errors="coerce"
        ).fillna(0)
        _act_slip = pd.to_numeric(out.get("exit_realized_slippage_bps", np.nan), errors="coerce").div(10000).fillna(0)
        out["slippage_error"] = (_act_slip - _pred_slip).fillna(0)
        out["execution_quality"] = np.where(out["slippage_error"] <= 0, 1, 0)

        # strategy_quality — grouped by market_family + entry_model_family
        _strat = (
            _coalesce(out, "market_family", "technical_regime_bucket", "signal_label")
            .astype(str).fillna("unknown")
        )
        _strat_mean = out.groupby(_strat)["entry_edge_realized"].transform("mean")
        out["strategy_quality"] = np.where(_strat_mean > 0, 1, 0)

        # Canonical timestamps — normalize column names
        if "signal_timestamp" not in out.columns:
            out["signal_timestamp"] = _parse_ts(_coalesce(out, "opened_at", "closed_at"))
        if "fill_timestamp" not in out.columns:
            out["fill_timestamp"] = _parse_ts(out.get("opened_at", pd.Series(dtype=str)))
        if "close_timestamp" not in out.columns:
            out["close_timestamp"] = _parse_ts(out.get("closed_at", pd.Series(dtype=str)))

        # market_implied_prob — alias for entry_price in CLOB (the "price" IS the probability)
        if "market_implied_prob" not in out.columns and "entry_price" in out.columns:
            out["market_implied_prob"] = pd.to_numeric(out["entry_price"], errors="coerce")

        # spread_at_signal — coalesce from available spread columns
        if "spread_at_signal" not in out.columns:
            out["spread_at_signal"] = pd.to_numeric(
                _coalesce(out, "spread", "spread_pct"), errors="coerce"
            )

        return out

    # ── output assembly ───────────────────────────────────────────────────────

    def _select_sections(self, df: pd.DataFrame, is_btc: bool) -> pd.DataFrame:
        """Select columns in manifest order for the given family."""
        sections = (
            SECTION_A_IDENTITY + SECTION_B_MARKET + SECTION_C_TIMESTAMPS
            + SECTION_D_MICROSTRUCTURE
            + (SECTION_E_BTC_REGIME if is_btc else [])
            + (SECTION_F_WEATHER_FORECAST if not is_btc else [])
            + SECTION_G_WALLET + SECTION_H_ORDER_DECISION
            + SECTION_I_FILL + SECTION_J_CLOSE
            + (SECTION_K_BTC_ATTRIBUTION if is_btc else [])
            + (SECTION_K_WEATHER_ATTRIBUTION if not is_btc else [])
            + SECTION_L_DERIVED_LABELS + SECTION_M_QUALITY
        )
        present = [c for c in sections if c in df.columns]
        extra = [c for c in df.columns if c not in sections and c not in {
            "_spine_ts", "_hist_ts", "_feval_ts", "entry_signal_snapshot_json",
        }]
        return df[present + extra].copy()

    def _dedup(self, df: pd.DataFrame) -> pd.DataFrame:
        key_cols = [c for c in ["candidate_id", "cycle_id", "token_id", "fill_timestamp"]
                    if c in df.columns]
        if key_cols:
            df = df.drop_duplicates(subset=key_cols, keep="last")
        return df.reset_index(drop=True)

    # ── main ──────────────────────────────────────────────────────────────────

    def build(self) -> pd.DataFrame:
        df = self._load_spine()
        if df.empty:
            return df

        df = self._enrich_btc_attribution(df)
        df = self._enrich_decisions(df)
        df = self._enrich_historical_features(df)
        df = self._enrich_forecast_eval(df)
        df = self._enrich_snapshot_json(df)
        df = self._derive_labels(df)
        df = self._dedup(df)
        # Drop exact duplicate columns (same name, keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()

        logger.info(
            "Canonical dataset: %d rows, %d columns | eligible=%d (%.1f%%)",
            len(df), len(df.columns),
            df["learning_eligible"].sum() if "learning_eligible" in df.columns else 0,
            100 * df["learning_eligible"].mean() if "learning_eligible" in df.columns else 0,
        )
        return df

    def write(self) -> dict[str, pd.DataFrame]:
        """Build and write family-split canonical datasets. Returns {family: df}."""
        from model_feature_safety import clean_dataframe_for_training

        df = self.build()
        if df.empty:
            return {}

        df = clean_dataframe_for_training(df, context="canonical_dataset")

        # Split by family
        _family_col = df.get("market_family", pd.Series("", index=df.index))
        _weather_mask = _family_col.astype(str).str.lower().str.startswith("weather")

        btc_df = self._select_sections(df[~_weather_mask].copy(), is_btc=True)
        weather_df = self._select_sections(df[_weather_mask].copy(), is_btc=False)

        results: dict[str, pd.DataFrame] = {}

        if not btc_df.empty:
            self.btc_out.parent.mkdir(parents=True, exist_ok=True)
            btc_df.to_csv(self.btc_out, index=False)
            logger.info("BTC canonical: %d rows, %d cols → %s", len(btc_df), len(btc_df.columns), self.btc_out)
            results["btc"] = btc_df
        else:
            logger.warning("No BTC rows in canonical dataset")

        if not weather_df.empty:
            self.weather_out.parent.mkdir(parents=True, exist_ok=True)
            weather_df.to_csv(self.weather_out, index=False)
            logger.info("Weather canonical: %d rows, %d cols → %s", len(weather_df), len(weather_df.columns), self.weather_out)
            results["weather_temperature"] = weather_df
        else:
            logger.info("No weather rows yet (expected while weather trading not live)")

        # Combined file for backward compat — clearly labeled, never train from this
        df_combined = self._select_sections(df, is_btc=True)  # carry all cols
        for c in SECTION_F_WEATHER_FORECAST + SECTION_K_WEATHER_ATTRIBUTION:
            if c in df.columns and c not in df_combined.columns:
                df_combined[c] = df[c]
        df_combined.to_csv(self.combined_out, index=False)
        logger.info("Combined canonical (DO NOT TRAIN FROM THIS): %d rows → %s", len(df_combined), self.combined_out)

        return results


# ── validation ────────────────────────────────────────────────────────────────

def validate_canonical(df: pd.DataFrame, family: str) -> list[str]:
    warns: list[str] = []
    tag = f"[{family}]"
    n = len(df)
    if n == 0:
        return [f"{tag} empty dataset"]

    # Identity completeness
    for col in ["token_id", "fill_timestamp", "close_timestamp"]:
        if col in df.columns:
            missing_pct = 100 * df[col].isna().mean()
            if missing_pct > 10:
                warns.append(f"{tag} {col}: {missing_pct:.1f}% missing")

    # Label integrity
    if "trade_outcome_label" in df.columns:
        dist = df["trade_outcome_label"].value_counts(normalize=True)
        if dist.max() > 0.97:
            warns.append(f"{tag} trade_outcome_label: {dist.max():.1%} one class ({dist.idxmax()})")

    if "entry_edge_realized" in df.columns:
        vals = pd.to_numeric(df["entry_edge_realized"], errors="coerce")
        if vals.isna().all():
            warns.append(f"{tag} entry_edge_realized: all NaN")

    # Family-specific cross-contamination guard
    if family.startswith("weather"):
        btc_only = [c for c in SECTION_E_BTC_REGIME if c in df.columns and df[c].notna().any()]
        if btc_only:
            warns.append(f"{tag} BTC regime columns present in weather table: {btc_only[:5]}")
        if "exit_regret" in df.columns and pd.to_numeric(df["exit_regret"], errors="coerce").notna().any():
            non_zero = (pd.to_numeric(df["exit_regret"], errors="coerce") != 0).sum()
            if non_zero > 0:
                warns.append(f"{tag} exit_regret has {non_zero} non-NaN/non-zero values — BTC price metric leaked into weather")
    else:
        weather_only = [c for c in SECTION_F_WEATHER_FORECAST if c in df.columns and df[c].notna().any()]
        if weather_only:
            warns.append(f"{tag} weather forecast columns present in BTC table: {weather_only[:5]}")

    # Artifact bleed-through
    for flag in ["reconciliation_artifact_flag", "operational_close_flag"]:
        if flag in df.columns and "learning_eligible" in df.columns:
            eligible_mask = df["learning_eligible"].fillna(False).astype(bool)
            # Guard against duplicate column names returning a DataFrame
            raw = df[flag]
            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[:, 0]
            n_bad = int(raw[eligible_mask].fillna(False).astype(bool).sum())
            if n_bad > 0:
                warns.append(f"{tag} {n_bad} eligible rows still have {flag}=True")

    # Coverage report
    coverage = {
        col: f"{100 * df[col].notna().mean():.0f}%"
        for col in ["cycle_id", "calibrated_edge", "btc_predicted_direction",
                    "forecast_eval_correct", "wallet_alpha_30d", "btc_market_regime_label"]
        if col in df.columns
    }
    logger.info("%s column coverage: %s", tag, coverage)

    return warns


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    import argparse

    parser = argparse.ArgumentParser(description="Build canonical per-trade learning dataset.")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    builder = CanonicalDatasetBuilder(logs_dir=args.logs_dir)
    results = builder.write()

    if args.validate and results:
        print("\nVALIDATION")
        print("-" * 60)
        any_warn = False
        for fam, table in results.items():
            warns = validate_canonical(table, fam)
            if warns:
                any_warn = True
                for w in warns:
                    print(f"  WARN  {w}")
            else:
                print(f"  OK    [{fam}] {len(table)} rows, {len(table.columns)} columns")
        if not any_warn:
            print("All canonical tables passed validation.")
        print()
