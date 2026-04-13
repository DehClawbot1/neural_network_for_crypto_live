"""
feedback_attribution_engine.py
Phase 2 — Feedback Attribution Engine

Every completed trade is decomposed into four attribution channels:
  2A  Alpha attribution  — was the model's edge and direction correct?
  2B  Execution attribution — did fills/slippage help or hurt?
  2C  Policy attribution — did the governor/sizing help or hurt?
  2D  Operational attribution — did the trade fail for non-market reasons?

Each channel writes rows to its own CSV sink.  These CSVs become the
training data for the separate learning targets in Phase 5.

Rule 0.2 is enforced here: every outcome must be tagged before it can
influence any model.  No unattributed row ever reaches training.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from csv_utils import safe_csv_append

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_OPERATIONAL_CLOSE_REASONS = {
    "reconciliation",
    "external_manual_close",
    "api_error",
    "stale_data",
    "broken_fill",
    "manual_override",
    "missing_orderbook",
}

_ATTRIBUTION_SCHEMA_ALPHA = [
    "attributed_at",
    "report_id",
    "token_id",
    "condition_id",
    "market_family",
    "signal_label",
    "direction_correct",
    "ev_sign_correct",
    "confidence_calibrated",
    "trade_quality_justified",
    "entry_p_tp",
    "entry_ev",
    "entry_edge",
    "realized_pnl",
    "roi",
    "holding_minutes",
    "outcome_class",
    "alpha_verdict",       # "alpha_success" | "alpha_failure"
    "reliability_score",
    "market_implied_prob",
    "fair_prob",
    "model_version",
]

_ATTRIBUTION_SCHEMA_EXECUTION = [
    "attributed_at",
    "report_id",
    "token_id",
    "market_family",
    "predicted_fill_prob",
    "actual_filled",
    "predicted_slippage",
    
    # 4-part Slippage Decomposition
    "decision_to_arrival_slippage",
    "spread_cost",
    "market_impact_cost",
    "queue_delay_cost",
    
    # Partial Fill Intent
    "filled_qty",
    "unfilled_qty",
    "cancel_reason",
    "partial_fill_reason",
    "time_to_first_fill",
    "time_to_last_fill",

    "fill_prob_error",
    "slippage_error",
    "fill_latency_ms",
    "fill_latency_ms_entry",      # Fix 4: separate entry latency
    "fill_latency_ms_exit",       # Fix 4: separate exit latency
    "spread_at_entry",
    "orderbook_depth_at_entry",   # Fix 4: total depth visible at order time
    "cancel_risk_at_entry",       # Fix 4: estimated cancel risk score
    "order_type",                 # Fix 4: GTC vs FOK vs GTD
    "requested_size",             # Fix 4: intended position size
    "exec_verdict",               # "exec_success" | "exec_failure"
    "reliability_score",
]

_ATTRIBUTION_SCHEMA_POLICY = [
    "attributed_at",
    "report_id",
    "token_id",
    "market_family",
    "signal_label",
    "governor_level",
    "size_multiplier_applied",
    "top_signal_only_active",
    "trade_skipped",
    "skip_avoided_loss",
    "size_helped",
    "portfolio_cap_triggered",
    "realized_pnl",
    "policy_verdict",      # "policy_success" | "policy_failure"
    "reliability_score",
]

_ATTRIBUTION_SCHEMA_OPERATIONAL = [
    "attributed_at",
    "report_id",
    "token_id",
    "market_family",
    "close_reason",
    "failure_type",        # reconciliation | api_error | stale_data | broken_fill | manual_override | missing_orderbook
    "operational_verdict", # "operational_failure"
]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        num = float(value)
        return float(default) if not np.isfinite(num) else num
    except Exception:
        return float(default)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reliability_from_flags(
    *,
    reconciliation_close: bool,
    operational_close: bool,
    missing_entry: bool,
    missing_exit: bool,
    fill_quality_poor: bool,
    forecast_disagreement_extreme: bool,
) -> float:
    """Phase 3 reliability score: 1.0 / 0.5 / 0.1 / 0.0."""
    if reconciliation_close or operational_close or missing_entry or missing_exit:
        return 0.0
    if fill_quality_poor or forecast_disagreement_extreme:
        return 0.5
    return 1.0


# ---------------------------------------------------------------------------
# FeedbackAttributionEngine
# ---------------------------------------------------------------------------
class FeedbackAttributionEngine:
    """
    Decomposes every closed trade outcome into four attribution channels
    and writes the results to CSV sinks under logs_dir.

    Usage:
        engine = FeedbackAttributionEngine(logs_dir="logs")
        engine.attribute(report_row)   # report_row is a dict from TradeFeedbackLearner
    """

    def __init__(self, logs_dir: str = "logs") -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.alpha_csv = self.logs_dir / "alpha_feedback_clean.csv"
        self.execution_csv = self.logs_dir / "execution_feedback.csv"
        self.policy_csv = self.logs_dir / "policy_feedback.csv"
        self.operational_csv = self.logs_dir / "operational_failures.csv"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attribute(self, report: dict[str, Any]) -> dict[str, str]:
        """
        Attribute a single closed-trade report dict.
        Returns a summary dict of verdicts so callers can log/inspect.
        """
        verdicts: dict[str, str] = {}

        close_reason = str(report.get("close_reason", "") or "").strip().lower()
        recon_flag = bool(report.get("reconciliation_close_flag", False))
        ops_flag = bool(report.get("operational_close_flag", False))

        # --- 2D: operational first (fast-path exit) --------------------
        if recon_flag or ops_flag or close_reason in _OPERATIONAL_CLOSE_REASONS:
            ops_row = self._build_operational_row(report, close_reason)
            safe_csv_append(self.operational_csv, pd.DataFrame([ops_row]))
            verdicts["operational"] = "operational_failure"
            # Still write policy and execution rows so we have full coverage,
            # but mark reliability = 0.0 so they are excluded from training.

        # --- 2A: alpha attribution -------------------------------------
        try:
            alpha_row = self._attribute_alpha(report)
            safe_csv_append(self.alpha_csv, pd.DataFrame([alpha_row]))
            verdicts["alpha"] = alpha_row["alpha_verdict"]
        except Exception as exc:
            logger.warning("Alpha attribution failed for %s: %s", report.get("report_id"), exc)

        # --- 2B: execution attribution ---------------------------------
        try:
            exec_row = self._attribute_execution(report)
            safe_csv_append(self.execution_csv, pd.DataFrame([exec_row]))
            verdicts["execution"] = exec_row["exec_verdict"]
        except Exception as exc:
            logger.warning("Execution attribution failed for %s: %s", report.get("report_id"), exc)

        # --- 2C: policy attribution ------------------------------------
        try:
            policy_row = self._attribute_policy(report)
            safe_csv_append(self.policy_csv, pd.DataFrame([policy_row]))
            verdicts["policy"] = policy_row["policy_verdict"]
        except Exception as exc:
            logger.warning("Policy attribution failed for %s: %s", report.get("report_id"), exc)

        logger.info(
            "FeedbackAttribution report_id=%s verdicts=%s",
            report.get("report_id"),
            verdicts,
        )
        return verdicts

    def attribute_batch(self, reports: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Attribute a list of report dicts. Returns list of verdict dicts."""
        return [self.attribute(r) for r in reports]

    def attribute_from_csv(self) -> int:
        """
        Read TradeFeedbackLearner's report CSV and attribute all rows
        that have not been attributed yet.  Returns count of new rows
        attributed this call.
        """
        report_csv = self.logs_dir / "trade_feedback_reports.csv"
        attributed_ids = self._load_attributed_ids()

        if not report_csv.exists():
            return 0
        try:
            df = pd.read_csv(report_csv, engine="python", on_bad_lines="skip")
        except Exception:
            return 0
        if df.empty or "report_id" not in df.columns:
            return 0

        new_df = df[~df["report_id"].astype(str).isin(attributed_ids)].copy()
        if new_df.empty:
            return 0

        count = 0
        for _, row in new_df.iterrows():
            self.attribute(row.to_dict())
            count += 1
        logger.info("FeedbackAttributionEngine: attributed %d new rows from CSV", count)
        return count

    # ------------------------------------------------------------------
    # Private — alpha attribution (2A)
    # ------------------------------------------------------------------
    def _attribute_alpha(self, r: dict[str, Any]) -> dict[str, Any]:
        realized_pnl = _safe_float(r.get("realized_pnl"))
        roi = _safe_float(r.get("roi"))
        entry_p_tp = _safe_float(r.get("entry_p_tp_before_sl"))
        entry_ev = _safe_float(r.get("entry_expected_return"))
        entry_edge = _safe_float(r.get("entry_edge_score"))
        confidence = _safe_float(r.get("confidence_at_entry", r.get("entry_confidence")))
        market_implied_prob = _safe_float(r.get("market_implied_prob"), 0.5)
        fair_prob = _safe_float(r.get("fair_prob"), entry_p_tp)

        # Direction correct: did the model call the right side?
        direction_correct = (
            (entry_p_tp >= 0.5 and realized_pnl > 0)
            or (entry_p_tp < 0.5 and realized_pnl < 0)
        )

        # EV sign correct: did expected return sign match outcome?
        ev_sign_correct = (entry_ev > 0 and realized_pnl > 0) or (entry_ev <= 0 and realized_pnl <= 0)

        # Calibration: confidence vs win
        win = realized_pnl > 0
        # Overconfidence flag: conf >= 0.70 but lost
        confidence_calibrated = not (confidence >= 0.70 and not win)

        # Trade quality justified: edge score > 0 and won, or edge <= 0 and lost
        trade_quality_justified = (entry_edge > 0 and win) or (entry_edge <= 0 and not win)

        alpha_verdict = "alpha_success" if (direction_correct and ev_sign_correct) else "alpha_failure"

        reliability = _reliability_from_flags(
            reconciliation_close=bool(r.get("reconciliation_close_flag", False)),
            operational_close=bool(r.get("operational_close_flag", False)),
            missing_entry=(_safe_float(r.get("entry_price")) <= 0),
            missing_exit=(_safe_float(r.get("exit_price")) <= 0),
            fill_quality_poor=bool(r.get("fill_quality_poor", False)),
            forecast_disagreement_extreme=bool(r.get("forecast_disagreement_extreme", False)),
        )

        return {
            "attributed_at": _now_iso(),
            "report_id": r.get("report_id", ""),
            "token_id": r.get("token_id", ""),
            "condition_id": r.get("condition_id", ""),
            "market_family": r.get("market_family", ""),
            "signal_label": r.get("signal_label", ""),
            "direction_correct": int(direction_correct),
            "ev_sign_correct": int(ev_sign_correct),
            "confidence_calibrated": int(confidence_calibrated),
            "trade_quality_justified": int(trade_quality_justified),
            "entry_p_tp": round(entry_p_tp, 6),
            "entry_ev": round(entry_ev, 6),
            "entry_edge": round(entry_edge, 6),
            "realized_pnl": round(realized_pnl, 6),
            "roi": round(roi, 6),
            "holding_minutes": _safe_float(r.get("holding_minutes")),
            "outcome_class": r.get("outcome_class", ""),
            "alpha_verdict": alpha_verdict,
            "reliability_score": reliability,
            "market_implied_prob": round(market_implied_prob, 6),
            "fair_prob": round(fair_prob, 6),
            "model_version": r.get("entry_model_version", ""),
        }

    # ------------------------------------------------------------------
    # Private — execution attribution (2B)
    # ------------------------------------------------------------------
    def _attribute_execution(self, r: dict[str, Any]) -> dict[str, Any]:
        predicted_fill_prob = _safe_float(r.get("predicted_fill_prob"), np.nan)
        actual_filled = 1.0 if _safe_float(r.get("shares")) > 0 else 0.0
        predicted_slippage = _safe_float(r.get("predicted_slippage"), np.nan)
        actual_entry = _safe_float(r.get("entry_price"))
        actual_exit = _safe_float(r.get("exit_price"))

        # 4-part Slippage Decomposition
        decision_to_arrival_slippage = _safe_float(r.get("decision_to_arrival_slippage"), 0.0)
        spread_cost = _safe_float(r.get("spread_cost"), 0.0)
        market_impact_cost = _safe_float(r.get("market_impact_cost"), 0.0)
        queue_delay_cost = _safe_float(r.get("queue_delay_cost"), 0.0)
        total_actual_slippage = decision_to_arrival_slippage + spread_cost + market_impact_cost + queue_delay_cost
        
        # Fallback if old row with only actual_slippage
        if total_actual_slippage == 0.0:
            total_actual_slippage = _safe_float(r.get("actual_slippage"), 0.0)

        # Partial Fill Intent Properties
        filled_qty = _safe_float(r.get("filled_qty"), _safe_float(r.get("shares"), 0.0))
        unfilled_qty = _safe_float(r.get("unfilled_qty"), 0.0)
        cancel_reason = str(r.get("cancel_reason", ""))
        partial_fill_reason = str(r.get("partial_fill_reason", ""))
        time_to_first_fill = _safe_float(r.get("time_to_first_fill"), 0.0)
        time_to_last_fill = _safe_float(r.get("time_to_last_fill"), 0.0)

        fill_prob_error = (
            abs(predicted_fill_prob - actual_filled)
            if np.isfinite(predicted_fill_prob)
            else np.nan
        )
        slippage_error = (
            abs(predicted_slippage - total_actual_slippage)
            if np.isfinite(predicted_slippage)
            else np.nan
        )

        # Exec success: fill happened and slippage error < 3%
        exec_success = (actual_filled == 1.0) and (
            not np.isfinite(slippage_error) or slippage_error < 0.03
        )
        exec_verdict = "exec_success" if exec_success else "exec_failure"

        reliability = _reliability_from_flags(
            reconciliation_close=bool(r.get("reconciliation_close_flag", False)),
            operational_close=bool(r.get("operational_close_flag", False)),
            missing_entry=(actual_entry <= 0),
            missing_exit=(actual_exit <= 0),
            fill_quality_poor=bool(r.get("fill_quality_poor", False)),
            forecast_disagreement_extreme=False,
        )

        return {
            "attributed_at": _now_iso(),
            "report_id": r.get("report_id", ""),
            "token_id": r.get("token_id", ""),
            "market_family": r.get("market_family", ""),
            "predicted_fill_prob": round(predicted_fill_prob, 6) if np.isfinite(predicted_fill_prob) else None,
            "actual_filled": actual_filled,
            "predicted_slippage": round(predicted_slippage, 6) if np.isfinite(predicted_slippage) else None,
            
            "decision_to_arrival_slippage": round(decision_to_arrival_slippage, 6),
            "spread_cost": round(spread_cost, 6),
            "market_impact_cost": round(market_impact_cost, 6),
            "queue_delay_cost": round(queue_delay_cost, 6),

            "filled_qty": round(filled_qty, 6),
            "unfilled_qty": round(unfilled_qty, 6),
            "cancel_reason": cancel_reason,
            "partial_fill_reason": partial_fill_reason,
            "time_to_first_fill": time_to_first_fill,
            "time_to_last_fill": time_to_last_fill,

            "fill_prob_error": round(fill_prob_error, 6) if np.isfinite(fill_prob_error) else None,
            "slippage_error": round(slippage_error, 6) if np.isfinite(slippage_error) else None,
            # Latency — overall and split by entry/exit (Fix 4)
            "fill_latency_ms": _safe_float(r.get("fill_latency_ms"), 0.0),
            "fill_latency_ms_entry": _safe_float(r.get("fill_latency_ms_entry") or r.get("fill_latency_ms"), 0.0),
            "fill_latency_ms_exit": _safe_float(r.get("fill_latency_ms_exit"), 0.0),
            # Orderbook context at entry (Fix 4)
            "spread_at_entry": _safe_float(r.get("spread_at_entry"), 0.0),
            "orderbook_depth_at_entry": _safe_float(r.get("orderbook_depth_at_entry"), None),
            "cancel_risk_at_entry": _safe_float(r.get("cancel_risk_at_entry"), None),
            "order_type": str(r.get("order_type") or "GTC"),
            "requested_size": _safe_float(r.get("requested_size"), None),
            "exec_verdict": exec_verdict,
            "reliability_score": reliability,
        }

    # ------------------------------------------------------------------
    # Private — policy attribution (2C)
    # ------------------------------------------------------------------
    def _attribute_policy(self, r: dict[str, Any]) -> dict[str, Any]:
        realized_pnl = _safe_float(r.get("realized_pnl"))
        governor_level = int(_safe_float(r.get("performance_governor_level"), 0))
        size_multiplier = _safe_float(r.get("size_multiplier_applied"), 1.0)
        top_signal_only = bool(r.get("top_signal_only_active", False))
        trade_skipped = bool(r.get("trade_skipped", False))
        portfolio_cap_triggered = bool(r.get("portfolio_cap_triggered", False))

        # Did reduced size help? If governor reduced size and pnl was negative,
        # a smaller loss means the reduction helped.
        size_helped = (size_multiplier < 1.0 and realized_pnl < 0)

        # Did skip logic avoid a bad trade? Only knowable when we have skip candidates.
        skip_avoided_loss = bool(r.get("skip_avoided_loss", False))

        policy_verdict = "policy_success" if (
            (size_helped)
            or (skip_avoided_loss)
            or (not trade_skipped and realized_pnl > 0)
        ) else "policy_failure"

        reliability = _reliability_from_flags(
            reconciliation_close=bool(r.get("reconciliation_close_flag", False)),
            operational_close=bool(r.get("operational_close_flag", False)),
            missing_entry=(_safe_float(r.get("entry_price")) <= 0),
            missing_exit=(_safe_float(r.get("exit_price")) <= 0),
            fill_quality_poor=bool(r.get("fill_quality_poor", False)),
            forecast_disagreement_extreme=False,
        )

        return {
            "attributed_at": _now_iso(),
            "report_id": r.get("report_id", ""),
            "token_id": r.get("token_id", ""),
            "market_family": r.get("market_family", ""),
            "signal_label": r.get("signal_label", ""),
            "governor_level": governor_level,
            "size_multiplier_applied": round(size_multiplier, 4),
            "top_signal_only_active": int(top_signal_only),
            "trade_skipped": int(trade_skipped),
            "skip_avoided_loss": int(skip_avoided_loss),
            "size_helped": int(size_helped),
            "portfolio_cap_triggered": int(portfolio_cap_triggered),
            "realized_pnl": round(realized_pnl, 6),
            "policy_verdict": policy_verdict,
            "reliability_score": reliability,
        }

    # ------------------------------------------------------------------
    # Private — operational attribution (2D)
    # ------------------------------------------------------------------
    def _build_operational_row(self, r: dict[str, Any], close_reason: str) -> dict[str, Any]:
        failure_type = "other"
        if bool(r.get("reconciliation_close_flag", False)):
            failure_type = "reconciliation"
        elif "api" in close_reason:
            failure_type = "api_error"
        elif "stale" in close_reason:
            failure_type = "stale_data"
        elif "broken" in close_reason or "fill" in close_reason:
            failure_type = "broken_fill"
        elif "manual" in close_reason or "override" in close_reason:
            failure_type = "manual_override"
        elif "orderbook" in close_reason:
            failure_type = "missing_orderbook"
        elif bool(r.get("operational_close_flag", False)):
            failure_type = "operational_flag"

        return {
            "attributed_at": _now_iso(),
            "report_id": r.get("report_id", ""),
            "token_id": r.get("token_id", ""),
            "market_family": r.get("market_family", ""),
            "close_reason": close_reason,
            "failure_type": failure_type,
            "operational_verdict": "operational_failure",
        }

    # ------------------------------------------------------------------
    # Private — helpers
    # ------------------------------------------------------------------
    def _load_attributed_ids(self) -> set[str]:
        """Return set of report_ids already present in any attribution sink."""
        ids: set[str] = set()
        for csv_path in [self.alpha_csv, self.execution_csv, self.policy_csv]:
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path, usecols=["report_id"], engine="python", on_bad_lines="skip")
                ids.update(df["report_id"].dropna().astype(str).tolist())
            except Exception:
                pass
        return ids
