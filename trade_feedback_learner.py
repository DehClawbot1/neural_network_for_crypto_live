import hashlib
import json
import logging
import os
import re
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from db import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TradeFeedbackLearner:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = self.logs_dir / "trade_feedback_reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.report_csv = self.logs_dir / "trade_feedback_reports.csv"
        self.summary_csv = self.logs_dir / "trade_feedback_summary.csv"
        self.positions_csv = self.logs_dir / "positions.csv"
        self.open_pain_summary_csv = self.logs_dir / "trade_feedback_open_pain_summary.csv"
        self.db = Database(self.logs_dir / "trading.db")

    def _safe_float(self, value, default=0.0):
        try:
            value = float(value)
        except Exception:
            return float(default)
        if not np.isfinite(value):
            return float(default)
        return float(value)

    def _safe_read(self, path: Path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def _safe_json_load(self, value, default=None):
        if value in [None, ""]:
            return {} if default is None else default
        if isinstance(value, dict):
            return value
        try:
            return json.loads(value)
        except Exception:
            return {} if default is None else default

    def _env_float(self, name: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
        try:
            value = float(os.getenv(name, str(default)) or default)
        except Exception:
            value = float(default)
        if minimum is not None:
            value = max(float(minimum), value)
        if maximum is not None:
            value = min(float(maximum), value)
        return float(value)

    def _env_int(self, name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
        try:
            value = int(os.getenv(name, str(default)) or default)
        except Exception:
            value = int(default)
        if minimum is not None:
            value = max(int(minimum), value)
        if maximum is not None:
            value = min(int(maximum), value)
        return int(value)

    def _numeric_series(self, frame: pd.DataFrame, column: str, default=0.0):
        if frame is None or frame.empty:
            return pd.Series(dtype=float)
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(default)
        return pd.Series(default, index=frame.index, dtype=float)

    def _numeric_series_any(self, frame: pd.DataFrame, columns, default=0.0):
        if frame is None or frame.empty:
            return pd.Series(dtype=float)
        for column in columns:
            if column in frame.columns:
                return pd.to_numeric(frame[column], errors="coerce").fillna(default)
        return pd.Series(default, index=frame.index, dtype=float)

    def _safe_iso(self, value):
        try:
            ts = pd.to_datetime(value, utc=True, errors="coerce")
            if pd.isna(ts):
                return None
            return ts.isoformat()
        except Exception:
            return None

    def _slugify(self, value: str):
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9._-]+", "_", text)
        return text.strip("_") or "unknown"

    def _trade_report_id(self, trade):
        token_id = self._slugify(getattr(trade, "token_id", "") or "no_token")
        outcome_side = self._slugify(getattr(trade, "outcome_side", "") or "na")
        opened_at = self._safe_iso(getattr(trade, "opened_at", None)) or ""
        closed_at = self._safe_iso(getattr(trade, "closed_at", None)) or datetime.now(timezone.utc).isoformat()
        opened_stamp = re.sub(r"[^0-9]", "", opened_at)[:14] or "unknown"
        closed_stamp = re.sub(r"[^0-9]", "", closed_at)[:14] or "unknown"
        close_fingerprint = str(getattr(trade, "close_fingerprint", "") or "").strip()
        digest_source = close_fingerprint or f"{token_id}|{outcome_side}|{opened_stamp}|{closed_stamp}"
        digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:10]
        return f"{closed_stamp}_{opened_stamp}_{token_id}_{outcome_side}_{digest}"

    def _query_one(self, query, params):
        rows = self.db.query_all(query, params)
        return rows[0] if rows else {}

    def _lookup_entry_context(self, trade):
        token_id = str(getattr(trade, "token_id", "") or "")
        condition_id = str(getattr(trade, "condition_id", "") or "")
        outcome_side = str(getattr(trade, "outcome_side", "") or "")
        entry_ctx = self._query_one(
            """
            SELECT *
            FROM candidate_decisions
            WHERE token_id = ?
              AND COALESCE(condition_id, '') = ?
              AND COALESCE(outcome_side, '') = ?
              AND final_decision IN ('ENTRY_FILLED', 'PAPER_OPENED')
            ORDER BY decision_id DESC
            LIMIT 1
            """,
            (token_id, condition_id, outcome_side),
        )
        model_ctx = self._query_one(
            """
            SELECT *
            FROM model_decisions
            WHERE token_id = ?
              AND COALESCE(condition_id, '') = ?
              AND COALESCE(outcome_side, '') = ?
            ORDER BY decision_id DESC
            LIMIT 1
            """,
            (token_id, condition_id, outcome_side),
        )
        return entry_ctx, model_ctx

    def _grade_trade(self, realized_pnl: float, roi: float):
        if realized_pnl > 0 and roi >= 0.06:
            return "A"
        if realized_pnl > 0:
            return "B"
        if abs(realized_pnl) < 1e-9:
            return "C"
        if roi > -0.03:
            return "D"
        return "F"

    def _extract_takeaways(self, *, realized_pnl, roi, close_reason, confidence_at_entry, holding_minutes, entry_candidate):
        tags = []
        strengths = []
        weaknesses = []
        adjustments = []

        p_tp = self._safe_float(entry_candidate.get("p_tp_before_sl"), 0.0)
        expected_return = self._safe_float(entry_candidate.get("expected_return"), 0.0)
        edge_score = self._safe_float(entry_candidate.get("edge_score"), 0.0)

        if realized_pnl > 0:
            tags.append("profitable")
            strengths.append("The trade finished positive and validated at least part of the entry thesis.")
        elif realized_pnl < 0:
            tags.append("loss")
            weaknesses.append("The trade closed at a loss, so the original thesis did not translate into profitable execution.")
        else:
            tags.append("flat")
            adjustments.append("This was effectively flat. Capital may have been tied up without enough reward.")

        if confidence_at_entry >= 0.7 and realized_pnl < 0:
            tags.append("overconfidence")
            weaknesses.append("Entry confidence was high, but the outcome was negative. This is a sign the model was overconfident here.")
            adjustments.append("Down-weight similar high-confidence setups unless recent win rate improves.")
        elif confidence_at_entry <= 0.5 and realized_pnl > 0:
            tags.append("underconfidence")
            strengths.append("The setup worked despite modest confidence, which suggests similar signals may deserve more respect.")

        if close_reason == "stop_loss":
            tags.append("stop_loss_hit")
            weaknesses.append("The position moved against the thesis quickly enough to trigger the stop-loss.")
            adjustments.append("Be stricter on similar entries with weak edge or late timing.")
        elif close_reason in {"take_profit_roi", "take_profit_price_move", "take_profit_model_target"}:
            tags.append("take_profit")
            strengths.append("The bot captured gains using its take-profit logic instead of letting the trade drift.")
        elif close_reason == "trailing_stop":
            tags.append("trailing_stop")
            strengths.append("The trailing stop protected gains after the trade moved in favor.")
        elif close_reason == "time_stop":
            tags.append("time_stop")
            weaknesses.append("The thesis did not resolve fast enough, so capital stayed tied up until the time stop.")
            adjustments.append("Prefer faster-resolving setups when the book is already crowded.")
        elif close_reason == "rl_exit":
            tags.append("rl_exit")
            adjustments.append("Review whether the RL exit arrived early enough or left too much on the table.")

        if holding_minutes is not None:
            if holding_minutes <= 10 and realized_pnl < 0:
                tags.append("fast_invalidation")
                weaknesses.append("The trade invalidated quickly, which usually points to poor timing or stale entry context.")
            elif holding_minutes >= 90 and realized_pnl <= 0:
                tags.append("capital_drag")
                weaknesses.append("The trade consumed time and capital without producing enough return.")

        if p_tp >= 0.6 and expected_return > 0 and realized_pnl < 0:
            tags.append("prediction_miss")
            weaknesses.append("The pre-trade model metrics looked attractive, but the realized outcome missed that prediction.")
            adjustments.append("Penalize similar setups until feedback improves.")
        elif p_tp >= 0.6 and expected_return > 0 and realized_pnl > 0:
            tags.append("prediction_match")
            strengths.append("The realized outcome matched the positive pre-trade model signal.")

        if edge_score <= 0 and realized_pnl <= 0:
            tags.append("weak_edge_confirmed")
            adjustments.append("Continue blocking or shrinking weak-edge setups; the result confirmed the caution.")

        verdict = "The trade aligned well with the model thesis."
        if realized_pnl < 0:
            verdict = "The trade did not validate the entry thesis and should count as negative feedback for similar setups."
        elif abs(realized_pnl) < 1e-9:
            verdict = "The trade was mostly neutral and offers limited signal beyond opportunity-cost feedback."

        return {
            "tags": tags,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "adjustments": adjustments,
            "verdict": verdict,
        }

    def _build_markdown_report(self, payload: dict):
        strengths = payload.get("strengths", []) or []
        weaknesses = payload.get("weaknesses", []) or []
        adjustments = payload.get("adjustments", []) or []
        entry_details = payload.get("entry_details", {}) or {}

        def _lines(items):
            if not items:
                return "- None"
            return "\n".join(f"- {item}" for item in items)

        return "\n".join(
            [
                f"# Trade Feedback Report: {payload.get('market')}",
                "",
                f"- Grade: `{payload.get('grade')}`",
                f"- Outcome: `{payload.get('outcome_class')}`",
                f"- Close reason: `{payload.get('close_reason')}`",
                f"- Realized PnL: `{payload.get('realized_pnl')}`",
                f"- ROI: `{payload.get('roi')}`",
                f"- Holding minutes: `{payload.get('holding_minutes')}`",
                f"- Entry confidence: `{payload.get('confidence_at_entry')}`",
                f"- Signal label: `{payload.get('signal_label')}`",
                f"- Entry model action: `{payload.get('entry_model_action')}`",
                f"- Predicted p_tp: `{payload.get('entry_p_tp_before_sl')}`",
                f"- Predicted expected return: `{payload.get('entry_expected_return')}`",
                f"- Predicted edge score: `{payload.get('entry_edge_score')}`",
                "",
                "## Verdict",
                payload.get("verdict", ""),
                "",
                "## Strengths",
                _lines(strengths),
                "",
                "## Weaknesses",
                _lines(weaknesses),
                "",
                "## Adjustments",
                _lines(adjustments),
                "",
                "## Tags",
                ", ".join(payload.get("learning_tags", []) or []) or "none",
                "",
                "## Entry Details",
                json.dumps(entry_details, indent=2, default=str) if entry_details else "{}",
            ]
        )

    def _compute_feedback_factors(self, group: pd.DataFrame):
        count = len(group.index)
        realized_pnl = self._numeric_series(group, "realized_pnl", default=0.0)
        roi = self._numeric_series(group, "roi", default=0.0)
        win_rate = self._safe_float((realized_pnl > 0).mean(), 0.5)
        avg_roi = self._safe_float(roi.mean(), 0.0)

        confidence_multiplier = np.clip(1.0 + ((win_rate - 0.5) * 0.40) + np.clip(avg_roi, -0.25, 0.25) * 0.20, 0.85, 1.15)
        expected_return_multiplier = np.clip(1.0 + ((win_rate - 0.5) * 0.60) + np.clip(avg_roi, -0.25, 0.25) * 0.35, 0.80, 1.20)
        edge_multiplier = np.clip((confidence_multiplier * 0.45) + (expected_return_multiplier * 0.55), 0.80, 1.20)

        return {
            "sample_count": int(count),
            "win_rate": round(win_rate, 4),
            "avg_roi": round(avg_roi, 6),
            "confidence_multiplier": round(float(confidence_multiplier), 4),
            "expected_return_multiplier": round(float(expected_return_multiplier), 4),
            "edge_multiplier": round(float(edge_multiplier), 4),
        }

    def _refresh_summary(self):
        reports_df = self._safe_read(self.report_csv)
        if reports_df.empty:
            return pd.DataFrame()

        lookback = max(5, int(os.getenv("TRADE_FEEDBACK_LOOKBACK", "50") or 50))
        min_samples = max(2, int(os.getenv("TRADE_FEEDBACK_MIN_SAMPLES", "3") or 3))

        work = reports_df.copy()
        if "closed_at" in work.columns:
            work["closed_at"] = pd.to_datetime(work["closed_at"], utc=True, errors="coerce")
            work = work.sort_values("closed_at")
        recent = work.tail(lookback).copy()
        summary_rows = []

        overall = self._compute_feedback_factors(recent)
        summary_rows.append({
            "scope": "overall",
            "scope_value": "overall",
            **overall,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

        if "signal_label" in recent.columns:
            for signal_label, group in recent.groupby("signal_label", dropna=True):
                if len(group.index) < min_samples:
                    continue
                summary_rows.append({
                    "scope": "signal_label",
                    "scope_value": str(signal_label),
                    **self._compute_feedback_factors(group),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                })

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df.to_csv(self.summary_csv, index=False)
        return summary_df

    def _reset_open_pain_summary(self):
        if self.open_pain_summary_csv.exists():
            try:
                self.open_pain_summary_csv.unlink()
            except Exception:
                pass

    def _compute_open_pain_factors(self, group: pd.DataFrame):
        count = len(group.index)
        if count <= 0:
            return {
                "sample_count": 0,
                "pain_rate": 0.0,
                "avg_open_return": 0.0,
                "avg_mae": 0.0,
                "avg_drawdown": 0.0,
                "avg_fast_adverse_count": 0.0,
                "confidence_multiplier": 1.0,
                "expected_return_multiplier": 1.0,
                "edge_multiplier": 1.0,
                "pain_score": 0.0,
            }

        open_return_trigger = self._env_float("OPEN_PAIN_TRIGGER_OPEN_RETURN", -0.015, minimum=-1.0, maximum=0.0)
        mae_trigger = self._env_float("OPEN_PAIN_TRIGGER_MAE", -0.03, minimum=-1.0, maximum=0.0)
        drawdown_trigger = self._env_float("OPEN_PAIN_TRIGGER_DRAWDOWN", 0.03, minimum=0.0, maximum=1.0)
        fast_count_trigger = self._env_int("OPEN_PAIN_TRIGGER_FAST_COUNT", 1, minimum=0, maximum=100)
        pain_sensitivity = self._env_float("OPEN_PAIN_SENSITIVITY", 1.0, minimum=0.1, maximum=3.0)
        conf_penalty_max = self._env_float("OPEN_PAIN_CONF_PENALTY_MAX", 0.18, minimum=0.0, maximum=0.50)
        ret_penalty_max = self._env_float("OPEN_PAIN_RET_PENALTY_MAX", 0.24, minimum=0.0, maximum=0.60)
        edge_floor = self._env_float("OPEN_PAIN_EDGE_MULTIPLIER_FLOOR", 0.78, minimum=0.50, maximum=1.0)
        conf_floor = self._env_float("OPEN_PAIN_CONF_MULTIPLIER_FLOOR", 0.82, minimum=0.50, maximum=1.0)
        ret_floor = self._env_float("OPEN_PAIN_RET_MULTIPLIER_FLOOR", 0.76, minimum=0.40, maximum=1.0)

        current_return = self._numeric_series(group, "open_return_pct", default=0.0)
        mae = self._numeric_series(group, "max_adverse_excursion_pct", default=0.0)
        drawdown = self._numeric_series_any(
            group,
            ["max_drawdown_from_peak_pct", "drawdown_from_peak"],
            default=0.0,
        )
        fast_count = self._numeric_series(group, "fast_adverse_move_count", default=0.0)

        pain_mask = (
            (current_return <= open_return_trigger)
            | (mae <= mae_trigger)
            | (drawdown >= drawdown_trigger)
            | (fast_count >= fast_count_trigger)
        )
        pain_rate = self._safe_float(pain_mask.mean(), 0.0)
        avg_open_return = self._safe_float(current_return.mean(), 0.0)
        avg_mae = self._safe_float(mae.mean(), 0.0)
        avg_drawdown = self._safe_float(drawdown.mean(), 0.0)
        avg_fast = self._safe_float(fast_count.mean(), 0.0)

        pain_score = float(
            np.clip(
                (
                    (pain_rate * 0.45)
                    + np.clip((-avg_open_return) / max(abs(open_return_trigger) * 4.0, 1e-9), 0.0, 1.0) * 0.20
                    + np.clip((-avg_mae) / max(abs(mae_trigger) * 4.0, 1e-9), 0.0, 1.0) * 0.20
                    + np.clip(avg_drawdown / max(drawdown_trigger * 3.5, 1e-9), 0.0, 1.0) * 0.10
                    + np.clip(avg_fast / max(fast_count_trigger + 1.0, 1.0), 0.0, 1.0) * 0.05
                ) * pain_sensitivity,
                0.0,
                1.0,
            )
        )

        confidence_multiplier = np.clip(1.0 - (pain_score * conf_penalty_max), conf_floor, 1.0)
        expected_return_multiplier = np.clip(1.0 - (pain_score * ret_penalty_max), ret_floor, 1.0)
        edge_multiplier = np.clip((confidence_multiplier * 0.45) + (expected_return_multiplier * 0.55), edge_floor, 1.0)

        return {
            "sample_count": int(count),
            "pain_rate": round(pain_rate, 4),
            "avg_open_return": round(avg_open_return, 6),
            "avg_mae": round(avg_mae, 6),
            "avg_drawdown": round(avg_drawdown, 6),
            "avg_fast_adverse_count": round(avg_fast, 4),
            "confidence_multiplier": round(float(confidence_multiplier), 4),
            "expected_return_multiplier": round(float(expected_return_multiplier), 4),
            "edge_multiplier": round(float(edge_multiplier), 4),
            "pain_score": round(float(pain_score), 4),
            "pain_sensitivity": round(float(pain_sensitivity), 4),
            "pain_open_return_trigger": round(float(open_return_trigger), 6),
            "pain_mae_trigger": round(float(mae_trigger), 6),
            "pain_drawdown_trigger": round(float(drawdown_trigger), 6),
            "pain_fast_count_trigger": int(fast_count_trigger),
        }

    def _refresh_open_pain_summary(self):
        positions_df = self._safe_read(self.positions_csv)
        if positions_df.empty:
            self._reset_open_pain_summary()
            return pd.DataFrame()

        work = positions_df.copy()
        if "status" in work.columns:
            work = work[work["status"].astype(str).str.strip().str.upper() == "OPEN"].copy()
        if work.empty:
            self._reset_open_pain_summary()
            return pd.DataFrame()

        entry_price = self._numeric_series(work, "entry_price", default=0.0)
        current_price = self._numeric_series_any(work, ["current_price"], default=np.nan).fillna(entry_price)
        work["open_return_pct"] = np.where(entry_price > 0, (current_price - entry_price) / entry_price, 0.0)
        work["max_adverse_excursion_pct"] = self._numeric_series(work, "max_adverse_excursion_pct", default=0.0)
        work["max_drawdown_from_peak_pct"] = self._numeric_series_any(
            work,
            ["max_drawdown_from_peak_pct", "drawdown_from_peak"],
            default=0.0,
        )
        work["fast_adverse_move_count"] = self._numeric_series(work, "fast_adverse_move_count", default=0.0)

        rows = [
            {
                "scope": "overall",
                "scope_value": "overall",
                **self._compute_open_pain_factors(work),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        ]

        if "signal_label" in work.columns:
            for signal_label, group in work.groupby("signal_label", dropna=True):
                if len(group.index) < 1:
                    continue
                rows.append(
                    {
                        "scope": "signal_label",
                        "scope_value": str(signal_label),
                        **self._compute_open_pain_factors(group),
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

        pain_df = pd.DataFrame(rows)
        if pain_df.empty:
            self._reset_open_pain_summary()
            return pain_df
        pain_df.to_csv(self.open_pain_summary_csv, index=False)
        return pain_df

    def _closed_row_to_trade(self, row: dict):
        exit_price = row.get("exit_price", row.get("current_price"))
        return SimpleNamespace(
            position_id=row.get("position_id"),
            market=row.get("market") or row.get("market_title"),
            token_id=row.get("token_id"),
            condition_id=row.get("condition_id"),
            outcome_side=row.get("outcome_side"),
            signal_label=row.get("signal_label"),
            confidence_at_entry=self._safe_float(row.get("confidence_at_entry", row.get("confidence", 0.0)), 0.0),
            entry_price=self._safe_float(row.get("entry_price"), 0.0),
            current_price=self._safe_float(exit_price, self._safe_float(row.get("current_price"), 0.0)),
            realized_pnl=self._safe_float(row.get("realized_pnl"), 0.0),
            size_usdc=self._safe_float(row.get("size_usdc"), 0.0),
            shares=self._safe_float(row.get("shares"), 0.0),
            opened_at=row.get("opened_at"),
            closed_at=row.get("closed_at"),
            close_reason=row.get("close_reason"),
            close_fingerprint=row.get("close_fingerprint"),
            state="CLOSED",
        )

    def backfill_from_closed_positions_csv(self, include_reconciliation=False):
        closed_path = self.logs_dir / "closed_positions.csv"
        closed_df = self._safe_read(closed_path)
        if closed_df.empty:
            return {"csv_rows": 0, "processed_reports": 0}

        trades = []
        for _, row in closed_df.iterrows():
            close_reason = str(row.get("close_reason", "") or "").strip().lower()
            if not include_reconciliation and close_reason == "external_manual_close":
                continue
            trades.append(self._closed_row_to_trade(row.to_dict()))

        processed = self.record_closed_trades(trades)
        return {"csv_rows": len(closed_df.index), "processed_reports": processed}

    def _relabel(self, row: dict, labels: dict):
        confidence = self._safe_float(row.get("confidence"), 0.0)
        if confidence < 0.45:
            action_code = 0
        elif confidence < 0.60:
            action_code = 1
        elif confidence < 0.78:
            action_code = 2
        else:
            action_code = 3
        row["action_code"] = action_code
        row["signal_label"] = labels.get(action_code, "IGNORE")
        return row

    def apply_to_scored_df(self, scored_df: pd.DataFrame, signal_engine):
        if scored_df is None or scored_df.empty or signal_engine is None:
            return scored_df

        pain_summary_df = self._refresh_open_pain_summary()
        summary_df = self._safe_read(self.summary_csv)
        overall_row = {}
        if not summary_df.empty:
            overall = summary_df[
                (summary_df.get("scope") == "overall") & (summary_df.get("scope_value") == "overall")
            ]
            if not overall.empty:
                overall_row = overall.iloc[0].to_dict()

        label_rows = {}
        label_df = summary_df[summary_df.get("scope") == "signal_label"] if "scope" in summary_df.columns else pd.DataFrame()
        if not label_df.empty:
            label_rows = {
                str(row.get("scope_value")): row.to_dict()
                for _, row in label_df.iterrows()
            }

        pain_overall_row = {}
        pain_label_rows = {}
        if not pain_summary_df.empty:
            overall_pain = pain_summary_df[
                (pain_summary_df.get("scope") == "overall") & (pain_summary_df.get("scope_value") == "overall")
            ]
            if not overall_pain.empty:
                pain_overall_row = overall_pain.iloc[0].to_dict()
            pain_label_df = pain_summary_df[pain_summary_df.get("scope") == "signal_label"] if "scope" in pain_summary_df.columns else pd.DataFrame()
            if not pain_label_df.empty:
                pain_label_rows = {
                    str(row.get("scope_value")): row.to_dict()
                    for _, row in pain_label_df.iterrows()
                }

        adjusted_rows = []
        for _, row in scored_df.iterrows():
            working = row.to_dict()
            existing_label = str(working.get("signal_label", "") or "")
            label_ctx = label_rows.get(existing_label, {})
            pain_label_ctx = pain_label_rows.get(existing_label, {})

            conf_mult = self._safe_float(overall_row.get("confidence_multiplier"), 1.0) * self._safe_float(label_ctx.get("confidence_multiplier"), 1.0)
            ret_mult = self._safe_float(overall_row.get("expected_return_multiplier"), 1.0) * self._safe_float(label_ctx.get("expected_return_multiplier"), 1.0)
            edge_mult = self._safe_float(overall_row.get("edge_multiplier"), 1.0) * self._safe_float(label_ctx.get("edge_multiplier"), 1.0)
            pain_conf_mult = self._safe_float(pain_overall_row.get("confidence_multiplier"), 1.0) * self._safe_float(pain_label_ctx.get("confidence_multiplier"), 1.0)
            pain_ret_mult = self._safe_float(pain_overall_row.get("expected_return_multiplier"), 1.0) * self._safe_float(pain_label_ctx.get("expected_return_multiplier"), 1.0)
            pain_edge_mult = self._safe_float(pain_overall_row.get("edge_multiplier"), 1.0) * self._safe_float(pain_label_ctx.get("edge_multiplier"), 1.0)

            conf_mult *= pain_conf_mult
            ret_mult *= pain_ret_mult
            edge_mult *= pain_edge_mult

            working["pre_feedback_confidence"] = self._safe_float(working.get("confidence"), 0.0)
            working["pre_feedback_expected_return"] = self._safe_float(working.get("expected_return"), 0.0)
            working["feedback_confidence_multiplier"] = round(conf_mult, 4)
            working["feedback_expected_return_multiplier"] = round(ret_mult, 4)
            working["feedback_edge_multiplier"] = round(edge_mult, 4)
            working["feedback_recent_win_rate"] = self._safe_float(label_ctx.get("win_rate"), self._safe_float(overall_row.get("win_rate"), 0.5))
            working["feedback_open_pain_score"] = self._safe_float(pain_label_ctx.get("pain_score"), self._safe_float(pain_overall_row.get("pain_score"), 0.0))
            working["feedback_open_pain_rate"] = self._safe_float(pain_label_ctx.get("pain_rate"), self._safe_float(pain_overall_row.get("pain_rate"), 0.0))

            working["expected_return"] = self._safe_float(working.get("expected_return"), 0.0) * ret_mult
            working["edge_score"] = self._safe_float(working.get("edge_score"), 0.0) * edge_mult

            rescored = signal_engine.score_row(working)
            p_tp = self._safe_float(rescored.get("p_tp_before_sl"), 0.0)
            exp_ret = self._safe_float(rescored.get("expected_return"), 0.0)
            edge_score = self._safe_float(rescored.get("edge_score"), 0.0)
            confidence_cap = 1.0
            if exp_ret <= 0 or edge_score <= 0 or p_tp < 0.52:
                confidence_cap = min(confidence_cap, 0.59)
            if exp_ret < 0 and p_tp < 0.48:
                confidence_cap = min(confidence_cap, 0.44)

            rescored["confidence"] = round(min(confidence_cap, max(0.0, self._safe_float(rescored.get("confidence"), 0.0) * conf_mult)), 4)
            rescored = self._relabel(rescored, signal_engine.LABELS)
            rescored["reason"] = (
                f"{rescored.get('reason')} | "
                f"feedback_wr={self._safe_float(rescored.get('feedback_recent_win_rate'), 0.5):.2f}, "
                f"feedback_conf_mult={self._safe_float(conf_mult, 1.0):.2f}, "
                f"feedback_ret_mult={self._safe_float(ret_mult, 1.0):.2f}, "
                f"open_pain={self._safe_float(rescored.get('feedback_open_pain_score'), 0.0):.2f}"
            )
            adjusted_rows.append(rescored)

        adjusted_df = pd.DataFrame(adjusted_rows)
        if not adjusted_df.empty:
            logging.info(
                "Trade feedback applied to %s scored rows (overall_wr=%.2f, conf_mult=%.2f, ret_mult=%.2f, open_pain=%.2f).",
                len(adjusted_df),
                self._safe_float(overall_row.get("win_rate"), 0.5),
                self._safe_float(overall_row.get("confidence_multiplier"), 1.0),
                self._safe_float(overall_row.get("expected_return_multiplier"), 1.0),
                self._safe_float(pain_overall_row.get("pain_score"), 0.0),
            )
        return adjusted_df if not adjusted_df.empty else scored_df

    def record_closed_trades(self, closed_trades):
        if not closed_trades:
            return 0

        existing_reports_df = self._safe_read(self.report_csv)
        existing_report_ids = set()
        if not existing_reports_df.empty and "report_id" in existing_reports_df.columns:
            existing_report_ids = {
                str(report_id)
                for report_id in existing_reports_df["report_id"].dropna().astype(str).tolist()
                if str(report_id).strip()
            }

        report_rows = []
        processed = 0
        for trade in closed_trades:
            if str(getattr(trade, "state", "")).upper().endswith("OPEN"):
                continue
            close_reason = str(getattr(trade, "close_reason", "") or "").strip().lower()
            if close_reason == "external_manual_close":
                logging.info(
                    "Skipping learning report for externally reconciled close token=%s.",
                    str(getattr(trade, "token_id", "") or "")[:16],
                )
                continue

            entry_price = self._safe_float(getattr(trade, "entry_price", 0.0), 0.0)
            exit_price = self._safe_float(getattr(trade, "current_price", 0.0), entry_price)
            realized_pnl = self._safe_float(getattr(trade, "realized_pnl", 0.0), 0.0)
            size_usdc = self._safe_float(getattr(trade, "size_usdc", 0.0), 0.0)
            shares = self._safe_float(getattr(trade, "shares", 0.0), 0.0)
            opened_at = self._safe_iso(getattr(trade, "opened_at", None))
            closed_at = self._safe_iso(getattr(trade, "closed_at", None))
            opened_ts = pd.to_datetime(opened_at, utc=True, errors="coerce")
            closed_ts = pd.to_datetime(closed_at, utc=True, errors="coerce")
            holding_minutes = None
            if pd.notna(opened_ts) and pd.notna(closed_ts):
                holding_minutes = round(max(0.0, (closed_ts - opened_ts).total_seconds() / 60.0), 4)

            roi = 0.0
            if entry_price > 0:
                roi = (exit_price - entry_price) / entry_price
            outcome_class = "breakeven"
            if realized_pnl > 0:
                outcome_class = "win"
            elif realized_pnl < 0:
                outcome_class = "loss"

            candidate_ctx, model_ctx = self._lookup_entry_context(trade)
            report_id = self._trade_report_id(trade)
            if report_id in existing_report_ids:
                continue

            entry_details = self._safe_json_load(candidate_ctx.get("details_json"), {})
            confidence_at_entry = self._safe_float(
                getattr(trade, "confidence_at_entry", candidate_ctx.get("confidence", 0.0)),
                0.0,
            )
            signal_label = (
                getattr(trade, "signal_label", None)
                or candidate_ctx.get("model_action")
                or model_ctx.get("action")
            )
            grade = self._grade_trade(realized_pnl, roi)
            takeaways = self._extract_takeaways(
                realized_pnl=realized_pnl,
                roi=roi,
                close_reason=getattr(trade, "close_reason", None),
                confidence_at_entry=confidence_at_entry,
                holding_minutes=holding_minutes,
                entry_candidate=candidate_ctx,
            )
            report_payload = {
                "report_id": report_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "position_id": getattr(trade, "position_id", None),
                "close_fingerprint": getattr(trade, "close_fingerprint", None),
                "market": getattr(trade, "market", None),
                "token_id": getattr(trade, "token_id", None),
                "condition_id": getattr(trade, "condition_id", None),
                "outcome_side": getattr(trade, "outcome_side", None),
                "signal_label": signal_label,
                "confidence_at_entry": confidence_at_entry,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "size_usdc": size_usdc,
                "shares": shares,
                "opened_at": opened_at,
                "closed_at": closed_at,
                "holding_minutes": holding_minutes,
                "close_reason": getattr(trade, "close_reason", None),
                "realized_pnl": round(realized_pnl, 6),
                "roi": round(roi, 6),
                "outcome_class": outcome_class,
                "grade": grade,
                "entry_candidate": candidate_ctx,
                "entry_details": entry_details,
                "latest_model_decision": model_ctx,
                "entry_model_action": candidate_ctx.get("model_action"),
                "entry_market_slug": candidate_ctx.get("market_slug"),
                "entry_trader_wallet": candidate_ctx.get("trader_wallet"),
                "entry_p_tp_before_sl": self._safe_float(candidate_ctx.get("p_tp_before_sl"), 0.0),
                "entry_expected_return": self._safe_float(candidate_ctx.get("expected_return"), 0.0),
                "entry_edge_score": self._safe_float(candidate_ctx.get("edge_score"), 0.0),
                "entry_confidence": self._safe_float(candidate_ctx.get("confidence"), confidence_at_entry),
                "latest_model_action": model_ctx.get("action"),
                "latest_model_score": self._safe_float(model_ctx.get("score"), 0.0),
                "prediction_alignment": "correct" if realized_pnl > 0 else "incorrect" if realized_pnl < 0 else "flat",
                "learning_tags": takeaways.get("tags", []),
                "strengths": takeaways.get("strengths", []),
                "weaknesses": takeaways.get("weaknesses", []),
                "adjustments": takeaways.get("adjustments", []),
                "verdict": takeaways.get("verdict", ""),
            }

            report_file = self.reports_dir / f"{report_id}.json"
            markdown_file = self.reports_dir / f"{report_id}.md"
            try:
                report_file.write_text(json.dumps(report_payload, indent=2, default=str), encoding="utf-8")
            except Exception as exc:
                logging.warning("Failed to write trade feedback report %s: %s", report_id, exc)
            try:
                markdown_file.write_text(self._build_markdown_report(report_payload), encoding="utf-8")
            except Exception as exc:
                logging.warning("Failed to write markdown trade feedback report %s: %s", report_id, exc)

            report_rows.append(
                {
                    "report_id": report_id,
                    "generated_at": report_payload["generated_at"],
                    "position_id": report_payload["position_id"],
                    "close_fingerprint": report_payload["close_fingerprint"],
                    "market": report_payload["market"],
                    "token_id": report_payload["token_id"],
                    "condition_id": report_payload["condition_id"],
                    "outcome_side": report_payload["outcome_side"],
                    "signal_label": report_payload["signal_label"],
                    "confidence_at_entry": report_payload["confidence_at_entry"],
                    "entry_price": report_payload["entry_price"],
                    "exit_price": report_payload["exit_price"],
                    "size_usdc": report_payload["size_usdc"],
                    "shares": report_payload["shares"],
                    "opened_at": report_payload["opened_at"],
                    "closed_at": report_payload["closed_at"],
                    "holding_minutes": report_payload["holding_minutes"],
                    "close_reason": report_payload["close_reason"],
                    "realized_pnl": report_payload["realized_pnl"],
                    "roi": report_payload["roi"],
                    "outcome_class": report_payload["outcome_class"],
                    "grade": report_payload["grade"],
                    "prediction_alignment": report_payload["prediction_alignment"],
                    "entry_model_action": candidate_ctx.get("model_action"),
                    "entry_p_tp_before_sl": self._safe_float(candidate_ctx.get("p_tp_before_sl"), 0.0),
                    "entry_expected_return": self._safe_float(candidate_ctx.get("expected_return"), 0.0),
                    "entry_edge_score": self._safe_float(candidate_ctx.get("edge_score"), 0.0),
                    "entry_confidence": self._safe_float(candidate_ctx.get("confidence"), report_payload["confidence_at_entry"]),
                    "latest_model_action": model_ctx.get("action"),
                    "latest_model_score": self._safe_float(model_ctx.get("score"), 0.0),
                    "verdict": report_payload["verdict"],
                    "learning_tags": "|".join(report_payload["learning_tags"]),
                    "strength_count": len(report_payload["strengths"]),
                    "weakness_count": len(report_payload["weaknesses"]),
                    "adjustment_count": len(report_payload["adjustments"]),
                }
            )
            existing_report_ids.add(report_id)
            processed += 1

        if report_rows:
            report_df = pd.DataFrame(report_rows)
            existing_report_df = self._safe_read(self.report_csv)
            ordered_cols = list(existing_report_df.columns)
            for column in report_df.columns:
                if column not in ordered_cols:
                    ordered_cols.append(column)
            existing_report_df = existing_report_df.reindex(columns=ordered_cols)
            report_df = report_df.reindex(columns=ordered_cols)
            merged = report_df.copy() if existing_report_df.empty else pd.concat([existing_report_df, report_df], ignore_index=True)
            if "report_id" in merged.columns:
                merged = merged.drop_duplicates(subset=["report_id"], keep="last")
            merged.to_csv(self.report_csv, index=False)
            self._refresh_summary()
            logging.info("Recorded %s trade feedback reports.", processed)
        return processed
