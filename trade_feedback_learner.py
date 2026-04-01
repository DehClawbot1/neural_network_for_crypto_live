import json
import logging
import os
import re
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
        self.db = Database()

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
        closed_at = self._safe_iso(getattr(trade, "closed_at", None)) or datetime.now(timezone.utc).isoformat()
        closed_stamp = re.sub(r"[^0-9]", "", closed_at)[:14] or "unknown"
        return f"{closed_stamp}_{token_id}_{outcome_side}"

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

    def _compute_feedback_factors(self, group: pd.DataFrame):
        count = len(group.index)
        win_rate = self._safe_float((pd.to_numeric(group.get("realized_pnl"), errors="coerce").fillna(0.0) > 0).mean(), 0.5)
        avg_roi = self._safe_float(pd.to_numeric(group.get("roi"), errors="coerce").fillna(0.0).mean(), 0.0)

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

        summary_df = self._safe_read(self.summary_csv)
        if summary_df.empty:
            return scored_df

        overall_row = summary_df[
            (summary_df.get("scope") == "overall") & (summary_df.get("scope_value") == "overall")
        ]
        if overall_row.empty:
            return scored_df
        overall_row = overall_row.iloc[0].to_dict()

        label_rows = {}
        label_df = summary_df[summary_df.get("scope") == "signal_label"] if "scope" in summary_df.columns else pd.DataFrame()
        if not label_df.empty:
            label_rows = {
                str(row.get("scope_value")): row.to_dict()
                for _, row in label_df.iterrows()
            }

        adjusted_rows = []
        for _, row in scored_df.iterrows():
            working = row.to_dict()
            existing_label = str(working.get("signal_label", "") or "")
            label_ctx = label_rows.get(existing_label, {})

            conf_mult = self._safe_float(overall_row.get("confidence_multiplier"), 1.0) * self._safe_float(label_ctx.get("confidence_multiplier"), 1.0)
            ret_mult = self._safe_float(overall_row.get("expected_return_multiplier"), 1.0) * self._safe_float(label_ctx.get("expected_return_multiplier"), 1.0)
            edge_mult = self._safe_float(overall_row.get("edge_multiplier"), 1.0) * self._safe_float(label_ctx.get("edge_multiplier"), 1.0)

            working["pre_feedback_confidence"] = self._safe_float(working.get("confidence"), 0.0)
            working["pre_feedback_expected_return"] = self._safe_float(working.get("expected_return"), 0.0)
            working["feedback_confidence_multiplier"] = round(conf_mult, 4)
            working["feedback_expected_return_multiplier"] = round(ret_mult, 4)
            working["feedback_edge_multiplier"] = round(edge_mult, 4)
            working["feedback_recent_win_rate"] = self._safe_float(label_ctx.get("win_rate"), self._safe_float(overall_row.get("win_rate"), 0.5))

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
                f"feedback_ret_mult={self._safe_float(ret_mult, 1.0):.2f}"
            )
            adjusted_rows.append(rescored)

        adjusted_df = pd.DataFrame(adjusted_rows)
        if not adjusted_df.empty:
            logging.info(
                "Trade feedback applied to %s scored rows (overall_wr=%.2f, conf_mult=%.2f, ret_mult=%.2f).",
                len(adjusted_df),
                self._safe_float(overall_row.get("win_rate"), 0.5),
                self._safe_float(overall_row.get("confidence_multiplier"), 1.0),
                self._safe_float(overall_row.get("expected_return_multiplier"), 1.0),
            )
        return adjusted_df if not adjusted_df.empty else scored_df

    def record_closed_trades(self, closed_trades):
        if not closed_trades:
            return 0

        report_rows = []
        processed = 0
        for trade in closed_trades:
            if str(getattr(trade, "state", "")).upper().endswith("OPEN"):
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
            report_payload = {
                "report_id": report_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "market": getattr(trade, "market", None),
                "token_id": getattr(trade, "token_id", None),
                "condition_id": getattr(trade, "condition_id", None),
                "outcome_side": getattr(trade, "outcome_side", None),
                "signal_label": getattr(trade, "signal_label", None),
                "confidence_at_entry": self._safe_float(getattr(trade, "confidence_at_entry", 0.0), 0.0),
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
                "entry_candidate": candidate_ctx,
                "latest_model_decision": model_ctx,
                "prediction_alignment": "correct" if realized_pnl > 0 else "incorrect" if realized_pnl < 0 else "flat",
            }

            report_file = self.reports_dir / f"{report_id}.json"
            try:
                report_file.write_text(json.dumps(report_payload, indent=2, default=str), encoding="utf-8")
            except Exception as exc:
                logging.warning("Failed to write trade feedback report %s: %s", report_id, exc)

            report_rows.append(
                {
                    "report_id": report_id,
                    "generated_at": report_payload["generated_at"],
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
                    "prediction_alignment": report_payload["prediction_alignment"],
                    "entry_model_action": candidate_ctx.get("model_action"),
                    "entry_p_tp_before_sl": self._safe_float(candidate_ctx.get("p_tp_before_sl"), 0.0),
                    "entry_expected_return": self._safe_float(candidate_ctx.get("expected_return"), 0.0),
                    "entry_edge_score": self._safe_float(candidate_ctx.get("edge_score"), 0.0),
                    "entry_confidence": self._safe_float(candidate_ctx.get("confidence"), report_payload["confidence_at_entry"]),
                    "latest_model_action": model_ctx.get("action"),
                    "latest_model_score": self._safe_float(model_ctx.get("score"), 0.0),
                }
            )
            processed += 1

        if report_rows:
            pd.DataFrame(report_rows).to_csv(
                self.report_csv,
                mode="a",
                header=not self.report_csv.exists(),
                index=False,
            )
            self._refresh_summary()
            logging.info("Recorded %s trade feedback reports.", processed)
        return processed
