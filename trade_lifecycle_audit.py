from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from trade_quality import classify_exit_reason_family, enrich_quality_frame


class TradeLifecycleAuditor:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.closed_file = self.logs_dir / "closed_positions.csv"
        self.audit_file = self.logs_dir / "trade_lifecycle_audit.csv"
        self.ablation_file = self.logs_dir / "trade_quality_ablation_report.csv"

    def _safe_read(self, path: Path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def _append_csv_row(self, path: Path, row: dict, sort_by: str | None = None):
        existing = self._safe_read(path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        if sort_by and sort_by in updated.columns:
            try:
                sort_key = pd.to_datetime(updated[sort_by], errors="coerce", utc=True)
                updated = (
                    updated.assign(_sort_key=sort_key)
                    .sort_values("_sort_key", na_position="first")
                    .drop(columns=["_sort_key"])
                )
            except Exception:
                pass
        updated.to_csv(path, index=False)
        return updated

    def build_reports(self):
        closed_df = self._safe_read(self.closed_file)
        if closed_df.empty:
            return {"audit_rows": 0, "ablation_rows": 0}

        work = enrich_quality_frame(closed_df, logs_dir=self.logs_dir)

        if "closed_at" in work.columns:
            work["closed_at"] = pd.to_datetime(work["closed_at"], utc=True, errors="coerce")
            work = work.sort_values("closed_at")

        pnl_col = "net_realized_pnl" if "net_realized_pnl" in work.columns else "realized_pnl"
        work[pnl_col] = pd.to_numeric(work.get(pnl_col), errors="coerce").fillna(0.0)
        signal_label = work.get("signal_label", pd.Series("", index=work.index)).fillna("").astype(str).str.strip().str.upper()
        learning_eligible = work.get("learning_eligible", pd.Series(False, index=work.index)).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        operational = work.get("operational_close_flag", pd.Series(False, index=work.index)).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        reconciliation = work.get("reconciliation_close_flag", pd.Series(False, index=work.index)).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        complete = work.get("entry_context_complete", pd.Series(False, index=work.index)).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        exit_family = work.get("exit_reason_family", pd.Series("", index=work.index)).fillna("").astype(str)
        if exit_family.eq("").all():
            exit_family = work.get("close_reason", pd.Series("", index=work.index)).fillna("").astype(str).map(classify_exit_reason_family)
        quality_scope = work[~reconciliation].copy()
        quality_scope_signal = quality_scope.get("signal_label", pd.Series("", index=quality_scope.index)).fillna("").astype(str).str.strip().str.upper()
        quality_scope_complete = quality_scope.get("entry_context_complete", pd.Series(False, index=quality_scope.index)).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        quality_scope_learning = quality_scope.get("learning_eligible", pd.Series(False, index=quality_scope.index)).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        quality_scope_operational = quality_scope.get("operational_close_flag", pd.Series(False, index=quality_scope.index)).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        recent_window = min(200, len(work.index))
        recent = work.tail(recent_window).copy()
        recent_signal = recent.get("signal_label", pd.Series("", index=recent.index)).fillna("").astype(str).str.strip().str.upper()
        recent_operational = recent.get("operational_close_flag", pd.Series(False, index=recent.index)).astype(bool)
        recent_reconciliation = recent.get("reconciliation_close_flag", pd.Series(False, index=recent.index)).astype(bool)
        recent_quality_scope = recent[~recent_reconciliation].copy()
        recent_quality_signal = recent_quality_scope.get("signal_label", pd.Series("", index=recent_quality_scope.index)).fillna("").astype(str).str.strip().str.upper()
        recent_quality_complete = recent_quality_scope.get("entry_context_complete", pd.Series(False, index=recent_quality_scope.index)).astype(bool)
        recent_quality_learning = recent_quality_scope.get("learning_eligible", pd.Series(False, index=recent_quality_scope.index)).astype(bool)
        recent_quality_operational = recent_quality_scope.get("operational_close_flag", pd.Series(False, index=recent_quality_scope.index)).astype(bool)
        recent_non_operational = recent_quality_scope[~recent_quality_operational].copy()
        recent_non_operational_signal = recent_non_operational.get(
            "signal_label",
            pd.Series("", index=recent_non_operational.index),
        ).fillna("").astype(str).str.strip().str.upper()

        audit_row = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "date": datetime.now(timezone.utc).date().isoformat(),
            "total_closed_trades": int(len(work)),
            "quality_scope_closed_trades": int(len(quality_scope.index)),
            "reconciliation_close_ratio": round(float(reconciliation.mean()), 4),
            "learning_eligible_ratio": round(float(quality_scope_learning.mean()), 4) if len(quality_scope.index) else 0.0,
            "entry_context_complete_ratio": round(float(quality_scope_complete.mean()), 4) if len(quality_scope.index) else 0.0,
            "unknown_signal_label_ratio": round(float(quality_scope_signal.isin({"", "UNKNOWN"}).mean()), 4) if len(quality_scope.index) else 0.0,
            "operational_close_ratio": round(float(quality_scope_operational.mean()), 4) if len(quality_scope.index) else 0.0,
            "external_manual_close_ratio": round(float(work.get("close_reason", pd.Series("", index=work.index)).fillna("").astype(str).str.lower().eq("external_manual_close").mean()), 4),
            "recent_window": int(recent_window),
            "recent_quality_scope_closed_trades": int(len(recent_quality_scope.index)),
            "recent_reconciliation_close_ratio": round(float(recent_reconciliation.mean()), 4) if len(recent.index) else 0.0,
            "recent_learning_eligible_ratio": round(float(recent_quality_learning.mean()), 4) if len(recent_quality_scope.index) else 0.0,
            "recent_entry_context_complete_ratio": round(float(recent_quality_complete.mean()), 4) if len(recent_quality_scope.index) else 0.0,
            "recent_unknown_signal_label_ratio": round(float(recent_quality_signal.isin({"", "UNKNOWN"}).mean()), 4) if len(recent_quality_scope.index) else 0.0,
            "recent_non_operational_unknown_signal_label_ratio": round(float(recent_non_operational_signal.isin({"", "UNKNOWN"}).mean()), 4) if len(recent_non_operational.index) else 0.0,
            "recent_operational_close_ratio": round(float(recent_quality_operational.mean()), 4) if len(recent_quality_scope.index) else 0.0,
            "top_exit_reason_families": json.dumps(exit_family.value_counts().head(5).to_dict(), separators=(",", ":")),
        }
        self._append_csv_row(self.audit_file, audit_row, sort_by="generated_at")

        learning_work = work[learning_eligible].copy()
        ablation_rows = []
        if not learning_work.empty:
            for scope in ["signal_label", "market_family", "horizon_bucket", "technical_regime_bucket", "exit_reason_family", "performance_governor_level"]:
                if scope not in learning_work.columns:
                    continue
                grouped = learning_work.groupby(scope, dropna=True)[pnl_col].agg(["size", "mean"])
                for scope_value, values in grouped.iterrows():
                    series = learning_work[learning_work[scope] == scope_value][pnl_col]
                    ablation_rows.append(
                        {
                            "generated_at": audit_row["generated_at"],
                            "scope": scope,
                            "scope_value": str(scope_value),
                            "trades": int(values["size"]),
                            "average_pnl": round(float(values["mean"]), 6),
                            "win_rate": round(float((series > 0).mean()), 4),
                            "profit_factor": round(float(series[series > 0].sum() / max(abs(series[series < 0].sum()), 1e-9)), 6),
                        }
                    )
        if ablation_rows:
            pd.DataFrame(ablation_rows).to_csv(self.ablation_file, index=False)
        return {"audit_rows": 1, "ablation_rows": len(ablation_rows)}
