from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from trade_quality import build_quality_context, classify_exit_reason_family


class TradeLifecycleAuditor:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.closed_file = self.logs_dir / "closed_positions.csv"
        self.audit_file = self.logs_dir / "trade_lifecycle_audit.csv"
        self.ablation_file = self.logs_dir / "feature_ablation_report.csv"

    def _safe_read(self, path: Path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def build_reports(self):
        closed_df = self._safe_read(self.closed_file)
        if closed_df.empty:
            return {"audit_rows": 0, "ablation_rows": 0}

        work = closed_df.copy()
        for field in [
            "exit_reason_family",
            "operational_close_flag",
            "entry_context_complete",
            "learning_eligible",
            "market_family",
            "horizon_bucket",
            "liquidity_bucket",
            "volatility_bucket",
            "technical_regime_bucket",
        ]:
            if field not in work.columns:
                work[field] = None

        for idx, row in work.iterrows():
            enriched = build_quality_context(row.to_dict())
            for key, value in enriched.items():
                if pd.isna(work.at[idx, key]) or work.at[idx, key] in [None, "", "nan"]:
                    work.at[idx, key] = value

        if "closed_at" in work.columns:
            work["closed_at"] = pd.to_datetime(work["closed_at"], utc=True, errors="coerce")
            work = work.sort_values("closed_at")

        pnl_col = "net_realized_pnl" if "net_realized_pnl" in work.columns else "realized_pnl"
        work[pnl_col] = pd.to_numeric(work.get(pnl_col), errors="coerce").fillna(0.0)
        signal_label = work.get("signal_label", pd.Series("", index=work.index)).fillna("").astype(str).str.strip().str.upper()
        learning_eligible = work.get("learning_eligible", pd.Series(False, index=work.index)).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        operational = work.get("operational_close_flag", pd.Series(False, index=work.index)).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        complete = work.get("entry_context_complete", pd.Series(False, index=work.index)).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        exit_family = work.get("exit_reason_family", pd.Series("", index=work.index)).fillna("").astype(str)
        if exit_family.eq("").all():
            exit_family = work.get("close_reason", pd.Series("", index=work.index)).fillna("").astype(str).map(classify_exit_reason_family)

        audit_row = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "date": datetime.now(timezone.utc).date().isoformat(),
            "total_closed_trades": int(len(work)),
            "learning_eligible_ratio": round(float(learning_eligible.mean()), 4),
            "entry_context_complete_ratio": round(float(complete.mean()), 4),
            "unknown_signal_label_ratio": round(float(signal_label.isin({"", "UNKNOWN"}).mean()), 4),
            "operational_close_ratio": round(float(operational.mean()), 4),
            "external_manual_close_ratio": round(float(work.get("close_reason", pd.Series("", index=work.index)).fillna("").astype(str).str.lower().eq("external_manual_close").mean()), 4),
            "top_exit_reason_families": json.dumps(exit_family.value_counts().head(5).to_dict(), separators=(",", ":")),
        }
        pd.DataFrame([audit_row]).to_csv(self.audit_file, mode="a", header=not self.audit_file.exists(), index=False)

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
