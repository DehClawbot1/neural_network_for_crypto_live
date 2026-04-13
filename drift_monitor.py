"""
drift_monitor.py
Phase 11 — Active Drift Monitor

Operates beyond reactive PnL governors by identifying structural shifts
in feature-spaces and probability matrices before PnL collapse.
Outputs multi-band severities with required persistence trackers.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Severity Thresholds (Score thresholds mapping to strings)
_THRESHOLDS = {
    "feature_drift":  {"low": 0.10, "medium": 0.15, "high": 0.20, "critical": 0.25},
    "calibration":    {"low": 0.02, "medium": 0.04, "high": 0.06, "critical": 0.08},
    "precision":      {"low": 0.05, "medium": 0.10, "high": 0.15, "critical": 0.20},
    "execution":      {"low": 0.01, "medium": 0.03, "high": 0.05, "critical": 0.08}, # Slippage & Fill error
    "edge_decay":     {"low": 0.02, "medium": 0.04, "high": 0.06, "critical": 0.10}, # EV extraction mismatch
    "schema_health":  {"low": 0.05, "medium": 0.10, "high": 0.20, "critical": 0.35}, # Missing/fallback rates
}

def _get_severity(drift_type: str, score: float) -> str:
    """Map continuous score into severity bands."""
    bands = _THRESHOLDS.get(drift_type, {})
    if not bands:
        return "none"
        
    s = abs(score)
    if s >= bands["critical"]: return "critical"
    if s >= bands["high"]: return "high"
    if s >= bands["medium"]: return "medium"
    if s >= bands["low"]: return "low"
    return "none"


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        num = float(value)
        return float(default) if not np.isfinite(num) else num
    except Exception:
        return float(default)


def _psi_for_column(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if len(ref) < 10 or len(cur) < 10:
        return 0.0

    bins = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(0.0, psi)


def _brier_score(prob_pred: pd.Series, actual: pd.Series) -> float:
    p = pd.to_numeric(prob_pred, errors="coerce").fillna(0.5)
    y = pd.to_numeric(actual, errors="coerce").fillna(0.5)
    return float(((p - y) ** 2).mean())


class DriftMonitor:
    def __init__(self, logs_dir: str = "logs") -> None:
        self.logs_dir = Path(logs_dir)
        self.btc_logs = self.logs_dir / "btc"
        self.weather_logs = self.logs_dir / "weather_temperature"
        self.state_file = self.logs_dir / "drift_monitor_state.json"
        self.persistent_state = self.load_last_report()

    def check_all(self) -> dict:
        report = {
            "btc": {},
            "weather_temperature": {},
            "global": {},
            "checked_at": datetime.now(timezone.utc).isoformat()
        }

        for family in ["btc", "weather_temperature"]:
            report[family] = {
                "feature_drift": self._check_feature_drift(family),
                "calibration_drift": self._check_calibration_drift(family),
                "precision_drift": self._check_precision_collapse(family),
                "execution_drift": self._check_execution_drift(family),
                "edge_decay": self._check_joint_edge_drift(family),
                "schema_health": self._check_schema_health(family)
            }
            
        report["global"] = {
            "execution_drift": self._check_global_execution_drift(),
            "schema_health": self._check_global_schema_health()
        }

        # Inject Persistence Tracking
        self._apply_persistence(report)

        self._persist(report)
        return report

    def load_last_report(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return {}
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _apply_persistence(self, report: dict):
        # Loops across newly computed report matching fields to prior runs
        prior = self.persistent_state
        
        for level_key in ["btc", "weather_temperature", "global"]:
            for drift_key, result in report[level_key].items():
                if not isinstance(result, dict) or "severity" not in result:
                    continue
                
                current_sev = result["severity"]
                
                # Fetch history
                try:
                    prior_res = prior.get(level_key, {}).get(drift_key, {})
                    prior_sev = prior_res.get("severity", "none")
                    prior_streak = prior_res.get("persistence_count", 0)
                except Exception:
                    prior_sev = "none"
                    prior_streak = 0
                
                if current_sev == "none":
                    result["persistence_count"] = 0
                elif current_sev == prior_sev:
                    result["persistence_count"] = prior_streak + 1
                else:
                    result["persistence_count"] = 1
                    
                # Bind an action recommendation
                result["recommended_action"] = self._recommend_action(drift_key, current_sev, result["persistence_count"])

    def _recommend_action(self, drift_type: str, severity: str, streak: int) -> str:
        if severity == "none": return "none"
        if severity == "low": return "log_only"
        
        if drift_type in ("calibration_drift", "schema_health", "precision_drift"):
            if severity == "critical" or (severity == "high" and streak >= 2):
                return "choke_family"
            if severity == "medium" and streak >= 3:
                return "tighten_governor"
                
        if drift_type == "execution_drift" or drift_type == "edge_decay":
            if severity == "critical" or (severity == "high" and streak >= 2):
                return "reduce_size"
            if severity == "medium" and streak >= 2:
                return "tighten_liquidity"
                
        if drift_type == "feature_drift":
            if severity == "critical" or streak >= 3:
                return "raise_retrain_priority"
                
        return "log_only"

    def _make_result(self, drift_type: str, score: float, extra: dict = None) -> dict:
        r = {"score": score, "severity": _get_severity(drift_type, score)}
        if extra: r.update(extra)
        return r

    def _split_df(self, df: pd.DataFrame, time_col: str):
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
            df = df.sort_values(time_col)
        mid = len(df) // 2
        return df.iloc[:mid], df.iloc[mid:]

    def _check_feature_drift(self, family: str) -> dict:
        ct_path = getattr(self, f"{family.split('_')[0]}_logs") / "contract_targets.csv"
        df = _safe_read(ct_path)
        if len(df) < 100: return self._make_result("feature_drift", 0.0)

        ref, cur = self._split_df(df, "timestamp")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if not c.startswith("target") and not c.startswith("tp_")]
        
        drifted, max_psi = 0, 0.0
        for col in feature_cols[:30]:
            psi = _psi_for_column(ref[col], cur[col])
            max_psi = max(max_psi, psi)
            if psi > _THRESHOLDS["feature_drift"]["medium"]:
                drifted += 1
                
        # Final score represents maximum PSI across active set
        return self._make_result("feature_drift", max_psi, {"drifted_features_count": drifted})

    def _check_calibration_drift(self, family: str) -> dict:
        df = _safe_read(self.logs_dir / "alpha_feedback_clean.csv")
        subset = df[df["market_family"].fillna("").astype(str).str.startswith(family)].copy() if "market_family" in df.columns else pd.DataFrame()
        
        if len(subset) < 50 or "entry_p_tp" not in subset.columns: return self._make_result("calibration", 0.0)
        
        ref, cur = self._split_df(subset, "attributed_at")
        b_ref = _brier_score(ref["entry_p_tp"], ref["direction_correct"])
        b_cur = _brier_score(cur["entry_p_tp"], cur["direction_correct"])
        delta = b_cur - b_ref
        
        return self._make_result("calibration", max(0.0, delta))

    def _check_precision_collapse(self, family: str) -> dict:
        df = _safe_read(self.logs_dir / "alpha_feedback_clean.csv")
        subset = df[df["market_family"].fillna("").astype(str).str.startswith(family)].copy() if "market_family" in df.columns else pd.DataFrame()
        
        if len(subset) < 30 or "alpha_verdict" not in subset.columns: return self._make_result("precision", 0.0)
        
        ref, cur = self._split_df(subset, "attributed_at")
        ref_p = float((ref["alpha_verdict"] == "alpha_success").mean())
        cur_p = float((cur["alpha_verdict"] == "alpha_success").mean())
        
        return self._make_result("precision", max(0.0, ref_p - cur_p))

    def _check_execution_drift(self, family: str) -> dict:
        df = _safe_read(self.logs_dir / "execution_feedback.csv")
        subset = df[df["market_family"].fillna("").astype(str).str.startswith(family)].copy() if "market_family" in df.columns else pd.DataFrame()
        
        if len(subset) < 50 or "slippage_error" not in subset.columns: return self._make_result("execution", 0.0)
        
        ref, cur = self._split_df(subset, "attributed_at")
        ref_slip = pd.to_numeric(ref["slippage_error"], errors="coerce").fillna(0.0).mean()
        cur_slip = pd.to_numeric(cur["slippage_error"], errors="coerce").fillna(0.0).mean()
        
        return self._make_result("execution", max(0.0, float(cur_slip - ref_slip)))
        
    def _check_joint_edge_drift(self, family: str) -> dict:
        df = _safe_read(self.logs_dir / "closed_positions.csv")
        subset = df[df["market_family"].fillna("").astype(str).str.startswith(family)].copy() if "market_family" in df.columns else pd.DataFrame()
        
        if len(subset) < 30 or "expected_return" not in subset.columns: return self._make_result("edge_decay", 0.0)
        
        ref, cur = self._split_df(subset, "closed_at")
        def _get_decay(s_df):
            ev = pd.to_numeric(s_df["expected_return"], errors="coerce").fillna(0.0)
            cost = pd.to_numeric(s_df["exit_realized_slippage_bps"], errors="coerce").fillna(0.0) * 0.0001 * 2.0
            return float((cost - ev).mean())
            
        ref_decay = _get_decay(ref)
        cur_decay = _get_decay(cur)
        
        return self._make_result("edge_decay", max(0.0, float(cur_decay - ref_decay)))

    def _check_schema_health(self, family: str) -> dict:
        """Tracking prediction feature gaps."""
        df = _safe_read(self.logs_dir / "feature_lineage.csv") # Assuming schema metadata lives here or similar tracking
        # We approximate using contract targets instead if lineage lacks depth
        ct_path = getattr(self, f"{family.split('_')[0]}_logs") / "contract_targets.csv"
        subset = _safe_read(ct_path)
        
        if len(subset) < 50: return self._make_result("schema_health", 0.0)
        
        # Approximate schema misses by detecting columns heavily defaulting to zero or median in current vs ref
        ref, cur = self._split_df(subset, "timestamp")
        numeric_cols = subset.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if not c.startswith("target") and not c.startswith("tp_")]
        
        if not feature_cols: return self._make_result("schema_health", 0.0)
        
        null_rate_ref = float(ref[feature_cols].isna().mean().mean())
        null_rate_cur = float(cur[feature_cols].isna().mean().mean())
        
        return self._make_result("schema_health", max(0.0, null_rate_cur - null_rate_ref))

    def _check_global_execution_drift(self) -> dict:
        df = _safe_read(self.logs_dir / "execution_feedback.csv")
        if len(df) < 50 or "slippage_error" not in df.columns: return self._make_result("execution", 0.0)
        
        ref, cur = self._split_df(df, "attributed_at")
        ref_slip = pd.to_numeric(ref["slippage_error"], errors="coerce").fillna(0.0).mean()
        cur_slip = pd.to_numeric(cur["slippage_error"], errors="coerce").fillna(0.0).mean()
        
        return self._make_result("execution", max(0.0, float(cur_slip - ref_slip)))
        
    def _check_global_schema_health(self) -> dict:
        # Same proxy estimation across unified
        btc_s = self._check_schema_health("btc")
        wth_s = self._check_schema_health("weather_temperature")
        
        val = max(btc_s["score"], wth_s["score"])
        return self._make_result("schema_health", val)
        
    def _persist(self, report: dict) -> None:
        try:
            self.state_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("DriftMonitor: could not persist state: %s", exc)

if __name__ == "__main__":
    monitor = DriftMonitor()
    res = monitor.check_all()
    print(json.dumps(res, indent=2))
