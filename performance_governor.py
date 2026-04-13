from __future__ import annotations

import json
import logging
import os
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from csv_utils import safe_csv_append
from trade_quality import classify_exit_reason_family

logger = logging.getLogger(__name__)

# Base levels definition dict
_LEVEL_POLICY_DEFAULT = {
    "size_multiplier": 1.0,
    "min_confidence": 0.0,
    "min_liquidity_score": 0.0,
    "liquidity_size_multiplier": 1.0,
    "force_min_size": False,
    "top_signal_only": 0,
}

_LEVEL_POLICIES: dict[str, dict] = {
    "0": _LEVEL_POLICY_DEFAULT,
    "1": {
        "size_multiplier": 0.65,
        "min_confidence": 0.15,
        "min_liquidity_score": 0.10,
        "liquidity_size_multiplier": 0.80,
        "force_min_size": False,
        "top_signal_only": 0,
    },
    "2A": {
        "size_multiplier": 0.40,
        "min_confidence": 0.20,
        "min_liquidity_score": 0.30,
        "liquidity_size_multiplier": 0.60,
        "force_min_size": False,
        "top_signal_only": 2,
    },
    "2B": {
        "size_multiplier": 1.0,
        "min_confidence": 0.25,
        "min_liquidity_score": 0.50,
        "liquidity_size_multiplier": 0.50,
        "force_min_size": True,
        "top_signal_only": 1,
    },
}


def _get_level_policy(level: str) -> dict:
    return _LEVEL_POLICIES.get(str(level).upper(), _LEVEL_POLICY_DEFAULT)


class BaseGovernor:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.closed_file = self.logs_dir / "closed_positions.csv"

    def _safe_read(self) -> pd.DataFrame:
        if not self.closed_file.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.closed_file, engine="python", on_bad_lines="skip")
            if not df.empty and "closed_at" in df.columns:
                df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True, errors="coerce")
                df = df.sort_values("closed_at")
            return df
        except Exception:
            return pd.DataFrame()
            
    def _rank_severity(self, lvl: str) -> int:
        return {"0": 0, "1": 1, "2A": 2, "2B": 3}.get(str(lvl).upper(), 0)

    @staticmethod
    def max_severity(state1: dict, state2: dict) -> dict:
        """Merges two policies, adopting the MAXIMUM severity constraints of both."""
        rank1 = {"0": 0, "1": 1, "2A": 2, "2B": 3}.get(str(state1.get("governor_level", "0")).upper(), 0)
        rank2 = {"0": 0, "1": 1, "2A": 2, "2B": 3}.get(str(state2.get("governor_level", "0")).upper(), 0)
        
        dominant = state1 if rank1 >= rank2 else state2
        recessive = state2 if rank1 >= rank2 else state1
        
        combined_reasons = set(dominant.get("reason", "").split(",")) | set(recessive.get("reason", "").split(","))
        combined_reasons.discard("")
        
        merged = dict(dominant)
        merged["reason"] = ",".join(combined_reasons)
        
        # Enforce strict intersection overrides
        merged["size_multiplier"] = min(dominant.get("size_multiplier", 1.0), recessive.get("size_multiplier", 1.0))
        merged["min_confidence"] = max(dominant.get("min_confidence", 0.0), recessive.get("min_confidence", 0.0))
        merged["force_min_size"] = bool(dominant.get("force_min_size") or recessive.get("force_min_size"))
        
        ts1 = int(dominant.get("top_signal_only", 0))
        ts2 = int(recessive.get("top_signal_only", 0))
        if ts1 == 0 and ts2 == 0: merged["top_signal_only"] = 0
        elif ts1 == 0: merged["top_signal_only"] = ts2
        elif ts2 == 0: merged["top_signal_only"] = ts1
        else: merged["top_signal_only"] = min(ts1, ts2)
        
        return merged


class GlobalPerformanceGovernor(BaseGovernor):
    """
    Monitors Macro / Systemic collapses. Explicitly ignores family-level alpha decay.
    """
    def __init__(self, logs_dir="logs"):
        super().__init__(logs_dir)
        self.output_file = self.logs_dir / "global_governor.csv"
        
    def evaluate(self) -> dict:
        df = self._safe_read()
        if df.empty:
            return {**_get_level_policy("0"), "governor_level": "0", "reason": ""}
            
        recent_100 = df.tail(100)
        
        # Systemic Pain extraction
        reasons = recent_100.get("close_reason", pd.Series(dtype=str)).fillna("").astype(str).str.strip().str.lower()
        exit_family = recent_100.get("exit_reason_family", pd.Series(dtype=str)).fillna("").astype(str).str.strip().str.lower()
        
        operational_closes = float(exit_family.eq("operational").mean()) if not recent_100.empty else 0.0
        
        # Global Execution Edge Decay (Slippage Spikes)
        exec_slip = pd.to_numeric(recent_100.get("exit_realized_slippage_bps"), errors="coerce").fillna(0.0)
        sys_slippage_avg = float(exec_slip.tail(25).mean()) if not exec_slip.empty else 0.0
        
        # Total Drawdown
        pnl = pd.to_numeric(df.get("net_realized_pnl", df.get("realized_pnl")), errors="coerce").fillna(0.0)
        running = pnl.cumsum()
        drawdown = float((running.cummax() - running).max()) if not running.empty else 0.0
        
        failures_2B = []
        failures_2A = []
        failures_1 = []
        
        # Check Systemic Thresholds
        if operational_closes > 0.30: failures_2B.append("macro_operational_outage")
        if drawdown > 100.0: failures_2A.append("macro_drawdown_2A")
        if sys_slippage_avg > 250.0: failures_2A.append("macro_slippage_spike_2A") # 2.5% avg slip locally

        if drawdown > 50.0: failures_1.append("macro_drawdown_1")
        if sys_slippage_avg > 150.0: failures_1.append("macro_slippage_spike_1")
        if operational_closes > 0.15: failures_1.append("macro_operational_spike")

        level = "0"
        reasons_list = []
        if failures_2B:
            level = "2B"
            reasons_list = failures_2B
        elif failures_2A:
            level = "2A"
            reasons_list = failures_2A
        elif failures_1:
            level = "1"
            reasons_list = failures_1
            
        state = {
            **_get_level_policy(level),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "governor_level": level,
            "reason": ",".join(reasons_list),
            "macro_drawdown": round(drawdown, 2),
            "macro_operational_rate": round(operational_closes, 4),
            "macro_slippage_bps": round(sys_slippage_avg, 2)
        }
        
        df_out = pd.DataFrame([state])
        safe_csv_append(self.output_file, df_out)
        
        if level != "0":
            logger.warning(f"GLOBAL GOVERNOR ACTIVE [{level}]: {','.join(reasons_list)}")
            
        return state


class FamilyPerformanceGovernor(BaseGovernor):
    """
    Monitors Strategy / Alpha Pain for a specific family.
    Uses EWMA structures to avoid slow-moving small sample bias.
    """
    def __init__(self, family: str, logs_dir="logs"):
        super().__init__(logs_dir)
        self.family = family.lower()
        self.output_file = self.logs_dir / f"{self.family}_governor.csv"
        
    def _calculate_edge_decay(self, df: pd.DataFrame) -> float:
        """Returns average execution edge decay (EV estimate vs Actual cost) in percentages."""
        if df.empty or "expected_return" not in df.columns or "exit_realized_slippage_bps" not in df.columns:
            return 0.0
            
        ev = pd.to_numeric(df["expected_return"], errors="coerce").fillna(0.0)
        # Assuming slippage is in bps, convert to pct (1 bps = 0.0001)
        # Multiply by 2.0 assuming entry+exit slippage match exit slippage observation coarsely
        realized_cost = pd.to_numeric(df["exit_realized_slippage_bps"], errors="coerce").fillna(0.0) * 0.0001 * 2.0
        
        decay = realized_cost - ev
        return float(decay.mean())

    def evaluate(self) -> dict:
        df = self._safe_read()
        
        # Isolate exactly this family
        if not df.empty and "market_family" in df.columns:
            df["market_family_clean"] = df["market_family"].astype(str).str.lower()
            if self.family == "weather":
                df = df[df["market_family_clean"].str.startswith("weather_temperature")]
            else:
                df = df[~df["market_family_clean"].str.startswith("weather_temperature")]

        if df.empty or len(df) < 10:
            return {**_get_level_policy("0"), "governor_level": "0", "reason": "insufficient_data"}
            
        # EWMA weighting over the last 100 trades to smooth out small localized spikes
        recent = df.tail(100).copy()
        
        pnl = pd.to_numeric(recent.get("net_realized_pnl", recent.get("realized_pnl")), errors="coerce").fillna(0.0)
        
        wins = (pnl > 0).astype(float)
        # Simple EWMA win rate (span 30)
        win_rate_ewma = float(wins.ewm(span=30, adjust=False).mean().iloc[-1]) if not wins.empty else 0.0
        
        running = pnl.cumsum()
        peak = running.cummax()
        drawdown = float((peak - running).max()) if not running.empty else 0.0
        
        avg_pnl_ewma = float(pnl.ewm(span=30, adjust=False).mean().iloc[-1]) if not pnl.empty else 0.0
        
        edge_decay = self._calculate_edge_decay(recent.tail(25))

        failures_2B = []
        failures_2A = []
        failures_1 = []

        # BTC Specific Limits vs Weather Specific Limits
        if "btc" in self.family:
            min_wr_1, min_wr_2A, min_wr_2B = 0.38, 0.32, 0.28
            max_dd_1, max_dd_2A, max_dd_2B = 30.0, 50.0, 75.0
            max_decay_1, max_decay_2A = 0.01, 0.025 # 1% and 2.5% edge gap
        else:
            # Weather generally has lower win rate binary bets, so lower tolerance thresholds
            min_wr_1, min_wr_2A, min_wr_2B = 0.30, 0.25, 0.20
            max_dd_1, max_dd_2A, max_dd_2B = 25.0, 40.0, 60.0
            max_decay_1, max_decay_2A = 0.02, 0.04 # Wider spreads
            
        if win_rate_ewma < min_wr_2B: failures_2B.append("alpha_win_rate_2B")
        if drawdown > max_dd_2B: failures_2B.append("alpha_drawdown_2B")

        if win_rate_ewma < min_wr_2A: failures_2A.append("alpha_win_rate_2A")
        if drawdown > max_dd_2A: failures_2A.append("alpha_drawdown_2A")
        if edge_decay > max_decay_2A: failures_2A.append("alpha_edge_decay_2A")

        if win_rate_ewma < min_wr_1: failures_1.append("alpha_win_rate_1")
        if drawdown > max_dd_1: failures_1.append("alpha_drawdown_1")
        if edge_decay > max_decay_1: failures_1.append("alpha_edge_decay_1")
        
        level = "0"
        reasons = []
        if failures_2B:
            level = "2B"
            reasons = failures_2B
        elif failures_2A:
            level = "2A"
            reasons = failures_2A
        elif failures_1:
            level = "1"
            reasons = failures_1

        state = {
            **_get_level_policy(level),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "governor_level": level,
            "reason": ",".join(reasons),
            "family_win_rate": round(win_rate_ewma, 4),
            "family_drawdown": round(drawdown, 2),
            "family_edge_decay": round(edge_decay, 4),
        }
        
        state = self._apply_drift_overrides(state)
        
        df_out = pd.DataFrame([state])
        safe_csv_append(self.output_file, df_out)
        return state

    def _apply_drift_overrides(self, state: dict) -> dict:
        import json
        drift_path = self.logs_dir / "drift_monitor_state.json"
        if not drift_path.exists(): 
            return state
        
        try:
            drift_report = json.loads(drift_path.read_text(encoding="utf-8"))
        except Exception:
            return state
            
        fam_key = "weather_temperature" if self.family == "weather" else self.family
        fam_drift = drift_report.get(fam_key, {})
            
        for drift_type, res in fam_drift.items():
            action = res.get("recommended_action", "none")
            
            if action == "choke_family":
                state["size_multiplier"] = state.get("size_multiplier", 1.0) * 0.5
                state["min_confidence"] = state.get("min_confidence", 0.0) + 0.10
                state["reason"] = f"{state.get('reason','')},{drift_type}_choke".strip(",")
                
            elif action == "tighten_governor":
                if state.get("governor_level", "0") == "0":
                    state.update(_get_level_policy("1"))
                    state["governor_level"] = "1"
                state["reason"] = f"{state.get('reason','')},{drift_type}_tighten".strip(",")
                
            elif action == "reduce_size":
                state["size_multiplier"] = state.get("size_multiplier", 1.0) * 0.7
                state["reason"] = f"{state.get('reason','')},{drift_type}_size_reduce".strip(",")
                
            elif action == "tighten_liquidity":
                state["min_liquidity_score"] = max(state.get("min_liquidity_score", 0.0), 0.50)
                state["reason"] = f"{state.get('reason','')},{drift_type}_liquidity".strip(",")
                
            elif action == "raise_retrain_priority":
                state["force_min_size"] = True
                state["reason"] = f"{state.get('reason','')},{drift_type}_retrain".strip(",")
                
        global_drift = drift_report.get("global", {})
        if global_drift.get("schema_health", {}).get("recommended_action") == "choke_family":
             state["size_multiplier"] = 0.0
             state["reason"] = f"{state.get('reason','')},global_schema_kill".strip(",")
             
        return state
