"""
exploration_policy.py
Phase 12 — Exploration Policy

A self-improving system needs a little controlled exploration to discover
new patterns.  But exploration must never damage the account.

Rules:
  - Exploration only allowed when all safety conditions are met
  - Exploration trades are smaller (size_multiplier *= EXPLORE_SIZE_FRACTION)
  - Exploration trades are tagged is_exploration=True
  - Exploration trades are measured separately (exploration_feedback.csv)
  - Exploration results are NEVER mixed blindly with production trades
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

_EXPLORE_SIZE_FRACTION = 0.50   # Exploration trades use 50% of normal size


def _safe_float(value, default: float = 0.0) -> float:
    try:
        num = float(value)
        return float(default) if not np.isfinite(num) else num
    except Exception:
        return float(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except Exception:
        return float(default)


class ExplorationPolicy:
    """
    Decides whether controlled exploration is safe to run for a candidate,
    and modifies the candidate row accordingly.

    Usage:
        policy = ExplorationPolicy(logs_dir="logs")
        if policy.is_exploration_allowed(portfolio_state, governor_state):
            candidate = policy.tag_exploration_trade(candidate)
    """

    def __init__(self, logs_dir: str = "logs") -> None:
        self.logs_dir = Path(logs_dir)
        self.exploration_csv = self.logs_dir / "exploration_feedback.csv"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_exploration_allowed(
        self,
        portfolio_state: dict[str, Any] | None = None,
        governor_state: dict[str, Any] | None = None,
    ) -> bool:
        """
        Return True only when ALL safety conditions are met.
        Hard gates — cannot be overridden by config.
        """
        portfolio = portfolio_state or {}
        governor = governor_state or {}

        # 1. Account health: balance above minimum
        min_balance_usdc = _env_float("EXPLORE_MIN_BALANCE_USDC", 50.0)
        balance = _safe_float(portfolio.get("usdc_balance"), 0.0)
        if balance < min_balance_usdc:
            return False

        # 2. Drawdown is low
        max_drawdown_pct = _env_float("EXPLORE_MAX_DRAWDOWN_PCT", 0.08)
        drawdown = _safe_float(portfolio.get("drawdown_pct"), 999.0)
        if drawdown > max_drawdown_pct:
            return False

        # 3. Confidence is moderate (not desperate, not peak)
        min_confidence = _env_float("EXPLORE_MIN_CONFIDENCE", 0.35)
        max_confidence = _env_float("EXPLORE_MAX_CONFIDENCE", 0.75)
        avg_confidence = _safe_float(portfolio.get("recent_avg_confidence"), 0.5)
        if avg_confidence < min_confidence or avg_confidence > max_confidence:
            return False

        # 4. Liquidity is acceptable
        min_liquidity = _env_float("EXPLORE_MIN_LIQUIDITY_SCORE", 0.20)
        liquidity = _safe_float(portfolio.get("liquidity_score"), 1.0)
        if liquidity < min_liquidity:
            return False

        # 5. Family exposure is low (we're not overexposed to this family)
        max_family_exposure = _env_float("EXPLORE_MAX_FAMILY_EXPOSURE", 0.40)
        family_exposure = _safe_float(portfolio.get("family_exposure"), 0.0)
        if family_exposure > max_family_exposure:
            return False

        # 6. Governor level must be 0 (no active performance alert)
        governor_level = int(_safe_float(governor.get("governor_level"), 1))
        if governor_level > 0:
            return False

        return True

    def tag_exploration_trade(self, candidate_row: dict[str, Any]) -> dict[str, Any]:
        """
        Tag a candidate as an exploration trade and halve its size.
        Returns a modified copy of the candidate row.
        """
        row = dict(candidate_row)
        row["is_exploration"] = True
        row["exploration_tagged_at"] = datetime.now(timezone.utc).isoformat()

        # Halve size multiplier
        current_mult = _safe_float(row.get("size_multiplier", 1.0), 1.0)
        row["size_multiplier"] = round(current_mult * _EXPLORE_SIZE_FRACTION, 4)
        row["pre_exploration_size_multiplier"] = round(current_mult, 4)

        logger.info(
            "ExplorationPolicy: tagged exploration trade for token=%s size_mult %.2f→%.2f",
            row.get("token_id", ""),
            current_mult,
            row["size_multiplier"],
        )
        return row

    def record_exploration_outcome(self, report_row: dict[str, Any]) -> None:
        """
        Write a completed exploration trade to exploration_feedback.csv
        (separate from production alpha_feedback_clean.csv).
        """
        row = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "report_id": report_row.get("report_id", ""),
            "token_id": report_row.get("token_id", ""),
            "market_family": report_row.get("market_family", ""),
            "signal_label": report_row.get("signal_label", ""),
            "entry_p_tp": _safe_float(report_row.get("entry_p_tp_before_sl")),
            "entry_ev": _safe_float(report_row.get("entry_expected_return")),
            "realized_pnl": _safe_float(report_row.get("realized_pnl")),
            "roi": _safe_float(report_row.get("roi")),
            "holding_minutes": _safe_float(report_row.get("holding_minutes")),
            "outcome_class": report_row.get("outcome_class", ""),
            "is_exploration": True,
        }
        safe_csv_append(self.exploration_csv, pd.DataFrame([row]))
        logger.info("ExplorationPolicy: recorded exploration outcome for %s", row["report_id"])

    def exploration_summary(self, lookback: int = 100) -> dict[str, Any]:
        """Return a summary of recent exploration outcomes."""
        if not self.exploration_csv.exists():
            return {"exploration_trades": 0}
        try:
            df = pd.read_csv(self.exploration_csv, engine="python", on_bad_lines="skip")
        except Exception:
            return {"exploration_trades": 0}

        if df.empty:
            return {"exploration_trades": 0}

        recent = df.tail(lookback)
        pnl = pd.to_numeric(recent.get("realized_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
        return {
            "exploration_trades": int(len(recent)),
            "exploration_win_rate": float((pnl > 0).mean()) if not pnl.empty else 0.0,
            "exploration_avg_pnl": float(pnl.mean()) if not pnl.empty else 0.0,
        }
