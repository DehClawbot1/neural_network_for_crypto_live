from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from db import Database


def _safe_str(value) -> str:
    """Normalize None/NaN/empty to consistent empty string for key comparison."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    s = str(value).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s


class StateMismatchDetector:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.logs_dir / "trading.db")

    def _trade_key(self, trade):
        # ── BUG FIX (BUG 8): Use _safe_str to normalize both TradeLifecycle
        #    attrs and DataFrame row values consistently ──
        return (
            _safe_str(getattr(trade, "token_id", "")),
            _safe_str(getattr(trade, "condition_id", "")),
            _safe_str(getattr(trade, "outcome_side", "")),
        )

    def detect(self, active_trades, live_positions_df):
        min_notional = float(os.getenv("MIN_RECONCILED_POSITION_NOTIONAL_USDC", "0.01") or 0.01)
        local_keys = set()
        for trade in (active_trades or []):
            token_id = _safe_str(getattr(trade, "token_id", ""))
            if not token_id:
                continue
            try:
                shares = float(getattr(trade, "shares", 0.0) or 0.0)
                px = float(getattr(trade, "current_price", getattr(trade, "entry_price", 0.0)) or 0.0)
                if shares <= 0 or (shares * max(px, 0.0)) < min_notional:
                    continue
            except Exception:
                continue
            local_keys.add(self._trade_key(trade))
        live_keys = set()
        if live_positions_df is not None and not live_positions_df.empty:
            for _, row in live_positions_df.iterrows():
                token_id = _safe_str(row.get("token_id", ""))
                if not token_id:
                    continue
                try:
                    shares = float(row.get("shares", 0.0) or 0.0)
                    px = float(
                        row.get("mark_price", row.get("current_price", row.get("avg_entry_price", row.get("entry_price", 0.0))))
                        or 0.0
                    )
                    if shares <= 0 or (shares * max(px, 0.0)) < min_notional:
                        continue
                except Exception:
                    continue
                live_keys.add(
                    (
                        token_id,
                        _safe_str(row.get("condition_id", "")),
                        _safe_str(row.get("outcome_side", "")),
                    )
                )

        local_only = sorted(local_keys - live_keys)
        live_only = sorted(live_keys - local_keys)

        severity = "ok"
        if local_only or live_only:
            severity = "severe"

        detail = {
            "local_only": local_only,
            "live_only": live_only,
            "local_count": len(local_keys),
            "live_count": len(live_keys),
        }
        return {
            "severity": severity,
            "source": "trade_manager_vs_live_positions",
            "detail": detail,
            "freeze_entries": severity == "severe",
        }

    def record(self, mismatch):
        mismatch_id = f"{mismatch.get('source', 'state_mismatch')}:{datetime.now(timezone.utc).isoformat()}"
        now = datetime.now(timezone.utc).isoformat()
        detail_json = json.dumps(mismatch.get("detail", {}), default=str)
        self.db.execute(
            "INSERT OR REPLACE INTO state_mismatches (mismatch_id, severity, source, detail, created_at, resolved_at) VALUES (?, ?, ?, ?, ?, ?)",
            (mismatch_id, mismatch.get("severity"), mismatch.get("source"), detail_json, now, None),
        )
        self.db.execute(
            "INSERT INTO risk_events (token_id, event_type, detail) VALUES (?, ?, ?)",
            (None, "state_mismatch", detail_json),
        )
        return mismatch_id
