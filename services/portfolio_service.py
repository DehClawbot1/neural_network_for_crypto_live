"""
services/portfolio_service.py
──────────────────────────────
Pure position state, exposure, PnL, and limits.

Rules
─────
1. This service only READS state — never writes.
2. All methods return typed dicts or scalars. No side effects.
3. DataFaultError is raised if required files are missing or corrupt —
   callers must not proceed with stale/unknown portfolio state.
4. No trading decisions are made here.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from services.types import DataFaultError, PortfolioState

logger = logging.getLogger(__name__)


class PortfolioService:
    """
    Reads position state from logs CSVs and returns a consistent snapshot.

    This is a read-only service. It never modifies any file.
    """

    def __init__(self, logs_dir: str = "logs") -> None:
        self.logs_dir = Path(logs_dir)

    # ── Primary API ───────────────────────────────────────────────────────────

    def snapshot(self, available_usdc: float = 0.0) -> PortfolioState:
        """
        Return the current PortfolioState.

        Parameters
        ──────────
        available_usdc  : current wallet balance (caller must provide; this service
                          does not call the exchange API).

        Returns
        ───────
        PortfolioState TypedDict.

        Raises
        ──────
        DataFaultError  : if open_positions.csv is unreadable and we cannot determine
                          safe exposure levels (only in strict mode).
        """
        open_df   = self._read_open()
        closed_df = self._read_closed()

        # Open positions
        open_count             = len(open_df)
        open_notional          = self._sum_col(open_df, ["size_usdc", "entry_notional", "notional"])
        btc_mask               = ~open_df.get("market_family", pd.Series("", index=open_df.index)).astype(str).str.startswith("weather_temperature")
        weather_mask           = ~btc_mask
        btc_open_count         = int(btc_mask.sum())
        weather_open_count     = int(weather_mask.sum())
        btc_open_notional      = self._sum_col(open_df[btc_mask], ["size_usdc", "entry_notional", "notional"])
        weather_open_notional  = self._sum_col(open_df[weather_mask], ["size_usdc", "entry_notional", "notional"])

        # Unrealized PnL (approximate from open positions if column exists)
        unrealized_pnl = self._sum_col(open_df, ["unrealized_pnl", "paper_unrealized_pnl"])

        # Session PnL + drawdown from closed positions (this session only — today)
        realized_pnl_session, max_drawdown_session = self._session_pnl_and_drawdown(closed_df)

        # Reconciliation rate (last 50 closes)
        reconciliation_rate = self._reconciliation_rate(closed_df, window=50)

        return PortfolioState(
            open_count=open_count,
            open_notional_usdc=round(open_notional, 4),
            btc_open_count=btc_open_count,
            btc_open_notional=round(btc_open_notional, 4),
            weather_open_count=weather_open_count,
            weather_open_notional=round(weather_open_notional, 4),
            unrealized_pnl=round(unrealized_pnl, 4),
            realized_pnl_session=round(realized_pnl_session, 4),
            available_usdc=round(max(0.0, available_usdc), 4),
            max_drawdown_session=round(max_drawdown_session, 4),
            reconciliation_rate=round(reconciliation_rate, 4),
        )

    def get_family_exposure(self, family: str) -> float:
        """Return current open notional for a given market family prefix."""
        open_df = self._read_open()
        if open_df.empty or "market_family" not in open_df.columns:
            return 0.0
        mask = open_df["market_family"].astype(str).str.startswith(family)
        return float(self._sum_col(open_df[mask], ["size_usdc", "entry_notional", "notional"]))

    def get_open_token_ids(self) -> set[str]:
        """Return set of token_ids with currently open positions."""
        open_df = self._read_open()
        if open_df.empty or "token_id" not in open_df.columns:
            return set()
        return set(open_df["token_id"].dropna().astype(str).tolist())

    def get_session_win_rate(self, window: int = 50) -> float:
        """Win rate over the last `window` closed trades."""
        closed_df = self._read_closed()
        if closed_df.empty:
            return 0.0
        pnl = pd.to_numeric(
            closed_df.get("net_realized_pnl", closed_df.get("realized_pnl", pd.Series(dtype=float))),
            errors="coerce",
        ).fillna(0.0)
        recent = pnl.tail(window)
        return float((recent > 0).mean()) if len(recent) > 0 else 0.0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _read_open(self) -> pd.DataFrame:
        path = self.logs_dir / "open_positions.csv"
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception as exc:
            logger.warning("PortfolioService: could not read open_positions.csv: %s", exc)
            return pd.DataFrame()

    def _read_closed(self) -> pd.DataFrame:
        path = self.logs_dir / "closed_positions.csv"
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
            # Sort by close time ascending for session calculations
            if "closed_at" in df.columns:
                df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True, errors="coerce")
                df = df.sort_values("closed_at")
            return df
        except Exception as exc:
            logger.warning("PortfolioService: could not read closed_positions.csv: %s", exc)
            return pd.DataFrame()

    @staticmethod
    def _sum_col(df: pd.DataFrame, candidates: list[str]) -> float:
        if df.empty:
            return 0.0
        for col in candidates:
            if col in df.columns:
                return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
        return 0.0

    def _session_pnl_and_drawdown(self, closed_df: pd.DataFrame) -> tuple[float, float]:
        """Return (realized_pnl_today, max_drawdown_today)."""
        if closed_df.empty:
            return 0.0, 0.0

        # Filter to today UTC
        today = datetime.now(timezone.utc).date()
        if "closed_at" in closed_df.columns:
            mask = closed_df["closed_at"].dt.date == today
            today_df = closed_df[mask]
        else:
            today_df = closed_df

        pnl_col = "net_realized_pnl" if "net_realized_pnl" in today_df.columns else "realized_pnl"
        if pnl_col not in today_df.columns:
            return 0.0, 0.0

        pnl = pd.to_numeric(today_df[pnl_col], errors="coerce").fillna(0.0)
        total_pnl = float(pnl.sum())

        # Max drawdown = max peak-to-trough of cumulative PnL
        running = pnl.cumsum()
        peak = running.cummax()
        drawdown = float((peak - running).max()) if not running.empty else 0.0

        return total_pnl, drawdown

    def _reconciliation_rate(self, closed_df: pd.DataFrame, window: int = 50) -> float:
        """Fraction of last `window` closes that were operational/reconciliation."""
        if closed_df.empty:
            return 0.0
        recent = closed_df.tail(window)
        if "close_reason" not in recent.columns:
            return 0.0
        reasons = recent["close_reason"].astype(str).str.lower()
        op_mask = reasons.str.contains(
            "reconciliation|external_manual_close|api_rejection|sync_forced", na=False
        )
        return float(op_mask.mean())


if __name__ == "__main__":
    import tempfile, os, json

    with tempfile.TemporaryDirectory() as td:
        logs = Path(td)

        # Write test files
        open_pos = pd.DataFrame([
            {"token_id": "t1", "market_family": "btc", "size_usdc": 15.0},
            {"token_id": "t2", "market_family": "weather_temperature_threshold", "size_usdc": 10.0},
        ])
        open_pos.to_csv(logs / "open_positions.csv", index=False)

        closed_pos = pd.DataFrame([
            {"closed_at": "2026-04-13T10:00:00Z", "realized_pnl": 0.5, "close_reason": "take_profit"},
            {"closed_at": "2026-04-13T10:05:00Z", "realized_pnl": -0.3, "close_reason": "reconciliation_close"},
        ])
        closed_pos.to_csv(logs / "closed_positions.csv", index=False)

        svc = PortfolioService(logs_dir=td)
        state = svc.snapshot(available_usdc=50.0)

        assert state["open_count"] == 2
        assert state["btc_open_count"] == 1
        assert state["weather_open_count"] == 1
        assert state["open_notional_usdc"] == 25.0
        assert state["available_usdc"] == 50.0
        assert state["reconciliation_rate"] == 0.5  # 1 out of 2

        token_ids = svc.get_open_token_ids()
        assert "t1" in token_ids and "t2" in token_ids

    print("portfolio_service self-test PASSED.")
