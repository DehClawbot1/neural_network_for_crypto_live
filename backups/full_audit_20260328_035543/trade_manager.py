from __future__ import annotations

import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

import pandas as pd

from trade_lifecycle import TradeLifecycle, TradeState
from config import TradingConfig

logger = logging.getLogger(__name__)


class TradeManager:
    """
    Manages a collection of TradeLifecycle instances, acting as the central
    runtime trade manager for the supervisor.

    BUG FIXES APPLIED:
      C - persist confidence_at_entry + signal_label to positions.csv
      D - pass real close_reason (not hardcoded "policy_exit")
      E - TradeLifecycle.close() now accepts reason param
      F - write both realized_pnl AND net_realized_pnl to closed_positions.csv
      H - consistent ISO timestamp format
    """

    def __init__(self, logs_dir="logs"):
        self.active_trades: Dict[str, TradeLifecycle] = {}
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.positions_file = self.logs_dir / "positions.csv"
        self.closed_file = self.logs_dir / "closed_positions.csv"
        logger.info("[+] Initialized TradeManager.")

    def _get_trade_key(self, signal_row: pd.Series) -> Optional[str]:
        market = signal_row.get("market_title") or signal_row.get("market")
        outcome_side = signal_row.get("outcome_side") or signal_row.get("side")
        if not market or not outcome_side:
            logger.warning("Missing market or outcome_side in signal: %s", dict(signal_row) if hasattr(signal_row, 'items') else signal_row)
            return None
        return f"{market}-{outcome_side}"

    def handle_signal(self, signal_row: pd.Series, confidence: float, size_usdc: float) -> Optional[TradeLifecycle]:
        trade_key = self._get_trade_key(signal_row)
        if trade_key is None:
            return None

        market = signal_row.get("market_title") or signal_row.get("market")
        outcome_side = signal_row.get("outcome_side") or signal_row.get("side")
        token_id = signal_row.get("token_id")
        condition_id = signal_row.get("condition_id")

        entry_price = (
            signal_row.get("current_price")
            or signal_row.get("entry_price")
            or signal_row.get("market_last_trade_price")
            or signal_row.get("price")
        )
        if entry_price is None or float(entry_price) <= 0:
            logger.warning("Cannot open trade — entry_price is missing or zero for %s", market)
            return None
        entry_price = float(entry_price)

        if trade_key in self.active_trades:
            trade = self.active_trades[trade_key]
            logger.info("[!] Trade already open for %s (%s). Skipping new entry.", market, outcome_side)
            return trade
        else:
            trade = TradeLifecycle(
                market=market,
                token_id=token_id,
                condition_id=condition_id,
                outcome_side=outcome_side,
            )
            trade.on_signal(signal_row.to_dict() if hasattr(signal_row, 'to_dict') else dict(signal_row))
            # ── BUG FIX C: Store confidence and label on the trade object ──
            trade.confidence_at_entry = float(confidence)
            trade.signal_label = str(signal_row.get("signal_label", "UNKNOWN") or "UNKNOWN")
            trade.enter(size_usdc=size_usdc, entry_price=entry_price)
            self.active_trades[trade_key] = trade
            logger.info("[+] New trade initiated for %s (%s) with %s USDC at %.4f.",
                        market, outcome_side, size_usdc, entry_price)
            return trade

    def update_markets(self, market_prices: Dict[str, float]):
        for trade_key, trade in list(self.active_trades.items()):
            if trade.market in market_prices:
                live_price = market_prices[trade.market]
                trade.update_market(live_price)

    def process_exits(self, current_timestamp: datetime, alerts_df: pd.DataFrame = None):
        closed_trades: List[TradeLifecycle] = []
        close_reasons: Dict[str, str] = {}  # ── BUG FIX D: track reasons ──

        for trade_key, trade in list(self.active_trades.items()):
            if trade.state == TradeState.CLOSED:
                closed_trades.append(trade)
                close_reasons[trade_key] = trade.close_reason or "already_closed"
                continue

            if not trade.opened_at:
                continue

            try:
                opened_dt = datetime.fromisoformat(trade.opened_at)
            except (ValueError, TypeError):
                continue

            entry_price = float(trade.entry_price or 0)
            current_price = float(trade.current_price or entry_price)
            if entry_price <= 0:
                continue

            roi = (current_price - entry_price) / entry_price
            minutes_open = (current_timestamp - opened_dt).total_seconds() / 60.0

            close_reason = None

            if roi >= TradingConfig.PAPER_TP_ROI:
                close_reason = "take_profit_roi"
            elif (current_price - entry_price) >= TradingConfig.SHADOW_TP_DELTA:
                close_reason = "take_profit_price_move"
            elif (entry_price - current_price) >= TradingConfig.SHADOW_SL_DELTA:
                close_reason = "stop_loss"
            elif minutes_open >= 180:
                close_reason = "time_stop"
            elif roi < -TradingConfig.PAPER_TRAILING_STOP and minutes_open > 15:
                close_reason = "trailing_stop"

            if close_reason:
                # ── BUG FIX E: pass reason to trade.close() ──
                trade.close(current_price, reason=close_reason)
                logger.info("[->] Closed trade for %s (%s). Reason: %s. PnL: %.4f",
                            trade.market, trade.outcome_side, close_reason, trade.realized_pnl)
                closed_trades.append(trade)
                close_reasons[trade_key] = close_reason

        for trade in closed_trades:
            key = f"{trade.market}-{trade.outcome_side}"
            self.active_trades.pop(key, None)

        if closed_trades:
            self._append_closed_trades(closed_trades)

        return closed_trades

    def get_open_positions(self) -> List[TradeLifecycle]:
        return [t for t in self.active_trades.values() if t.state != TradeState.CLOSED]

    def get_metrics(self) -> Dict[str, any]:
        open_trades = self.get_open_positions()
        total_unrealized_pnl = sum(trade.unrealized_pnl for trade in open_trades)
        return {
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_open_positions": len(open_trades),
            "last_reconciled_at": datetime.now().isoformat(),
        }

    def _trade_to_dict(self, trade: TradeLifecycle) -> dict:
        """
        Convert a TradeLifecycle to a dict matching the dashboard CSV schema.
        BUG FIX C: includes confidence_at_entry and signal_label.
        BUG FIX H: consistent ISO timestamps.
        """
        return {
            "position_id": f"{trade.market}-{trade.outcome_side}",
            "market": trade.market,
            "market_title": trade.market,
            "token_id": trade.token_id,
            "condition_id": trade.condition_id,
            "outcome_side": trade.outcome_side,
            "order_side": "BUY",
            "entry_price": trade.entry_price,
            "current_price": trade.current_price,
            "size_usdc": trade.size_usdc,
            "shares": trade.shares,
            "market_value": trade.shares * trade.current_price if trade.current_price else 0.0,
            "unrealized_pnl": round(trade.unrealized_pnl, 4),
            "realized_pnl": round(trade.realized_pnl, 4),
            # ── BUG FIX F: write both column names so dashboard finds it ──
            "net_realized_pnl": round(trade.realized_pnl, 4),
            "opened_at": trade.opened_at,
            "closed_at": trade.closed_at,
            "status": trade.state.value if hasattr(trade.state, 'value') else str(trade.state),
            # ── BUG FIX C: persist confidence and signal label ──
            "confidence": trade.confidence_at_entry,
            "confidence_at_entry": trade.confidence_at_entry,
            "signal_label": trade.signal_label,
        }

    def persist_open_positions(self):
        open_trades = self.get_open_positions()
        if not open_trades:
            pd.DataFrame(columns=[
                "position_id", "market", "market_title", "token_id",
                "condition_id", "outcome_side", "order_side",
                "entry_price", "current_price", "size_usdc", "shares",
                "market_value", "unrealized_pnl", "realized_pnl",
                "net_realized_pnl", "opened_at", "status",
                "confidence", "confidence_at_entry", "signal_label",
            ]).to_csv(self.positions_file, index=False)
            return

        rows = [self._trade_to_dict(t) for t in open_trades]
        pd.DataFrame(rows).to_csv(self.positions_file, index=False)

    def _append_closed_trades(self, closed_trades: List[TradeLifecycle]):
        """
        BUG FIX D: Use actual close_reason from TradeLifecycle, not hardcoded.
        BUG FIX F: Write both realized_pnl AND net_realized_pnl.
        FIX M7: Record wins/losses in MoneyManager for adaptive sizing.
        """
        if not closed_trades:
            return

        # FIX M7: Update MoneyManager with trade outcomes
        try:
            from money_manager import MoneyManager
            _mm = getattr(self, '_money_manager', None)
            if _mm is None:
                _mm = MoneyManager()
                self._money_manager = _mm
            for _ct in closed_trades:
                if _ct.realized_pnl >= 0:
                    _mm.record_win(_ct.realized_pnl)
                else:
                    _mm.record_loss(_ct.realized_pnl)
        except ImportError:
            pass

        rows = []
        for trade in closed_trades:
            row = self._trade_to_dict(trade)
            # ── BUG FIX D: Use actual reason stored on the trade ──
            row["close_reason"] = trade.close_reason or "policy_exit"
            row["exit_price"] = trade.current_price
            # ── BUG FIX F: Write both column names ──
            row["realized_pnl"] = round(trade.realized_pnl, 4)
            row["net_realized_pnl"] = round(trade.realized_pnl, 4)
            row["status"] = "CLOSED"
            rows.append(row)

        pd.DataFrame(rows).to_csv(
            self.closed_file, mode="a",
            header=not self.closed_file.exists(), index=False,
        )

    def reconcile_live_positions(self, execution_client=None, reconciled_positions_df: pd.DataFrame | None = None):
        if reconciled_positions_df is None or reconciled_positions_df.empty:
            logger.info("[~] Reconciling live positions with exchange (no reconciled positions supplied).")
            return

        rebuilt_trades: Dict[str, TradeLifecycle] = {}
        for _, row in reconciled_positions_df.iterrows():
            market = row.get("market") or row.get("market_title") or str(row.get("condition_id") or row.get("token_id") or "unknown_market")
            outcome_side = row.get("outcome_side") or row.get("side") or "UNKNOWN"
            trade_key = f"{market}-{outcome_side}"
            trade = TradeLifecycle(
                market=str(market),
                token_id=row.get("token_id"),
                condition_id=row.get("condition_id"),
                outcome_side=str(outcome_side),
            )
            trade.entry_price = float(row.get("avg_entry_price", row.get("entry_price", 0.0)) or 0.0)
            trade.current_price = float(row.get("mark_price", row.get("current_price", trade.entry_price)) or trade.entry_price)
            trade.shares = float(row.get("shares", 0.0) or 0.0)
            trade.size_usdc = trade.shares * trade.entry_price
            trade.realized_pnl = float(row.get("realized_pnl", 0.0) or 0.0)
            trade.unrealized_pnl = float(row.get("unrealized_pnl", 0.0) or 0.0)
            trade.opened_at = row.get("last_fill_at") or datetime.now().isoformat()
            trade.state = TradeState.OPEN
            rebuilt_trades[trade_key] = trade

        self.active_trades = rebuilt_trades
        logger.info("[~] Reconciled %s live positions into TradeManager.", len(self.active_trades))
