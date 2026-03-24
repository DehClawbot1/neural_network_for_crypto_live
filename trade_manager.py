from __future__ import annotations

import logging
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd

from trade_lifecycle import TradeLifecycle, TradeState
from config import TradingConfig

logger = logging.getLogger(__name__)


class TradeManager:
    """
    Manages a collection of TradeLifecycle instances, acting as the central
    runtime trade manager for the supervisor.
    """

    def __init__(self):
        self.active_trades: Dict[str, TradeLifecycle] = {}
        logger.info("[+] Initialized TradeManager.")

    def _get_trade_key(self, signal_row: pd.Series) -> Optional[str]:
        """Generates a unique key for a trade based on market and outcome.

        BUG FIX: Use the same column priority as the supervisor so keys
        are consistent.  Return None instead of raising on missing fields.
        """
        market = signal_row.get("market_title") or signal_row.get("market")
        outcome_side = signal_row.get("outcome_side") or signal_row.get("side")
        if not market or not outcome_side:
            logger.warning("Missing market or outcome_side in signal: %s", dict(signal_row) if hasattr(signal_row, 'items') else signal_row)
            return None
        return f"{market}-{outcome_side}"

    def handle_signal(self, signal_row: pd.Series, confidence: float, size_usdc: float) -> Optional[TradeLifecycle]:
        """
        Processes a new signal. If a trade already exists for this market/side,
        it updates it. Otherwise, it creates a new TradeLifecycle instance.
        """
        trade_key = self._get_trade_key(signal_row)
        if trade_key is None:
            return None

        market = signal_row.get("market_title") or signal_row.get("market")
        outcome_side = signal_row.get("outcome_side") or signal_row.get("side")

        # BUG FIX: use "token_id" not "outcome_token_id" — that key doesn't
        # exist anywhere in the pipeline.
        token_id = signal_row.get("token_id")
        condition_id = signal_row.get("condition_id")

        # BUG FIX: the scoring pipeline produces "current_price" / "entry_price",
        # not bare "price".  Fall through the chain so we never get None.
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
            trade.enter(size_usdc=size_usdc, entry_price=entry_price)
            self.active_trades[trade_key] = trade
            logger.info("[+] New trade initiated for %s (%s) with %s USDC at %.4f.",
                        market, outcome_side, size_usdc, entry_price)
            return trade

    def update_markets(self, market_prices: Dict[str, float]):
        """
        Updates current prices for all active trades and computes unrealized PnL.
        market_prices: a dict of {market_name: current_price}
        """
        for trade_key, trade in list(self.active_trades.items()):
            if trade.market in market_prices:
                live_price = market_prices[trade.market]
                trade.update_market(live_price)
            # Also try looking up by the full key if market name alone didn't match
            # (handles cases where market_prices uses market_title)

    def process_exits(self, current_timestamp: datetime, alerts_df: pd.DataFrame = None):
        """
        Evaluates exit conditions for all active trades.

        BUG FIX: Added real exit logic (take-profit, stop-loss, trailing stop,
        time stop) instead of only the 7-day placeholder.
        """
        closed_trades: List[TradeLifecycle] = []

        for trade_key, trade in list(self.active_trades.items()):
            if trade.state == TradeState.CLOSED:
                closed_trades.append(trade)
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

            # Take-profit on ROI
            if roi >= TradingConfig.PAPER_TP_ROI:
                close_reason = "take_profit_roi"

            # Take-profit on absolute price move
            elif (current_price - entry_price) >= TradingConfig.SHADOW_TP_DELTA:
                close_reason = "take_profit_price_move"

            # Stop-loss on absolute price move
            elif (entry_price - current_price) >= TradingConfig.SHADOW_SL_DELTA:
                close_reason = "stop_loss"

            # Time stop — 3 hours for paper trades
            elif minutes_open >= 180:
                close_reason = "time_stop"

            # Trailing stop (basic: if we've been positive and dropped back)
            # Note: TradeLifecycle doesn't track peak_price, so this is approximate
            elif roi < -TradingConfig.PAPER_TRAILING_STOP and minutes_open > 15:
                close_reason = "trailing_stop"

            if close_reason:
                trade.close(current_price)
                logger.info("[->] Closed trade for %s (%s). Reason: %s. PnL: %.4f",
                            trade.market, trade.outcome_side, close_reason, trade.realized_pnl)
                closed_trades.append(trade)

        # Remove closed trades from active set
        for trade in closed_trades:
            key = f"{trade.market}-{trade.outcome_side}"
            self.active_trades.pop(key, None)

        return closed_trades

    def get_open_positions(self) -> List[TradeLifecycle]:
        """Returns a list of all currently active TradeLifecycle instances."""
        return [t for t in self.active_trades.values() if t.state != TradeState.CLOSED]

    def get_metrics(self) -> Dict[str, any]:
        """Aggregates key metrics from active trades."""
        open_trades = self.get_open_positions()
        total_unrealized_pnl = sum(trade.unrealized_pnl for trade in open_trades)
        return {
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_open_positions": len(open_trades),
            "last_reconciled_at": datetime.now().isoformat(),
        }

    def reconcile_live_positions(self, execution_client=None, reconciled_positions_df: pd.DataFrame | None = None):
        """
        Reconcile active trades from a reconciled live position book when available.
        """
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
