"""
supervisor_betting_patch.py
============================
Patches the supervisor's entry logic to:
  1. Use FOK market orders instead of limit orders (faster fills for BTC 5-min)
  2. Use MoneyManager for intelligent bet sizing
  3. Properly read normalized balance before betting
  4. Focus on Bitcoin markets

Apply by importing and calling apply_supervisor_betting_patch(supervisor_module)
BEFORE main_loop() starts.
"""

import logging
import os
from datetime import datetime

from config import TradingConfig
from money_manager import MoneyManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_money_manager = MoneyManager()


def apply_supervisor_betting_patch(supervisor_module):
    """Patch the supervisor to use market orders + money management."""
    if getattr(supervisor_module, "_betting_patch_applied", False):
        return

    # Store reference to original functions
    _original_execute_paper_trade = supervisor_module.execute_paper_trade
    _original_log_live_fill_event = supervisor_module.log_live_fill_event

    def patched_calculate_bet_size(signal_row, order_manager=None):
        """
        Calculate bet size using MoneyManager instead of fixed $10/$50.
        """
        confidence = float(signal_row.get("confidence", 0.0) or 0.0)

        # Get available balance
        available_balance = 0.0
        if order_manager is not None:
            try:
                available_balance, _ = order_manager._get_available_balance(asset_type="COLLATERAL")
            except Exception as exc:
                logging.warning("Failed to get balance for bet sizing: %s", exc)
                # Try the execution client directly
                try:
                    available_balance = order_manager.client.get_available_balance(asset_type="COLLATERAL")
                except Exception:
                    pass

        if available_balance <= 0:
            logging.warning("Cannot calculate bet size: balance=$%.2f", available_balance)
            return 0.0

        # Calculate current exposure from open trades
        current_exposure = 0.0
        try:
            trade_manager = getattr(supervisor_module, '_current_trade_manager', None)
            if trade_manager is not None:
                open_trades = trade_manager.get_open_positions()
                current_exposure = sum(float(t.size_usdc or 0) for t in open_trades)
        except Exception:
            pass

        bet_size = _money_manager.calculate_bet_size(
            available_balance=available_balance,
            confidence=confidence,
            current_exposure=current_exposure,
        )

        return bet_size

    def patched_submit_live_entry(signal_row, order_manager, size_usdc):
        """
        Submit a live entry using FOK market orders when configured.
        Falls back to limit orders if market orders fail.
        """
        token_id = str(signal_row.get("token_id", "") or "")
        outcome_side = signal_row.get("outcome_side", signal_row.get("side", "YES"))
        condition_id = signal_row.get("condition_id")

        if not token_id:
            logging.warning("Cannot submit entry: missing token_id")
            return None, None

        if TradingConfig.USE_MARKET_ORDERS:
            # Use FOK market order (matching tutorial pattern)
            logging.info(
                "Submitting FOK market order: token=%s amount=$%.2f side=BUY",
                token_id[:16], size_usdc
            )
            try:
                entry_row, entry_response = order_manager.submit_market_entry(
                    token_id=token_id,
                    amount=size_usdc,
                    side="BUY",
                    condition_id=condition_id,
                    outcome_side=outcome_side,
                )
                return entry_row, entry_response
            except Exception as exc:
                logging.warning(
                    "Market order failed for %s: %s. Falling back to limit order.",
                    token_id[:16], exc
                )

        # Fallback: limit order
        fill_price = supervisor_module.quote_entry_price(signal_row)
        entry_row, entry_response = order_manager.submit_entry(
            token_id=token_id,
            price=fill_price,
            size=size_usdc,
            side=signal_row.get("order_side", "BUY"),
            condition_id=condition_id,
            outcome_side=outcome_side,
        )
        return entry_row, entry_response

    # Attach to supervisor module for access from main_loop
    supervisor_module.patched_calculate_bet_size = patched_calculate_bet_size
    supervisor_module.patched_submit_live_entry = patched_submit_live_entry
    supervisor_module._money_manager = _money_manager
    supervisor_module._betting_patch_applied = True

    logging.info("[+] Applied supervisor betting patch: market orders + money management")
    return supervisor_module
