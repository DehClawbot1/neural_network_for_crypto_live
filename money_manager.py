"""
money_manager.py
================
Intelligent bet sizing based on available balance, confidence, and risk limits.

Rules:
  - Never bet more than MAX_RISK_PER_TRADE_PCT of available balance
  - Scale bet size by confidence level
  - Never bet below MIN_BET_USDC (skip the trade instead)
  - Never bet above MAX_BET_USDC (cap it)
  - Track total exposure and reject if MAX_TOTAL_EXPOSURE_PCT exceeded
  - Learn from wins/losses: reduce size after consecutive losses
"""

import logging

from config import TradingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MoneyManager:
    def __init__(self):
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.total_trades = 0
        self.total_pnl = 0.0

    def record_win(self, pnl: float):
        self.consecutive_wins += 1
        self.consecutive_losses = 0
        self.total_trades += 1
        self.total_pnl += pnl

    def record_loss(self, pnl: float):
        self.consecutive_losses += 1
        self.consecutive_wins = 0
        self.total_trades += 1
        self.total_pnl += pnl

    def _loss_reduction_factor(self) -> float:
        """Reduce bet size after consecutive losses (Kelly-inspired)."""
        if self.consecutive_losses <= 1:
            return 1.0
        if self.consecutive_losses == 2:
            return 0.75
        if self.consecutive_losses == 3:
            return 0.50
        return 0.25  # After 4+ losses, bet 25% of normal

    def _confidence_bet_pct(self, confidence: float) -> float:
        """Map confidence to bet size as % of balance."""
        if confidence >= 0.70:
            return TradingConfig.HIGH_CONFIDENCE_BET_PCT
        if confidence >= 0.50:
            return TradingConfig.MEDIUM_CONFIDENCE_BET_PCT
        return TradingConfig.LOW_CONFIDENCE_BET_PCT

    def calculate_bet_size(
        self,
        available_balance: float,
        confidence: float,
        current_exposure: float = 0.0,
    ) -> float:
        """
        Calculates a DYNAMIC bet size that scales infinitely with account balance.
        Increases bets on profit, shrinks bets on drawdowns, and ensures capital preservation.
        """
        if available_balance <= 0:
            logging.warning("MoneyManager: No balance available ($%.2f)", available_balance)
            return 0.0

        reserve_pct = max(0.0, min(0.95, float(getattr(TradingConfig, "CAPITAL_RESERVE_PCT", 0.20))))
        reserve_usdc = available_balance * reserve_pct
        tradable_balance = max(0.0, available_balance - reserve_usdc)

        min_bet_usdc = float(getattr(TradingConfig, "MIN_BET_USDC", 1.0))
        if tradable_balance < min_bet_usdc:
            logging.info(
                "MoneyManager: Tradable balance below minimum bet (tradable=$%.2f, min=$%.2f).",
                tradable_balance, min_bet_usdc,
            )
            return 0.0

        # Check total exposure limit (e.g. 85% of total capital)
        max_total_exposure = (available_balance + current_exposure) * TradingConfig.MAX_TOTAL_EXPOSURE_PCT # BUG FIX 2: Compute against total portfolio
        remaining_capacity = max_total_exposure - current_exposure
        if remaining_capacity <= 0:
            logging.info(
                "MoneyManager: Exposure limit reached. Current=$%.2f, Max=$%.2f",
                current_exposure, max_total_exposure
            )
            return 0.0

        # Base bet: confidence-scaled % of current balance (Naturally scales with profit/loss)
        base_pct = self._confidence_bet_pct(confidence)
        base_bet = tradable_balance * base_pct

        # Apply loss reduction (Protects against losing streaks)
        loss_factor = self._loss_reduction_factor()
        adjusted_bet = base_bet * loss_factor

        # --- DYNAMIC LIMITS ---
        absolute_floor = min_bet_usdc # Exchange absolute minimum
        
        # Dynamic minimum grows slowly with account size but remains capped.
        dynamic_min_pct = max(0.0, float(getattr(TradingConfig, "MIN_BET_DYNAMIC_PCT", 0.002)))
        dynamic_min_cap = max(absolute_floor, float(getattr(TradingConfig, "MAX_DYNAMIC_MIN_BET_USDC", 5.0)))
        dynamic_min = max(absolute_floor, min(tradable_balance * dynamic_min_pct, dynamic_min_cap))
        
        # Dynamic Max: percentage of tradable balance + hard absolute safety cap.
        dynamic_max = tradable_balance * getattr(TradingConfig, 'MAX_RISK_PER_TRADE_PCT', 0.15)
        hard_max = max(absolute_floor, float(getattr(TradingConfig, "HARD_MAX_BET_USDC", 250.0)))
        dynamic_max = min(dynamic_max, hard_max)

        # Cap the bet at the dynamic max and remaining capacity
        adjusted_bet = min(adjusted_bet, dynamic_max)
        adjusted_bet = min(adjusted_bet, remaining_capacity)

        # Enforce the dynamic minimum
        if dynamic_max < absolute_floor:
            logging.info("MoneyManager: dynamic_max ($%.2f) < absolute_floor ($%.2f). Rejecting trade.", dynamic_max, absolute_floor)
            return 0.0
            
        if adjusted_bet < dynamic_min:
            if remaining_capacity >= dynamic_min:
                adjusted_bet = min(dynamic_min, dynamic_max) # BUG FIX 3: Never break max risk for the sake of minimums
            else:
                return 0.0

        adjusted_bet = round(adjusted_bet, 2)

        logging.info(
            "MoneyManager: balance=$%.2f reserve=$%.2f tradable=$%.2f conf=%.2f base_pct=%.1f%% loss_factor=%.2f -> final_bet=$%.2f (min=$%.2f, max=$%.2f)",
            available_balance, reserve_usdc, tradable_balance, confidence, base_pct * 100, loss_factor, adjusted_bet, dynamic_min, dynamic_max
        )

        return adjusted_bet

    def get_status(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "total_pnl": round(self.total_pnl, 4),
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "loss_reduction_factor": self._loss_reduction_factor(),
        }
