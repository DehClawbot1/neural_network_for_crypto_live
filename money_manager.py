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
        Calculate the optimal bet size in USDC.

        Args:
            available_balance: Current available USDC balance
            confidence: Model confidence (0.0 - 1.0)
            current_exposure: Total USDC currently in open positions

        Returns:
            Bet size in USDC, or 0.0 if trade should be skipped
        """
        if available_balance <= 0:
            logging.warning("MoneyManager: No balance available ($%.2f)", available_balance)
            return 0.0

        # Check total exposure limit
        max_total_exposure = available_balance * TradingConfig.MAX_TOTAL_EXPOSURE_PCT
        remaining_capacity = max_total_exposure - current_exposure
        if remaining_capacity <= 0:
            logging.info(
                "MoneyManager: Exposure limit reached. "
                "Current=$%.2f, Max=$%.2f (%.0f%% of $%.2f)",
                current_exposure, max_total_exposure,
                TradingConfig.MAX_TOTAL_EXPOSURE_PCT * 100, available_balance
            )
            return 0.0

        # Base bet: confidence-scaled % of balance
        base_pct = self._confidence_bet_pct(confidence)
        base_bet = available_balance * base_pct

        # Apply loss reduction
        loss_factor = self._loss_reduction_factor()
        adjusted_bet = base_bet * loss_factor

        # Apply absolute caps
        max_per_trade = available_balance * TradingConfig.MAX_RISK_PER_TRADE_PCT
        adjusted_bet = min(adjusted_bet, max_per_trade)
        adjusted_bet = min(adjusted_bet, TradingConfig.MAX_BET_USDC)
        adjusted_bet = min(adjusted_bet, remaining_capacity)

        # Check minimum — Polymarket CLOB requires at least $1 per order.
        # If bet is between $0.80 and MIN_BET, round up to MIN_BET instead of skipping.
        if adjusted_bet < TradingConfig.MIN_BET_USDC:
            if adjusted_bet >= TradingConfig.MIN_BET_USDC * 0.8 and remaining_capacity >= TradingConfig.MIN_BET_USDC:
                adjusted_bet = TradingConfig.MIN_BET_USDC
                logging.info(
                    "MoneyManager: Rounded up $%.2f → $%.2f (CLOB minimum)",
                    adjusted_bet, TradingConfig.MIN_BET_USDC
                )
            else:
                logging.info(
                    "MoneyManager: Bet $%.2f below minimum $%.2f — skipping trade",
                    adjusted_bet, TradingConfig.MIN_BET_USDC
                )
                return 0.0

        # Round to 2 decimal places
        adjusted_bet = round(adjusted_bet, 2)

        logging.info(
            "MoneyManager: balance=$%.2f conf=%.2f base_pct=%.1f%% "
            "loss_factor=%.2f -> bet=$%.2f",
            available_balance, confidence, base_pct * 100,
            loss_factor, adjusted_bet
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
