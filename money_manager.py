"""
money_manager.py
================
Intelligent bet sizing based on family sleeves, additive quality scores, and multiplicative penalties.
"""

import logging
from datetime import datetime, timezone
import numpy as np

from config import TradingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MoneyManager:
    def __init__(self, family: str, sleeve_pct: float = 1.0):
        self.family = family.lower()
        self.sleeve_pct = max(0.01, min(1.0, sleeve_pct))
        
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        
        self.daily_loss = 0.0
        self.weekly_loss = 0.0
        self.current_day_str = ""
        self.current_week_str = ""

    def record_trade_result(self, pnl: float, ts_iso: str = None):
        """Records win/loss strictly against the assigned family sleeve state."""
        if ts_iso is None:
            ts_iso = datetime.now(timezone.utc).isoformat()
        try:
            dt = datetime.fromisoformat(ts_iso)
            if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            dt = datetime.now(timezone.utc)
            
        day_str = dt.strftime("%Y-%m-%d")
        week_str = dt.strftime("%Y-%W")
        
        if day_str != self.current_day_str:
            self.daily_loss = 0.0
            self.current_day_str = day_str
            
        if week_str != self.current_week_str:
            self.weekly_loss = 0.0
            self.current_week_str = week_str

        # Win / Loss streak
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        elif pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.daily_loss += abs(pnl) # Accumulate absolute loss cleanly
            self.weekly_loss += abs(pnl)
            
        self.total_trades += 1
        self.total_pnl += pnl

    def _get_drawdown_penalty(self) -> float:
        if self.consecutive_losses <= 1: return 1.0
        if self.consecutive_losses == 2: return 0.75
        if self.consecutive_losses == 3: return 0.50
        return 0.25 # Max clamp

    def _is_kill_switched(self, max_daily_loss: float, max_weekly_loss: float) -> bool:
        if max_daily_loss > 0 and self.daily_loss >= max_daily_loss:
            return True
        if max_weekly_loss > 0 and self.weekly_loss >= max_weekly_loss:
            return True
        return False

    def _safe_float(self, val, fallback) -> float:
        try:
            v = float(val)
            if np.isnan(v) or np.isinf(v):
                return float(fallback)
            return v
        except Exception:
            return float(fallback)

    def calculate_bet_size(
        self,
        available_balance: float,
        current_exposure: float = 0.0,
        # Execution / Safety Constraints
        confidence: float = 0.0,
        uncertainty: float = 0.0,
        edge: float = 0.0,
        ev_after_cost: float = 0.0,
        fill_probability: float = 0.0,
        # Context constraints
        realized_volatility: float = 0.0,
        hours_to_resolution: float = 0.0,
        market_liquidity_score: float = 1.0,
        cluster_exposure_usdc: float = 0.0,
        active_correlated_positions: int = 0,
        # Environmental Overrides
        target_vol: float = 0.05,
        floor_vol: float = 0.02,
        cluster_cap_usdc: float = 250.0,
        max_daily_loss: float = 100.0,
        max_weekly_loss: float = 300.0,
        max_family_drawdown: float = 500.0,
        max_portfolio_drawdown: float = 1000.0,
        global_portfolio_drawdown: float = 0.0,
    ) -> dict:
        """
        Calculates size by returning a complete breakdown of multipliers, penalties, and final capped size.
        """
        # Guardrail 2: missing metrics force conservative defaults.
        # Edge, Ev, Fill automatically fall back to 0.0 if not provided or NaN
        confidence = self._safe_float(confidence, 0.0)
        uncertainty = self._safe_float(uncertainty, 0.5)
        edge = self._safe_float(edge, 0.0)
        ev_after_cost = self._safe_float(ev_after_cost, 0.0)
        fill_probability = self._safe_float(fill_probability, 0.0)
        
        # 1. Family Capital Allocation
        global_reserve_pct = max(0.0, min(0.95, float(getattr(TradingConfig, "CAPITAL_RESERVE_PCT", 0.20))))
        tradable_global = max(0.0, available_balance * (1.0 - global_reserve_pct))
        
        sleeve_capacity = tradable_global * self.sleeve_pct
        sleeve_available = max(0.0, sleeve_capacity - current_exposure)

        decomposition = {
            "family": self.family,
            "sleeve_capacity": round(sleeve_capacity, 2),
            "sleeve_available": round(sleeve_available, 2),
            "edge_component": edge,
            "ev_component": ev_after_cost,
            "fill_prob_component": fill_probability,
            "confidence_component": confidence,
            "uncertainty_component": uncertainty,
            "quality_score": 0.0,
            "volatility_penalty": 1.0,
            "drag_penalty": 1.0,
            "liquidity_penalty": 1.0,
            "drawdown_penalty": 1.0,
            "cluster_penalty": 1.0,
            "correlation_penalty": 1.0,
            "uncertainty_penalty": 1.0,
            "risk_of_ruin_penalty": 1.0,
            "final_size": 0.0,
            "reason": None
        }

        if sleeve_available <= 0:
            decomposition["reason"] = "sleeve_depleted"
            return decomposition

        if self._is_kill_switched(max_daily_loss, max_weekly_loss):
            decomposition["reason"] = "kill_switch_active"
            return decomposition
            
        if self.weekly_loss >= max_family_drawdown:
            decomposition["reason"] = "max_family_drawdown_exceeded"
            return decomposition
            
        if global_portfolio_drawdown >= max_portfolio_drawdown:
            decomposition["reason"] = "max_portfolio_drawdown_exceeded"
            return decomposition

        # 2. Additive Quality Score (Guardrail 1: 35/35/20/10)
        # Normalize the metrics strictly to 0-1 range for uniform combination.
        n_edge = max(0.0, min(1.0, edge * 20.0))  # e.g. 5% true edge hits max quality scale
        n_ev = max(0.0, min(1.0, ev_after_cost * 20.0))
        n_fill = max(0.0, min(1.0, fill_probability))
        n_conf = max(0.0, min(1.0, confidence * 4.0)) # map standard 0.25 PM probability to 1.0 scale

        qs = (n_edge * 0.35) + (n_ev * 0.35) + (n_fill * 0.20) + (n_conf * 0.10)
        qs = max(0.0, min(1.0, qs))
        decomposition["quality_score"] = round(qs, 4)

        if qs < 0.05: # Minimal quality floor
            decomposition["reason"] = "quality_score_too_low"
            return decomposition

        # Default single trade hard cap sizing inside sleeve
        # (Weather operates on a tighter 3% single position margin, BTC larger 6%)
        sleeve_single_trade_cap_pct = 0.06 if "btc" in self.family else 0.03
        base_size = sleeve_capacity * sleeve_single_trade_cap_pct * qs

        # 3. Multiplicative Penalty Stack
        
        # A) Drawdown Penalty
        dd_penalty = self._get_drawdown_penalty()
        decomposition["drawdown_penalty"] = dd_penalty
        
        # B) Liquidity Penalty
        liq = max(0.5, min(1.0, self._safe_float(market_liquidity_score, 1.0)))
        decomposition["liquidity_penalty"] = liq
        
        # Family-specific limits
        vol_penalty = 1.0
        drag_penalty = 1.0
        cluster_penalty = 1.0

        if "btc" in self.family:
            rv = self._safe_float(realized_volatility, floor_vol)
            v_pen = target_vol / max(rv, floor_vol)
            vol_penalty = max(0.50, min(1.25, v_pen))
            decomposition["volatility_penalty"] = round(vol_penalty, 4)
        else:
            # Weather Drag Target (Guardrail 5)
            hr = self._safe_float(hours_to_resolution, 24.0)
            if hr > 72.0: drag_penalty = 0.50
            elif hr > 48.0: drag_penalty = 0.65
            elif hr > 24.0: drag_penalty = 0.85
            else: drag_penalty = 1.00
            decomposition["drag_penalty"] = drag_penalty
            
            # Weather Cluster (Guardrail 6)
            cluster_usdc = self._safe_float(cluster_exposure_usdc, 0.0)
            if cluster_usdc >= cluster_cap_usdc:
                decomposition["cluster_penalty"] = 0.0
                decomposition["reason"] = "cluster_cap_exceeded"
                decomposition["final_size"] = 0.0
                return decomposition
            elif cluster_usdc >= (cluster_cap_usdc * 0.5):
                cluster_penalty = 0.5
            decomposition["cluster_penalty"] = cluster_penalty

        # 4. Final Application and Hard Caps (Guardrail 3)
        
        # Uncertainty Multiplier (decay extremely fast as uncertainty moves above 0.25)
        uncertainty_penalty = max(0.0, min(1.0, 1.0 - (uncertainty * 2.0)))
        decomposition["uncertainty_penalty"] = round(uncertainty_penalty, 4)
        
        # Risk of Ruin basic bound
        # A simple bounding metric: Kelly assumes f* = edge/odds. If we limit maximum bankroll exposure to 
        # prevent touching ruin margins. We enforce a 0.25 multiplier if edge is virtually static.
        risk_of_ruin_penalty = 1.0 if edge > 0.01 else 0.25
        decomposition["risk_of_ruin_penalty"] = risk_of_ruin_penalty
        
        # Correlation limit
        correlation_penalty = 1.0
        if int(active_correlated_positions) > 0:
            correlation_penalty = 1.0 / (int(active_correlated_positions) + 1)
        decomposition["correlation_penalty"] = round(correlation_penalty, 4)
        
        combined_multiplier = dd_penalty * liq * vol_penalty * drag_penalty * cluster_penalty * uncertainty_penalty * risk_of_ruin_penalty * correlation_penalty
        combined_multiplier = max(0.0, min(1.50, combined_multiplier))
        
        final_size = base_size * combined_multiplier

        # Apply ultimate caps vs available
        min_bet = float(getattr(TradingConfig, "MIN_BET_USDC", 1.0))
        final_size = min(final_size, sleeve_available)

        if final_size < min_bet:
            decomposition["reason"] = "below_exchange_minimum"
            decomposition["final_size"] = 0.0
            return decomposition

        decomposition["final_size"] = round(final_size, 2)
        decomposition["reason"] = "approved"
        
        return decomposition

    def get_status(self) -> dict:
        return {
            "family": self.family,
            "total_trades": self.total_trades,
            "total_pnl": round(self.total_pnl, 4),
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "daily_loss": round(self.daily_loss, 4),
            "weekly_loss": round(self.weekly_loss, 4)
        }
