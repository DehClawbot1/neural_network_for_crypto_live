from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import os
from enum import Enum
from pathlib import Path

class RiskTier(Enum):
    SHADOW = 0
    MICRO = 1
    PARTIAL = 2
    SCALED = 3

@dataclass
class RiskDecision:
    allowed: bool
    reason: str

class LiveRiskManager:
    """
    Live-test pre-trade and operational risk controls with persistent lockout blocks.
    """

    def __init__(self, db=None, max_position_size=100.0, max_open_orders=10, max_daily_loss=200.0, max_spread=0.05, cooldown_after_loss_minutes=15, max_failed_orders=10, max_unresolved_exposure=2000.0):
        self.db = db
        self.max_position_size = max_position_size
        self.max_open_orders = max_open_orders
        self.max_daily_loss = max_daily_loss
        self.max_spread = max_spread
        self.cooldown_after_loss_minutes = cooldown_after_loss_minutes
        self.max_failed_orders = max_failed_orders
        self.max_unresolved_exposure = max_unresolved_exposure
        self.last_loss_time = None
        self.failed_orders = 0
        self.lockfile = Path("operational_lockout.lock")
        self.kill_switch = self.lockfile.exists()
        
        # Apply Risk Tiers dynamically from .env
        tier_str = os.getenv("RISK_TIER", "SHADOW").upper()
        self.tier = RiskTier[tier_str] if tier_str in RiskTier.__members__ else RiskTier.SHADOW
        
        # Phase 8: Shadow-Live Validation (Minimum 30 Days)
        # If any primary model artifact is younger than 30 days, force strictly to SHADOW.
        self._enforce_shadow_period()
        
        if self.tier == RiskTier.SHADOW:
            self.max_position_size = 0.0 # Strict physical block
        elif self.tier == RiskTier.MICRO:
            self.max_position_size = min(10.0, self.max_position_size)
            self.max_daily_loss = min(50.0, self.max_daily_loss)

    def _enforce_shadow_period(self):
        import time
        max_age_days = 0.0
        weights_dir = Path("weights")
        if not weights_dir.exists():
            return
            
        found_model = False
        youngest_age = 9999.0
        
        for path in weights_dir.rglob("*.joblib"):
            if "model" in path.name or "classifier" in path.name:
                found_model = True
                age = (time.time() - path.stat().st_mtime) / 86400.0
                if age < youngest_age:
                    youngest_age = age
                    
        if found_model and youngest_age < 30.0:
            if self.tier != RiskTier.SHADOW:
                logging.critical(f"SHADOW VALIDATION ENFORCED: Model artifact is only {youngest_age:.2f} days old. Minimum required is 30 days. Forcing RiskTier.SHADOW.")
            self.tier = RiskTier.SHADOW

    def _evaluate(self, price, size, spread=None, open_orders=0, daily_pnl=0.0, unresolved_exposure=0.0):
        if self.kill_switch or self.lockfile.exists():
            return RiskDecision(False, "operational_lockout_active")
        if self.tier == RiskTier.SHADOW:
            return RiskDecision(False, "shadow_tier_block")
        if float(size or 0.0) > self.max_position_size:
            return RiskDecision(False, "max_position_size_exceeded")
        if unresolved_exposure >= self.max_unresolved_exposure:
             return RiskDecision(False, "max_unresolved_exposure_exceeded")
        if open_orders >= self.max_open_orders:
            return RiskDecision(False, "max_open_orders_exceeded")
        if daily_pnl <= -abs(self.max_daily_loss):
            self.activate_kill_switch("daily_loss_limit_hit")
            return RiskDecision(False, "max_daily_loss_hit_lockout")
        # Live spread/depth enforcement belongs to OrderBookGuard upstream so we
        # do not duplicate or contradict the real-time orderbook gate here.
        if self.last_loss_time and datetime.now(timezone.utc) - self.last_loss_time < timedelta(minutes=self.cooldown_after_loss_minutes):
            return RiskDecision(False, "cooldown_after_loss")
        if self.failed_orders >= self.max_failed_orders:
            self.activate_kill_switch("max_failed_orders_lockout")
            return RiskDecision(False, "circuit_breaker_failed_orders")
        return RiskDecision(True, "ok")

    def pre_trade_check(self, token_id=None, price=0.0, size=0.0, spread=None, open_orders=0, daily_pnl=0.0, unresolved_exposure=0.0):
        decision = self._evaluate(price=price, size=size, spread=spread, open_orders=open_orders, daily_pnl=daily_pnl, unresolved_exposure=unresolved_exposure)
        if not decision.allowed and self.db is not None:
            try:
                detail = f"price={price}, size={size}, spread={spread}, daily_pnl={daily_pnl}"
                self.db.execute(
                    "INSERT INTO risk_events (token_id, event_type, detail) VALUES (?, ?, ?)",
                    (str(token_id) if token_id is not None else None, decision.reason, detail),
                )
                if hasattr(self.db, "commit"): self.db.commit() # BUG FIX 6: Ensure audit logs save
            except Exception as exc:
                logging.error("Failed to log risk event to DB: %s", exc)
        return decision

    def record_failed_order(self):
        self.failed_orders += 1
        if self.failed_orders >= self.max_failed_orders:
            self.activate_kill_switch("max_failed_orders_exceeded")

    def record_successful_order(self):
        pass # INTENTIONAL REMOVAL: Do not auto-reset operational circuit breakers on a single success.

    def record_loss(self):
        self.last_loss_time = datetime.now(timezone.utc)

    def reset_failed_orders(self):
        # Must be called explicitly by operator intervention tools
        self.failed_orders = 0

    def activate_kill_switch(self, reason="manual_activation"):
        self.kill_switch = True
        try:
            self.lockfile.write_text(f"LOCKED at {datetime.now(timezone.utc).isoformat()} - REASON: {reason}")
            logging.critical(f"OPERATIONAL LOCKOUT ACTIVATED. Bot operations frozen. Reason: {reason}.")
        except Exception:
            pass

    def deactivate_kill_switch(self):
        self.kill_switch = False
        self.failed_orders = 0
        if self.lockfile.exists():
            try:
                self.lockfile.unlink()
            except Exception:
                pass
