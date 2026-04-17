"""
deployment_gate.py
──────────────────
Single authority for deployment stage and environment enforcement.

Deployment stages (ordered, opt-in promotion only)
───────────────────────────────────────────────────
  replay        — historical replay, no exchange contact
  paper         — paper trading with live market data, no real orders
  shadow-live   — live market data + order logic runs, but orders are simulated/logged
  micro-live    — real money orders, hard size cap ($5 max per position)
  scaled-live   — real money orders, normal risk limits apply

Environments
────────────
  dev           — local development, no production credentials required
  staging       — integration test environment (can use testnet or paper)
  production    — real money environment, strictest gates apply

Promotion gates
───────────────
Moving between stages requires explicit env var approval AND the correct
deployment environment. A live process CANNOT self-promote.

Hard rules (enforced by raising DeploymentViolationError — never a warning)
────────────────────────────────────────────────────────────────────────────
  A live process (shadow-live and above) MUST NOT:
    1. Train a model
    2. Promote a model
    3. Invent fallback logic for missing artifacts — raise instead
    4. Silently continue after a core dependency failure

Usage
─────
    gate = DeploymentGate.from_env()
    gate.assert_can_place_orders()          # raises if stage too early
    gate.assert_not_training_allowed()      # raises if stage permits training
    gate.assert_can_use_real_money()        # raises if stage too early
    gate.assert_artifact_exists(path, "classifier")  # raises in live if missing
"""
from __future__ import annotations

import logging
import os
import sys
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Enumerations ─────────────────────────────────────────────────────────────

class DeploymentStage(Enum):
    REPLAY      = "replay"
    PAPER       = "paper"
    SHADOW_LIVE = "shadow-live"
    MICRO_LIVE  = "micro-live"
    SCALED_LIVE = "scaled-live"

    def __lt__(self, other: "DeploymentStage") -> bool:
        return _STAGE_ORDER[self] < _STAGE_ORDER[other]

    def __le__(self, other: "DeploymentStage") -> bool:
        return _STAGE_ORDER[self] <= _STAGE_ORDER[other]

    def __gt__(self, other: "DeploymentStage") -> bool:
        return _STAGE_ORDER[self] > _STAGE_ORDER[other]

    def __ge__(self, other: "DeploymentStage") -> bool:
        return _STAGE_ORDER[self] >= _STAGE_ORDER[other]


_STAGE_ORDER = {
    DeploymentStage.REPLAY:      0,
    DeploymentStage.PAPER:       1,
    DeploymentStage.SHADOW_LIVE: 2,
    DeploymentStage.MICRO_LIVE:  3,
    DeploymentStage.SCALED_LIVE: 4,
}


class DeploymentEnvironment(Enum):
    DEV        = "dev"
    STAGING    = "staging"
    PRODUCTION = "production"


class DeploymentViolationError(RuntimeError):
    """
    Raised when a deployment hard rule is violated.

    This must NEVER be caught and silently continued.
    The process must halt.
    """
    def __init__(self, message: str, stage: DeploymentStage, rule: str):
        super().__init__(message)
        self.stage = stage
        self.rule = rule


# ── Promotion approval env vars ───────────────────────────────────────────────
#
# To promote between stages, an operator must set the corresponding env var
# in addition to DEPLOYMENT_STAGE. The gate validates the combination.

_PROMOTION_GATES: dict[DeploymentStage, str] = {
    DeploymentStage.SHADOW_LIVE: "SHADOW_LIVE_APPROVED",
    DeploymentStage.MICRO_LIVE:  "MICRO_LIVE_APPROVED",
    DeploymentStage.SCALED_LIVE: "SCALED_LIVE_APPROVED",
}

# Stages that require a minimum environment level
_REQUIRED_ENV_FOR_STAGE: dict[DeploymentStage, DeploymentEnvironment] = {
    DeploymentStage.SHADOW_LIVE: DeploymentEnvironment.STAGING,
    DeploymentStage.MICRO_LIVE:  DeploymentEnvironment.PRODUCTION,
    DeploymentStage.SCALED_LIVE: DeploymentEnvironment.PRODUCTION,
}

# Hard size caps per stage (USDC per position)
STAGE_MAX_POSITION_USDC: dict[DeploymentStage, float] = {
    DeploymentStage.REPLAY:      0.0,
    DeploymentStage.PAPER:       0.0,
    DeploymentStage.SHADOW_LIVE: 0.0,
    DeploymentStage.MICRO_LIVE:  5.0,    # $5 hard cap per position
    DeploymentStage.SCALED_LIVE: 500.0,  # normal risk limits apply
}


# ── Gate ─────────────────────────────────────────────────────────────────────

class DeploymentGate:
    """
    Runtime authority for deployment stage and environment.

    Constructed once at process startup. All capability checks go through here.
    All violations raise DeploymentViolationError — no silent continuation.
    """

    def __init__(self, stage: DeploymentStage, env: DeploymentEnvironment) -> None:
        self.stage = stage
        self.env   = env
        self._validate_promotion_gate()
        self._log_startup()

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "DeploymentGate":
        """
        Construct a DeploymentGate from environment variables.

        Environment variables
        ─────────────────────
        DEPLOYMENT_STAGE : required
          One of: replay, paper, shadow-live, micro-live, scaled-live
          Default: paper  (live is NOT the default)

        DEPLOYMENT_ENV : required in production
          One of: dev, staging, production
          Default: dev

        Raises
        ──────
        SystemExit(1) if DEPLOYMENT_STAGE is unrecognized.
        DeploymentViolationError if promotion gates are not satisfied.
        """
        raw_stage = os.getenv("DEPLOYMENT_STAGE", "paper").strip().lower()
        raw_env   = os.getenv("DEPLOYMENT_ENV", "dev").strip().lower()

        try:
            stage = DeploymentStage(raw_stage)
        except ValueError:
            valid = [s.value for s in DeploymentStage]
            logger.critical(
                "DEPLOYMENT_STAGE=%r is not a valid stage. Valid values: %s",
                raw_stage, valid,
            )
            print(
                f"\n[FATAL] DEPLOYMENT_STAGE={raw_stage!r} is not valid.\n"
                f"        Valid values: {valid}\n"
                f"        Set DEPLOYMENT_STAGE in your .env file.\n"
            )
            sys.exit(1)

        try:
            environment = DeploymentEnvironment(raw_env)
        except ValueError:
            valid_envs = [e.value for e in DeploymentEnvironment]
            logger.warning(
                "DEPLOYMENT_ENV=%r unrecognized (valid: %s) — defaulting to 'dev'",
                raw_env, valid_envs,
            )
            environment = DeploymentEnvironment.DEV

        return cls(stage=stage, env=environment)

    # ── Capabilities ─────────────────────────────────────────────────────────

    @property
    def is_live(self) -> bool:
        """True for shadow-live, micro-live, and scaled-live."""
        return self.stage >= DeploymentStage.SHADOW_LIVE

    @property
    def uses_real_money(self) -> bool:
        """True only when real orders with real money are placed."""
        return self.stage >= DeploymentStage.MICRO_LIVE

    @property
    def can_train(self) -> bool:
        """Training is allowed only in replay and paper stages."""
        return self.stage <= DeploymentStage.PAPER

    @property
    def can_promote_models(self) -> bool:
        """Model promotion is allowed only in replay and paper stages."""
        return self.stage <= DeploymentStage.PAPER

    @property
    def can_place_orders(self) -> bool:
        """Order placement logic runs in shadow-live and above."""
        return self.stage >= DeploymentStage.SHADOW_LIVE

    @property
    def orders_are_simulated(self) -> bool:
        """In shadow-live, orders are logged but not submitted to the exchange."""
        return self.stage == DeploymentStage.SHADOW_LIVE

    @property
    def max_position_usdc(self) -> float:
        return STAGE_MAX_POSITION_USDC[self.stage]

    # ── Assertion API (hard gates) ────────────────────────────────────────────

    def assert_can_place_orders(self) -> None:
        if not self.can_place_orders:
            raise DeploymentViolationError(
                f"Order placement is not permitted at stage={self.stage.value!r}. "
                f"Promote to shadow-live or above.",
                stage=self.stage,
                rule="can_place_orders",
            )

    def assert_can_use_real_money(self) -> None:
        if not self.uses_real_money:
            raise DeploymentViolationError(
                f"Real-money orders are not permitted at stage={self.stage.value!r}. "
                f"Promote to micro-live with explicit operator approval.",
                stage=self.stage,
                rule="uses_real_money",
            )

    def assert_training_forbidden(self) -> None:
        """
        Raise if training is attempted in a live stage.
        Call this at the start of any model training code path.
        """
        if self.is_live:
            raise DeploymentViolationError(
                f"Model training is FORBIDDEN at stage={self.stage.value!r}. "
                f"A live process must never train a model. "
                f"Run retraining in a separate offline process.",
                stage=self.stage,
                rule="training_forbidden_in_live",
            )

    def assert_promotion_forbidden(self) -> None:
        """
        Raise if model promotion is attempted in a live stage.
        Call this before any model promotion code path.
        """
        if self.is_live:
            raise DeploymentViolationError(
                f"Model promotion is FORBIDDEN at stage={self.stage.value!r}. "
                f"A live process must never promote a model. "
                f"Promote artifacts in staging, then restart the live process.",
                stage=self.stage,
                rule="promotion_forbidden_in_live",
            )

    def assert_artifact_exists(self, path: str | Path, artifact_name: str) -> None:
        """
        In live stages, raise if a required artifact is missing.

        In non-live stages, log a warning and return — fallback is acceptable
        during development.

        This enforces the rule: a live process must never invent fallback
        logic for missing artifacts.
        """
        p = Path(path)
        if p.exists():
            return

        if self.is_live:
            raise DeploymentViolationError(
                f"Required artifact {artifact_name!r} is missing at {p}. "
                f"A live process must never start without all required artifacts. "
                f"Build and promote the artifact in staging first.",
                stage=self.stage,
                rule="artifact_must_exist_in_live",
            )
        else:
            logger.warning(
                "DeploymentGate: artifact %r missing at %s (stage=%s — non-fatal in non-live)",
                artifact_name, p, self.stage.value,
            )

    def assert_no_silent_continuation(self, error: Exception, context: str) -> None:
        """
        In live stages, raise instead of silently continuing after a core failure.

        In non-live stages, log the error and return.

        Usage:
            try:
                critical_operation()
            except Exception as exc:
                gate.assert_no_silent_continuation(exc, "market data fetch")
        """
        if self.is_live:
            raise DeploymentViolationError(
                f"Core dependency failure in {context!r} at stage={self.stage.value!r}: {error}. "
                f"A live process must never silently continue after a core failure.",
                stage=self.stage,
                rule="no_silent_continuation_in_live",
            ) from error
        else:
            logger.warning(
                "DeploymentGate: non-fatal failure in %r (stage=%s): %s",
                context, self.stage.value, error,
            )

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "stage":               self.stage.value,
            "environment":         self.env.value,
            "is_live":             self.is_live,
            "uses_real_money":     self.uses_real_money,
            "can_train":           self.can_train,
            "can_promote_models":  self.can_promote_models,
            "can_place_orders":    self.can_place_orders,
            "orders_are_simulated": self.orders_are_simulated,
            "max_position_usdc":   self.max_position_usdc,
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _validate_promotion_gate(self) -> None:
        """
        Validate that the operator has explicitly approved promotion to this stage.

        Rules:
        - shadow-live: SHADOW_LIVE_APPROVED=true required
        - micro-live:  MICRO_LIVE_APPROVED=true required + DEPLOYMENT_ENV=production
        - scaled-live: SCALED_LIVE_APPROVED=true required + DEPLOYMENT_ENV=production

        Raises SystemExit(1) on failure — this is not a runtime error, it is a
        misconfiguration that must be fixed before the process starts.
        """
        approval_var = _PROMOTION_GATES.get(self.stage)
        if approval_var is None:
            return  # replay and paper need no explicit approval

        approved = os.getenv(approval_var, "false").strip().lower() in {"1", "true", "yes", "on"}
        if not approved:
            print(
                f"\n[FATAL] DEPLOYMENT_STAGE={self.stage.value!r} requires operator approval.\n"
                f"        Set {approval_var}=true in your .env to confirm you understand\n"
                f"        this process will interact with live infrastructure.\n"
            )
            logger.critical(
                "DeploymentGate: promotion to %r blocked — %s not set",
                self.stage.value, approval_var,
            )
            sys.exit(1)

        # Check minimum environment requirement
        required_env = _REQUIRED_ENV_FOR_STAGE.get(self.stage)
        if required_env is not None:
            env_order = {DeploymentEnvironment.DEV: 0, DeploymentEnvironment.STAGING: 1, DeploymentEnvironment.PRODUCTION: 2}
            if env_order[self.env] < env_order[required_env]:
                print(
                    f"\n[FATAL] DEPLOYMENT_STAGE={self.stage.value!r} requires\n"
                    f"        DEPLOYMENT_ENV={required_env.value!r} or higher.\n"
                    f"        Current DEPLOYMENT_ENV={self.env.value!r}.\n"
                )
                logger.critical(
                    "DeploymentGate: stage %r requires env >= %r, got %r",
                    self.stage.value, required_env.value, self.env.value,
                )
                sys.exit(1)

    def _log_startup(self) -> None:
        s = self.summary()
        if self.is_live:
            logger.warning(
                "DeploymentGate: LIVE stage=%r env=%r | real_money=%s | simulated=%s | max_pos=$%.0f",
                s["stage"], s["environment"],
                s["uses_real_money"], s["orders_are_simulated"], s["max_position_usdc"],
            )
        else:
            logger.info(
                "DeploymentGate: stage=%r env=%r | can_train=%s | can_promote=%s",
                s["stage"], s["environment"], s["can_train"], s["can_promote_models"],
            )


# ── Module-level singleton ────────────────────────────────────────────────────
#
# Callers may use the singleton if they don't construct their own gate.
# It is lazily initialized on first access.

_gate_singleton: DeploymentGate | None = None


def get_deployment_gate() -> DeploymentGate:
    """Return the process-level DeploymentGate singleton."""
    global _gate_singleton
    if _gate_singleton is None:
        _gate_singleton = DeploymentGate.from_env()
    return _gate_singleton


def reset_gate_singleton() -> None:
    """Reset singleton — for use in tests only."""
    global _gate_singleton
    _gate_singleton = None


if __name__ == "__main__":
    import os
    from unittest.mock import patch

    # Test: default (paper stage, dev env) — no approvals needed
    with patch.dict(os.environ, {"DEPLOYMENT_STAGE": "paper", "DEPLOYMENT_ENV": "dev"}, clear=False):
        reset_gate_singleton()
        g = get_deployment_gate()
        assert g.stage == DeploymentStage.PAPER
        assert not g.is_live
        assert g.can_train
        assert not g.can_place_orders
        assert not g.uses_real_money
        print(f"  paper/dev: {g.summary()}")

    # Test: replay
    with patch.dict(os.environ, {"DEPLOYMENT_STAGE": "replay", "DEPLOYMENT_ENV": "dev"}, clear=False):
        reset_gate_singleton()
        g = DeploymentGate.from_env()
        assert g.can_train
        assert not g.is_live
        print(f"  replay/dev: OK")

    # Test: shadow-live requires approval + staging env
    with patch.dict(os.environ, {
        "DEPLOYMENT_STAGE": "shadow-live",
        "DEPLOYMENT_ENV": "staging",
        "SHADOW_LIVE_APPROVED": "true",
    }, clear=False):
        reset_gate_singleton()
        g = DeploymentGate.from_env()
        assert g.is_live
        assert not g.uses_real_money
        assert g.orders_are_simulated
        assert not g.can_train
        assert not g.can_promote_models
        print(f"  shadow-live/staging: {g.summary()}")

    # Test: shadow-live raises DeploymentViolationError on training attempt
    with patch.dict(os.environ, {
        "DEPLOYMENT_STAGE": "shadow-live",
        "DEPLOYMENT_ENV": "staging",
        "SHADOW_LIVE_APPROVED": "true",
    }, clear=False):
        reset_gate_singleton()
        g = DeploymentGate.from_env()
        try:
            g.assert_training_forbidden()
            assert False, "should have raised"
        except DeploymentViolationError as e:
            assert e.rule == "training_forbidden_in_live"
            print(f"  shadow-live training blocked: {e}")

    # Test: micro-live requires approval + production env
    with patch.dict(os.environ, {
        "DEPLOYMENT_STAGE": "micro-live",
        "DEPLOYMENT_ENV": "production",
        "MICRO_LIVE_APPROVED": "true",
    }, clear=False):
        reset_gate_singleton()
        g = DeploymentGate.from_env()
        assert g.uses_real_money
        assert g.max_position_usdc == 5.0
        print(f"  micro-live/production: max_pos=${g.max_position_usdc}")

    # Test: paper stage asserts can_place_orders raises
    with patch.dict(os.environ, {"DEPLOYMENT_STAGE": "paper", "DEPLOYMENT_ENV": "dev"}, clear=False):
        reset_gate_singleton()
        g = DeploymentGate.from_env()
        try:
            g.assert_can_place_orders()
            assert False
        except DeploymentViolationError as e:
            assert e.rule == "can_place_orders"
            print(f"  paper can_place_orders blocked: OK")

    reset_gate_singleton()
    print("deployment_gate self-test PASSED.")
