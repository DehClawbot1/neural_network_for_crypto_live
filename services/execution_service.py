"""
services/execution_service.py
──────────────────────────────
Order placement with explicit lifecycle tracking.

Rules
─────
1. Every order is tracked through the OrderLifecycle state machine.
   No order can be submitted without first entering the registry.
2. ExecutionFaultError is raised on unrecoverable faults.
   Callers MUST NOT catch it and continue silently.
3. This service never makes trading decisions.
   It only executes what is handed to it.
4. On fault, the order is moved to OrderState.ERROR before raising.
   The cycle must halt for that candidate.

Usage
─────
    svc = ExecutionService(execution_client)
    result = svc.place_order(
        token_id="abc",
        side=OrderSide.BUY,
        price=0.55,
        size_usdc=10.0,
        order_type="GTC",
    )
    if result["success"]:
        fill = result["fill"]
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from services.order_lifecycle import OrderLifecycle, OrderRegistry
from services.types import (
    ExecutionFaultError,
    ExecutionResult,
    FillRecord,
    FillStatus,
    OrderSide,
    OrderState,
)

logger = logging.getLogger(__name__)

# Fill polling defaults (overridable via constructor)
_DEFAULT_POLL_INTERVAL_SEC = 2.0
_DEFAULT_FILL_TIMEOUT_SEC  = 30.0
_DEFAULT_MAX_RETRIES       = 2


class ExecutionService:
    """
    Places and tracks orders using the OrderLifecycle state machine.

    This service wraps ExecutionClient with explicit state tracking.
    Every submitted order is registered, transitioned, and resolved
    to a terminal state before returning.

    It never makes trading decisions — only executes what is passed in.
    """

    def __init__(
        self,
        execution_client,
        *,
        poll_interval_sec: float = _DEFAULT_POLL_INTERVAL_SEC,
        fill_timeout_sec: float = _DEFAULT_FILL_TIMEOUT_SEC,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        dry_run: bool = False,
    ) -> None:
        self._client = execution_client
        self._registry = OrderRegistry()
        self._poll_interval = poll_interval_sec
        self._fill_timeout = fill_timeout_sec
        self._max_retries = max_retries
        self._dry_run = dry_run

    # ── Primary API ───────────────────────────────────────────────────────────

    def place_order(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size_usdc: float,
        *,
        order_type: str = "GTC",
        options: Optional[dict] = None,
    ) -> ExecutionResult:
        """
        Place a limit order and wait for fill or terminal state.

        Parameters
        ──────────
        token_id   : Polymarket token_id (YES or NO side)
        side       : OrderSide.BUY or OrderSide.SELL
        price      : limit price in [0.02, 0.97]
        size_usdc  : notional size in USDC (dollar amount, not shares)
        order_type : "GTC" (default) or "GTD"
        options    : optional dict forwarded to ExecutionClient (tick_size, neg_risk, etc.)

        Returns
        ───────
        ExecutionResult TypedDict.

        Raises
        ──────
        ExecutionFaultError : on unrecoverable fault (auth failure, unexpected exception, etc.)
                              The order will be in OrderState.ERROR before this is raised.
        """
        if size_usdc <= 0:
            raise ExecutionFaultError(
                f"Invalid size_usdc={size_usdc} for {token_id}",
                context={"token_id": token_id, "size_usdc": size_usdc},
            )

        order_id = _new_order_id()
        shares = round(size_usdc / price, 6) if price > 0 else 0.0

        lc = OrderLifecycle(
            order_id=order_id,
            token_id=token_id,
            side=side,
            requested_price=price,
            requested_shares=shares,
            requested_usdc=size_usdc,
        )
        self._registry.register(lc)

        t0 = time.monotonic()

        # ── dry-run path ──────────────────────────────────────────────────────
        if self._dry_run:
            lc.mark_submitted()
            fill: FillRecord = {
                "order_id":      order_id,
                "fill_price":    price,
                "fill_shares":   shares,
                "fill_notional": size_usdc,
                "fill_ts":       _now_iso(),
            }
            lc.mark_filled(fill)
            logger.info("ExecutionService [DRY-RUN] filled %s %.4f @ %.4f", token_id, shares, price)
            return _make_result(lc, latency_ms=(time.monotonic() - t0) * 1000)

        # ── live path ─────────────────────────────────────────────────────────
        for attempt in range(1, self._max_retries + 2):
            try:
                logger.info(
                    "ExecutionService: submitting %s %s %.4f @ %.4f (attempt %d/%d)",
                    side.value, token_id, shares, price, attempt, self._max_retries + 1,
                )
                resp = self._client.create_and_post_order(
                    token_id=token_id,
                    price=price,
                    size=shares,
                    side=side.value,
                    options=options or {},
                    order_type=order_type,
                )
            except Exception as exc:
                logger.error("ExecutionService: submission error on attempt %d: %s", attempt, exc)
                if attempt > self._max_retries:
                    lc.mark_error(str(exc))
                    raise ExecutionFaultError(
                        f"Order submission failed after {attempt} attempts: {exc}",
                        order_id=order_id,
                        context={"token_id": token_id, "attempt": attempt, "error": str(exc)},
                    ) from exc
                time.sleep(self._poll_interval)
                continue

            # Validate exchange response
            exchange_id = _extract_order_id(resp)
            if not exchange_id:
                lc.mark_rejected()
                error_str = str(resp)
                logger.warning("ExecutionService: order rejected by exchange: %s", error_str[:200])
                return _make_result(lc, latency_ms=(time.monotonic() - t0) * 1000, error_message=error_str)

            # Register on exchange and transition to SUBMITTED
            lc.mark_submitted()
            logger.info("ExecutionService: order submitted exchange_id=%s", exchange_id)

            # ── Poll for fill ─────────────────────────────────────────────────
            fill = self._wait_for_fill(lc, exchange_id)
            return _make_result(lc, latency_ms=(time.monotonic() - t0) * 1000)

        # Should not reach here
        lc.mark_error("exhausted retries without result")
        raise ExecutionFaultError(
            "Exhausted retries without result",
            order_id=order_id,
            context={"token_id": token_id},
        )

    def place_market_order(
        self,
        token_id: str,
        side: OrderSide,
        size_usdc: float,
    ) -> ExecutionResult:
        """
        Place a FOK market order (spend `size_usdc` USDC).

        Unlike place_order(), this does not poll — FOK resolves immediately.

        Raises
        ──────
        ExecutionFaultError : on submission error or unexpected exchange response.
        """
        if size_usdc <= 0:
            raise ExecutionFaultError(
                f"Invalid size_usdc={size_usdc} for market order on {token_id}",
                context={"token_id": token_id, "size_usdc": size_usdc},
            )

        order_id = _new_order_id()
        lc = OrderLifecycle(
            order_id=order_id,
            token_id=token_id,
            side=side,
            requested_usdc=size_usdc,
        )
        self._registry.register(lc)

        t0 = time.monotonic()

        if self._dry_run:
            lc.mark_submitted()
            fill: FillRecord = {
                "order_id":      order_id,
                "fill_price":    0.0,
                "fill_shares":   0.0,
                "fill_notional": size_usdc,
                "fill_ts":       _now_iso(),
            }
            lc.mark_filled(fill)
            logger.info("ExecutionService [DRY-RUN] market order filled %s $%.2f", token_id, size_usdc)
            return _make_result(lc, latency_ms=(time.monotonic() - t0) * 1000)

        try:
            resp = self._client.create_and_post_market_order(
                token_id=token_id,
                amount=size_usdc,
                side=side.value,
                order_type="FOK",
            )
        except Exception as exc:
            lc.mark_error(str(exc))
            raise ExecutionFaultError(
                f"Market order submission failed: {exc}",
                order_id=order_id,
                context={"token_id": token_id, "size_usdc": size_usdc, "error": str(exc)},
            ) from exc

        lc.mark_submitted()
        exchange_id = _extract_order_id(resp)

        if not exchange_id:
            lc.mark_rejected()
            return _make_result(
                lc,
                latency_ms=(time.monotonic() - t0) * 1000,
                error_message=str(resp)[:200],
            )

        # FOK: check fill immediately
        fill_info = _parse_fill_from_response(resp, order_id)
        if fill_info:
            lc.mark_filled(fill_info)
        else:
            # FOK that didn't fill is treated as expired
            lc.mark_expired()

        return _make_result(lc, latency_ms=(time.monotonic() - t0) * 1000)

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order by order_id.

        Returns True if the cancel was acknowledged.
        Moves the order to CANCELLED state in the registry.
        Never raises — cancellation failures are logged as warnings.
        """
        try:
            lc = self._registry.lookup(order_id)
        except KeyError:
            logger.warning("ExecutionService.cancel_order: unknown order_id=%r", order_id)
            return False

        if lc.is_terminal:
            logger.debug("ExecutionService.cancel_order: order %r already terminal (%s)", order_id, lc.state.value)
            return False

        try:
            self._client.cancel_order(order_id)
        except Exception as exc:
            logger.warning("ExecutionService.cancel_order: exchange cancel error for %r: %s", order_id, exc)
            # Don't raise — we still try to mark it as cancelled locally

        try:
            self._registry.advance(order_id, OrderState.CANCELLED)
        except Exception as exc:
            logger.warning("ExecutionService.cancel_order: local state transition error: %s", exc)

        return True

    @property
    def registry(self) -> OrderRegistry:
        return self._registry

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _wait_for_fill(self, lc: OrderLifecycle, exchange_id: str) -> Optional[FillRecord]:
        """
        Poll the exchange until the order reaches a terminal state or timeout.

        Returns the FillRecord if filled, or None if cancelled/expired/rejected.
        On unexpected error, transitions to ERROR and raises ExecutionFaultError.
        """
        deadline = time.monotonic() + self._fill_timeout

        while time.monotonic() < deadline:
            try:
                status_resp = self._client.get_order(exchange_id)
            except Exception as exc:
                logger.warning("ExecutionService: poll error for %r: %s", exchange_id, exc)
                time.sleep(self._poll_interval)
                continue

            status = _extract_status(status_resp)
            logger.debug("ExecutionService: poll %r → %s", exchange_id, status)

            if status == FillStatus.COMPLETE:
                fill = _parse_fill_from_response(status_resp, lc.order_id)
                if fill:
                    lc.mark_filled(fill)
                else:
                    # Filled but no fill data parseable — fault
                    lc.mark_error("fill confirmed but no fill data in response")
                    raise ExecutionFaultError(
                        "Exchange reported fill but no fill data could be parsed",
                        order_id=lc.order_id,
                        context={"exchange_id": exchange_id, "response": str(status_resp)[:200]},
                    )
                return fill

            if status == FillStatus.PARTIAL:
                fill = _parse_fill_from_response(status_resp, lc.order_id)
                if fill and not lc.is_filled:
                    lc.mark_partial(fill)
                time.sleep(self._poll_interval)
                continue

            if status == FillStatus.FAILED:
                lc.mark_rejected()
                logger.warning("ExecutionService: order %r rejected by exchange", exchange_id)
                return None

            # PENDING — still waiting
            time.sleep(self._poll_interval)

        # Timeout expired
        logger.warning(
            "ExecutionService: fill timeout after %.1fs for %r (exchange_id=%r)",
            self._fill_timeout, lc.order_id, exchange_id,
        )
        # Attempt cancel on timeout
        try:
            self._client.cancel_order(exchange_id)
            lc.mark_expired()
        except Exception as exc:
            lc.mark_error(f"timeout + cancel failed: {exc}")

        return None


# ── Module-level helpers ──────────────────────────────────────────────────────

def _new_order_id() -> str:
    return f"ord-{uuid.uuid4().hex[:12]}"


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _extract_order_id(resp) -> Optional[str]:
    """Extract the exchange-assigned order_id from a POST response."""
    if not resp:
        return None
    if isinstance(resp, dict):
        return (
            resp.get("orderID")
            or resp.get("order_id")
            or resp.get("id")
            or (resp.get("order", {}) or {}).get("id")
        )
    return None


def _extract_status(resp) -> FillStatus:
    """Map exchange status string to FillStatus enum."""
    if not resp or not isinstance(resp, dict):
        return FillStatus.PENDING
    raw = str(resp.get("status", "") or "").upper()
    if raw in ("MATCHED", "FILLED", "COMPLETE"):
        return FillStatus.COMPLETE
    if raw in ("PARTIALLY_FILLED", "PARTIAL"):
        return FillStatus.PARTIAL
    if raw in ("CANCELLED", "CANCELED", "REJECTED", "FAILED"):
        return FillStatus.FAILED
    return FillStatus.PENDING


def _parse_fill_from_response(resp, order_id: str) -> Optional[FillRecord]:
    """
    Parse fill details from an exchange order response.

    Returns None if essential fill data is missing.
    """
    if not resp or not isinstance(resp, dict):
        return None

    # Try to extract fill fields from various response shapes
    fill_price    = _safe_float(resp.get("average_price") or resp.get("price") or resp.get("avgPrice"))
    fill_notional = _safe_float(resp.get("size_matched") or resp.get("amount") or resp.get("filled_size"))
    fill_shares   = _safe_float(resp.get("original_size") or resp.get("shares") or resp.get("size"))

    if fill_price is None or fill_notional is None:
        return None
    if fill_shares is None and fill_price and fill_price > 0:
        fill_shares = fill_notional / fill_price

    return FillRecord(
        order_id=order_id,
        fill_price=fill_price,
        fill_shares=fill_shares or 0.0,
        fill_notional=fill_notional,
        fill_ts=_now_iso(),
    )


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _make_result(lc: OrderLifecycle, *, latency_ms: float = 0.0, error_message: Optional[str] = None) -> ExecutionResult:
    return ExecutionResult(
        success=lc.state == OrderState.FILLED,
        order_id=lc.order_id,
        final_state=lc.state,
        fill=dict(lc.fill) if lc.fill else None,
        error_code=lc.error_message if lc.state == OrderState.ERROR else None,
        error_message=error_message or lc.error_message,
        attempts=len(lc.history),
        latency_ms=round(latency_ms, 2),
    )


if __name__ == "__main__":
    # Self-test with a mock ExecutionClient
    import sys

    class _MockClient:
        def create_and_post_order(self, token_id, price, size, side, options, order_type):
            return {"orderID": "exch-001", "status": "LIVE"}
        def get_order(self, order_id):
            if order_id == "exch-001":
                return {
                    "status": "MATCHED",
                    "average_price": 0.55,
                    "size_matched": 5.5,
                    "original_size": 10.0,
                }
            return {}
        def cancel_order(self, order_id):
            return {"status": "canceled"}

    svc = ExecutionService(_MockClient(), fill_timeout_sec=5.0, poll_interval_sec=0.1)

    result = svc.place_order(
        token_id="tok-001",
        side=OrderSide.BUY,
        price=0.55,
        size_usdc=10.0,
    )
    assert result["success"], f"Expected success: {result}"
    assert result["fill"]["fill_price"] == 0.55, f"Wrong fill price: {result['fill']}"
    print(f"  Live order: fill={result['fill']}, latency={result['latency_ms']:.1f}ms")

    # Dry-run
    svc2 = ExecutionService(_MockClient(), dry_run=True)
    result2 = svc2.place_order("tok-002", OrderSide.BUY, price=0.50, size_usdc=5.0)
    assert result2["success"]
    print(f"  Dry-run: fill={result2['fill']}")

    print("execution_service self-test PASSED.")
