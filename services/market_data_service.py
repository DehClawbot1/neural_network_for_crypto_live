"""
services/market_data_service.py
─────────────────────────────────
Fetch and normalize raw Polymarket market data.

Rules
─────
1. This service fetches and normalizes data only. No trading decisions.
2. DataFaultError is raised if a market's critical fields are missing or
   uncoercible (token_id, condition_id, current_price). Callers skip that market.
3. The snapshot_ts on every NormalizedMarket is the UTC ISO timestamp when
   this service fetched the data — not a field from the API response.
4. This service never writes to disk. All output is in-memory.

Usage
─────
    svc = MarketDataService(capabilities_client)
    markets = svc.fetch_open_markets(family_prefix="btc")
    # returns list[NormalizedMarket]
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from services.types import DataFaultError, NormalizedMarket

logger = logging.getLogger(__name__)

# Polymarket resolution: prices are probabilities in [0, 1]
_MIN_PRICE = 0.001
_MAX_PRICE = 0.999

# Markets resolving within this many seconds are considered "near-closed"
_MIN_TIME_LEFT_SEC = 300.0    # 5 minutes


class MarketDataService:
    """
    Fetches open markets from Polymarket and normalizes them to NormalizedMarket.

    The injected client should have:
      - get_markets(collect_all, ...) → list of raw market dicts
      - get_market(condition_id) → single raw market dict (optional)

    The service itself has no knowledge of how to call the exchange.
    """

    def __init__(self, polymarket_client) -> None:
        self._client = polymarket_client

    # ── Primary API ───────────────────────────────────────────────────────────

    def fetch_open_markets(
        self,
        *,
        family_prefix: Optional[str] = None,
        max_pages: Optional[int] = None,
    ) -> list[NormalizedMarket]:
        """
        Fetch all open (non-closed) markets and normalize to NormalizedMarket list.

        Parameters
        ──────────
        family_prefix : if given, only return markets whose market_family starts
                        with this prefix (e.g. "btc", "weather_temperature")
        max_pages     : limit how many pages of results to fetch (None = all)

        Returns
        ───────
        List of NormalizedMarket TypedDicts for open markets.
        Skips individual markets that fail normalization (logged as warnings).
        Never raises on per-market failures.

        Raises
        ──────
        DataFaultError : if the API call itself fails (not per-market failures).
        """
        try:
            raw_markets = self._client.get_markets(
                collect_all=True,
                max_pages=max_pages,
            )
        except Exception as exc:
            raise DataFaultError(
                f"Failed to fetch markets from Polymarket: {exc}",
                field="markets",
                context={"error": str(exc)},
            ) from exc

        if not isinstance(raw_markets, list):
            # Some API wrappers return {"data": [...], "next_cursor": ...}
            if isinstance(raw_markets, dict) and "data" in raw_markets:
                raw_markets = raw_markets["data"]
            else:
                raise DataFaultError(
                    f"Unexpected market list format: {type(raw_markets).__name__}",
                    field="markets",
                )

        snapshot_ts = _now_iso()
        result: list[NormalizedMarket] = []

        for raw in raw_markets:
            if not isinstance(raw, dict):
                continue
            try:
                normalized = self._normalize(raw, snapshot_ts)
            except DataFaultError as exc:
                logger.debug("MarketDataService: skipping market (fault=%s)", exc)
                continue
            except Exception as exc:
                logger.warning("MarketDataService: unexpected error normalizing market: %s", exc)
                continue

            if not normalized["is_open"]:
                continue
            if normalized["time_left_sec"] < _MIN_TIME_LEFT_SEC:
                continue
            if family_prefix and not normalized["market_family"].startswith(family_prefix):
                continue

            result.append(normalized)

        logger.info(
            "MarketDataService: fetched %d open markets%s (from %d raw)",
            len(result),
            f" [family={family_prefix}]" if family_prefix else "",
            len(raw_markets),
        )
        return result

    def fetch_market(self, condition_id: str) -> NormalizedMarket:
        """
        Fetch and normalize a single market by condition_id.

        Raises
        ──────
        DataFaultError : if the market cannot be fetched or normalized.
        """
        try:
            raw = self._client.get_market(condition_id)
        except Exception as exc:
            raise DataFaultError(
                f"Failed to fetch market {condition_id!r}: {exc}",
                field="condition_id",
                context={"condition_id": condition_id, "error": str(exc)},
            ) from exc

        if not raw or not isinstance(raw, dict):
            raise DataFaultError(
                f"Empty or non-dict response for market {condition_id!r}",
                field="condition_id",
                context={"condition_id": condition_id},
            )

        return self._normalize(raw, _now_iso())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _normalize(self, raw: dict, snapshot_ts: str) -> NormalizedMarket:
        """
        Normalize a raw Polymarket market dict to NormalizedMarket.

        Raises DataFaultError on missing/invalid required fields.
        """
        # ── Required fields ───────────────────────────────────────────────────
        condition_id = str(raw.get("condition_id") or raw.get("conditionId") or "").strip()
        if not condition_id:
            raise DataFaultError("market has no condition_id", field="condition_id")

        # token_id — extract YES token from tokens list, or fall back to top-level field
        token_id = _extract_yes_token_id(raw)
        if not token_id:
            raise DataFaultError(
                f"market {condition_id!r} has no usable token_id",
                field="token_id",
                context={"condition_id": condition_id},
            )

        # ── Optional / derived fields ─────────────────────────────────────────
        market_slug   = str(raw.get("market_slug") or raw.get("slug") or raw.get("question_slug") or condition_id)
        question      = str(raw.get("question") or raw.get("title") or "")
        market_family = _classify_family(question, raw.get("market_family") or raw.get("category") or "")

        current_price = _extract_current_price(raw, token_id)
        if current_price is None:
            raise DataFaultError(
                f"market {condition_id!r} has no usable current_price",
                field="current_price",
                context={"condition_id": condition_id},
            )

        liquidity  = _safe_float(raw.get("liquidity") or raw.get("liquidityNum") or 0.0)
        volume_24h = _safe_float(raw.get("volume24hr") or raw.get("volume24h") or raw.get("volume") or 0.0)
        time_left  = _compute_time_left(raw.get("end_date_iso") or raw.get("endDateIso") or raw.get("endDate"))

        # is_open: market is active and not closed
        is_open = bool(raw.get("active", True)) and not bool(raw.get("closed", False))

        return NormalizedMarket(
            token_id=token_id,
            condition_id=condition_id,
            market_slug=market_slug,
            market_family=market_family,
            current_price=round(float(current_price), 6),
            liquidity=round(liquidity, 4),
            volume_24h=round(volume_24h, 4),
            time_left_sec=round(time_left, 1),
            snapshot_ts=snapshot_ts,
            is_open=is_open,
        )


# ── Module-level helpers ──────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _extract_yes_token_id(raw: dict) -> Optional[str]:
    """
    Extract the YES-outcome token_id from a raw market dict.

    Polymarket markets have a 'tokens' list: [{"token_id": "...", "outcome": "Yes"}, ...]
    Falls back to top-level 'token_id' field if tokens list is absent.
    """
    tokens = raw.get("tokens") or []
    if tokens:
        for tok in tokens:
            if isinstance(tok, dict):
                outcome = str(tok.get("outcome") or "").lower()
                if outcome in ("yes", "true", "1"):
                    tid = str(tok.get("token_id") or "").strip()
                    if tid:
                        return tid
        # No YES token found — return first token_id
        first = tokens[0]
        if isinstance(first, dict):
            tid = str(first.get("token_id") or "").strip()
            if tid:
                return tid

    # Top-level fallback
    tid = str(raw.get("token_id") or raw.get("tokenId") or "").strip()
    return tid or None


def _extract_current_price(raw: dict, token_id: str) -> Optional[float]:
    """
    Extract the current mid-price for the YES token.

    Checks: tokens[].price for the YES token, then last_trade_price, then price.
    """
    tokens = raw.get("tokens") or []
    for tok in tokens:
        if isinstance(tok, dict):
            if str(tok.get("token_id") or "") == token_id:
                price = _safe_float(tok.get("price"), default=-1.0)
                if _MIN_PRICE <= price <= _MAX_PRICE:
                    return price

    # Fallback to top-level price fields
    for field in ("last_trade_price", "lastTradePrice", "price", "midpoint"):
        val = raw.get(field)
        if val is not None:
            price = _safe_float(val, default=-1.0)
            if _MIN_PRICE <= price <= _MAX_PRICE:
                return price

    return None


def _compute_time_left(end_date_str: Optional[str]) -> float:
    """Return seconds until end_date, or a large sentinel if not parseable."""
    if not end_date_str:
        return 86400.0 * 30  # unknown → assume 30 days

    try:
        # Handle both "2026-01-01T00:00:00Z" and "2026-01-01T00:00:00+00:00"
        s = str(end_date_str).replace("Z", "+00:00")
        end_dt = datetime.fromisoformat(s)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (end_dt - now).total_seconds())
    except Exception:
        return 0.0


def _classify_family(question: str, raw_family: str) -> str:
    """
    Derive market_family from question text and/or raw category.

    This mirrors the classification logic already in the codebase.
    """
    if raw_family:
        normalized_family = str(raw_family).lower().replace(" ", "_").replace("-", "_")
        if normalized_family:
            return normalized_family

    q = question.lower()
    if any(kw in q for kw in ["bitcoin", "btc", "crypto", "ethereum", "eth"]):
        return "btc"
    if any(kw in q for kw in [
        "temperature", "high temp", "highest temp", "degrees", "fahrenheit", "celsius",
        "weather", "heat", "cold front",
    ]):
        return "weather_temperature_threshold"
    if any(kw in q for kw in ["election", "president", "senator", "vote"]):
        return "politics_election"
    if any(kw in q for kw in ["stock", "nasdaq", "s&p", "dow", "market cap"]):
        return "equity"

    return "other"


if __name__ == "__main__":
    # Self-test using a mock client
    from services.types import DataFaultError

    class _MockClient:
        def get_markets(self, collect_all=True, max_pages=None):
            return [
                {
                    "condition_id": "cond-001",
                    "question": "Will Bitcoin close above $100k?",
                    "tokens": [
                        {"token_id": "tok-yes-001", "outcome": "Yes", "price": 0.65},
                        {"token_id": "tok-no-001",  "outcome": "No",  "price": 0.35},
                    ],
                    "active": True,
                    "closed": False,
                    "endDateIso": "2099-01-01T00:00:00Z",
                    "liquidity": 1500.0,
                    "volume24hr": 250.0,
                },
                {
                    "condition_id": "cond-002",
                    "question": "Will NYC highest temperature exceed 90°F on July 4?",
                    "tokens": [
                        {"token_id": "tok-yes-002", "outcome": "Yes", "price": 0.40},
                    ],
                    "active": True,
                    "closed": False,
                    "endDateIso": "2099-07-05T00:00:00Z",
                    "liquidity": 800.0,
                    "volume24hr": 100.0,
                },
                {
                    "condition_id": "cond-003-closed",
                    "question": "Some closed market",
                    "tokens": [{"token_id": "tok-003", "outcome": "Yes", "price": 0.99}],
                    "active": False,
                    "closed": True,
                    "endDateIso": "2020-01-01T00:00:00Z",
                    "liquidity": 0.0,
                    "volume24hr": 0.0,
                },
            ]
        def get_market(self, condition_id):
            return self.get_markets()[0]

    svc = MarketDataService(_MockClient())

    markets = svc.fetch_open_markets()
    assert len(markets) == 2, f"Expected 2 open markets, got {len(markets)}"
    btc_m = next(m for m in markets if m["market_family"] == "btc")
    assert btc_m["token_id"] == "tok-yes-001"
    assert btc_m["current_price"] == 0.65
    weather_m = next(m for m in markets if "weather" in m["market_family"])
    assert weather_m["token_id"] == "tok-yes-002"
    assert weather_m["current_price"] == 0.40

    # Family filter
    btc_only = svc.fetch_open_markets(family_prefix="btc")
    assert len(btc_only) == 1

    # fetch_market
    single = svc.fetch_market("cond-001")
    assert single["condition_id"] == "cond-001"

    print("market_data_service self-test PASSED.")
