"""
orderbook_guard.py
==================
Pre-trade order book analysis gate.

Before every entry, the bot checks:
  1. Does the order book exist for this token?
  2. What is the spread? (wide spread = hidden cost)
  3. What is the midpoint vs best ask? (real fill price)
  4. Is there enough depth on both sides?
  5. Is the order book imbalanced? (warns about thin exits)

From the Polymarket docs:
  - Spread = best_ask - best_bid
  - Midpoint = (best_ask + best_bid) / 2
  - If spread > 0.10, Polymarket shows last traded price instead of midpoint
  - When you BUY, you pay the ask. When you SELL, you hit the bid.
  - Real cost = fee + spread impact + slippage

Usage in supervisor.py:
    from orderbook_guard import OrderBookGuard

    guard = OrderBookGuard()
    check = guard.check_before_entry(token_id, intended_size_usdc=10.0)
    if not check["tradable"]:
        logging.info("Skipping trade: %s", check["reason"])
        continue

    # Use the guard's recommended price instead of naive quote
    fill_price = check["recommended_entry_price"]
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone
from token_utils import normalize_token_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class OrderBookGuard:
    """
    Pre-trade order book analysis.

    Configurable thresholds:
      max_spread:         Max acceptable spread (default 0.10 = 10 cents on $1 token)
      min_bid_depth:      Minimum number of bid levels required (default 2)
      min_ask_depth:      Minimum number of ask levels required (default 2)
      min_depth_usdc:     Minimum total depth in USDC on the side you're trading (default 50)
      max_imbalance:      Max order book imbalance before warning (default 0.7)
      wide_spread_warn:   Spread threshold that triggers a warning but still allows trade (default 0.06)
    """

    def __init__(
        self,
        max_spread=0.20,
        min_bid_depth=1,
        min_ask_depth=1,
        min_depth_usdc=50.0,
        max_imbalance=0.7,
        wide_spread_warn=0.06,
    ):
        self.max_spread = max_spread
        self.min_bid_depth = min_bid_depth
        self.min_ask_depth = min_ask_depth
        self.min_depth_usdc = min_depth_usdc
        self.max_imbalance = max_imbalance
        self.wide_spread_warn = wide_spread_warn
        self._clob_client = None
        # Per-instance 404 cache: token_id -> expiry monotonic timestamp.
        # Any token that returns 404 is suppressed for OB_NO_BOOK_CACHE_TTL_SECONDS
        # (default 30 min) — no HTTP request will be made for it until the TTL expires.
        self._no_book_cache: dict[str, float] = {}
        self._cache_lock = threading.Lock()

    def _get_clob_client(self):
        """Lazy-init a read-only ClobClient for order book queries."""
        if self._clob_client is None:
            try:
                from py_clob_client.client import ClobClient
                self._clob_client = ClobClient("https://clob.polymarket.com")
            except Exception:
                self._clob_client = None
        return self._clob_client

    def _fetch_order_book(self, token_id):
        """Fetch raw order book from CLOB API."""
        token_id = normalize_token_id(token_id)
        if not token_id:
            return None

        # --- 404 cache check ---
        no_book_ttl = max(60, int(os.getenv("OB_NO_BOOK_CACHE_TTL_SECONDS", "1800") or 1800))
        now_mono = time.monotonic()
        with self._cache_lock:
            expiry = self._no_book_cache.get(token_id)
        if expiry is not None and now_mono < expiry:
            logging.debug("OrderBookGuard: skipping cached 404 token %s (%.0fs remaining)", token_id[:16], expiry - now_mono)
            return None

        client = self._get_clob_client()
        if client is None:
            return None

        try:
            book = client.get_order_book(str(token_id))
            # Successful fetch — ensure this token is not in the 404 cache
            with self._cache_lock:
                self._no_book_cache.pop(token_id, None)
            return book
        except Exception as exc:
            # 404 = no orderbook for this token (resolved/inactive market) — expected, not a warning
            _is_404 = "404" in str(exc) or "No orderbook exists" in str(exc)
            if _is_404:
                logging.debug("OrderBookGuard: No orderbook for %s (404 — market likely inactive)", str(token_id))
                with self._cache_lock:
                    self._no_book_cache[token_id] = now_mono + no_book_ttl
            else:
                logging.warning("OrderBookGuard: Failed to fetch book for %s: %s", str(token_id), exc)
            return None

    def _fetch_midpoint(self, token_id):
        """Fetch midpoint from CLOB API."""
        client = self._get_clob_client()
        if client is None:
            return None
        try:
            mid = client.get_midpoint(str(token_id))
            if isinstance(mid, dict):
                return float(mid.get("mid", 0.0))
            return float(mid) if mid else None
        except Exception:
            return None

    def _fetch_spread(self, token_id):
        """Fetch spread from CLOB API."""
        client = self._get_clob_client()
        if client is None:
            return None
        try:
            spread = client.get_spread(str(token_id))
            if isinstance(spread, dict):
                return float(spread.get("spread", 0.0))
            return float(spread) if spread else None
        except Exception:
            return None

    def analyze_book(self, token_id, depth=10):
        """
        Full order book analysis for a token.

        Returns dict with:
          - best_bid, best_ask, midpoint, spread
          - bid_depth, ask_depth (number of levels)
          - bid_volume, ask_volume (total size in top N levels)
          - imbalance (-1 to +1, positive = more bids than asks)
          - top_bids, top_asks (price/size lists)
        """
        token_id = normalize_token_id(token_id)
        result = {
            "token_id": str(token_id or ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "book_available": False,
            "best_bid": None,
            "best_ask": None,
            "midpoint": None,
            "spread": None,
            "spread_pct": None,
            "bid_depth": 0,
            "ask_depth": 0,
            "bid_volume": 0.0,
            "ask_volume": 0.0,
            "imbalance": 0.0,
            "top_bids": [],
            "top_asks": [],
        }

        if not token_id:
            return result

        # --- 404 cache check (fast path before any HTTP) ---
        now_mono = time.monotonic()
        with self._cache_lock:
            expiry = self._no_book_cache.get(str(token_id))
        if expiry is not None and now_mono < expiry:
            logging.debug("OrderBookGuard.analyze_book: cached 404 for %s — skipping HTTP", str(token_id)[:16])
            return result

        book = self._fetch_order_book(token_id)
        if book is None:
            return result

        # Extract bids and asks (handle both dict and object responses)
        bids = getattr(book, "bids", None) or (book.get("bids") if isinstance(book, dict) else None) or []
        asks = getattr(book, "asks", None) or (book.get("asks") if isinstance(book, dict) else None) or []

        # Sort: bids descending by price, asks ascending by price
        def _price(level):
            if isinstance(level, dict):
                return float(level.get("price", 0))
            return float(getattr(level, "price", 0))

        def _size(level):
            if isinstance(level, dict):
                return float(level.get("size", 0))
            return float(getattr(level, "size", 0))

        sorted_bids = sorted(bids, key=_price, reverse=True)
        sorted_asks = sorted(asks, key=_price, reverse=False)

        result["book_available"] = True
        result["bid_depth"] = len(sorted_bids)
        result["ask_depth"] = len(sorted_asks)

        # Top N levels
        top_bids = [{"price": _price(b), "size": _size(b)} for b in sorted_bids[:depth]]
        top_asks = [{"price": _price(a), "size": _size(a)} for a in sorted_asks[:depth]]
        result["top_bids"] = top_bids
        result["top_asks"] = top_asks

        # Best bid/ask
        if sorted_bids:
            result["best_bid"] = _price(sorted_bids[0])
        if sorted_asks:
            result["best_ask"] = _price(sorted_asks[0])

        # Midpoint and spread
        if result["best_bid"] is not None and result["best_ask"] is not None:
            result["midpoint"] = (result["best_bid"] + result["best_ask"]) / 2.0
            result["spread"] = result["best_ask"] - result["best_bid"]
            if result["midpoint"] > 0:
                result["spread_pct"] = result["spread"] / result["midpoint"]
        else:
            # Fallback to API midpoint
            api_mid = self._fetch_midpoint(token_id)
            if api_mid:
                result["midpoint"] = api_mid

        # Volume depth (total size in top levels)
        result["bid_volume"] = sum(b["size"] for b in top_bids)
        result["ask_volume"] = sum(a["size"] for a in top_asks)

        # Order book imbalance: +1 = all bids, -1 = all asks, 0 = balanced
        total_vol = result["bid_volume"] + result["ask_volume"]
        if total_vol > 0:
            result["imbalance"] = (result["bid_volume"] - result["ask_volume"]) / total_vol

        return result

    def dead_token_ids(self) -> set:
        """Return the set of token IDs currently in the 404 cache (no orderbook exists).
        Safe to call at any time; expired entries are pruned on access."""
        now_mono = time.monotonic()
        with self._cache_lock:
            expired = [tid for tid, exp in self._no_book_cache.items() if now_mono >= exp]
            for tid in expired:
                del self._no_book_cache[tid]
            return set(self._no_book_cache.keys())

    def check_before_entry(self, token_id, side="BUY", intended_size_usdc=10.0):
        """
        Main gate: should the bot trade this token right now?

        Returns dict with:
          - tradable: bool
          - reason: str (why rejected, or "ok")
          - warnings: list of str
          - recommended_entry_price: float (best price to use for BUY)
          - recommended_exit_price: float (best price to use for SELL)
          - spread_cost_usdc: estimated spread cost for the intended size
          - analysis: full order book analysis dict
        """
        analysis = self.analyze_book(token_id)

        check = {
            "tradable": False,
            "reason": "unknown",
            "warnings": [],
            "recommended_entry_price": None,
            "recommended_exit_price": None,
            "spread_cost_usdc": 0.0,
            "analysis": analysis,
        }

        # Gate 1: Book must exist
        if not normalize_token_id(token_id):
            check["reason"] = "invalid_token_id"
            return check
        if not analysis["book_available"]:
            check["reason"] = "orderbook_not_available"
            logging.info("OrderBookGuard: BLOCKED %s — no order book found", str(token_id))
            return check

        # Gate 2: Must have bids AND asks
        if analysis["bid_depth"] < self.min_bid_depth:
            check["reason"] = f"insufficient_bid_depth ({analysis['bid_depth']} < {self.min_bid_depth})"
            logging.warning("OrderBookGuard: BLOCKED %s — only %d bid levels", str(token_id), analysis["bid_depth"])
            return check

        if analysis["ask_depth"] < self.min_ask_depth:
            check["reason"] = f"insufficient_ask_depth ({analysis['ask_depth']} < {self.min_ask_depth})"
            logging.warning("OrderBookGuard: BLOCKED %s — only %d ask levels", str(token_id), analysis["ask_depth"])
            return check

        # Gate 3: Spread must not be too wide
        spread = analysis.get("spread")
        if spread is not None and spread > self.max_spread:
            check["reason"] = f"spread_too_wide ({spread:.4f} > {self.max_spread})"
            logging.warning(
                "OrderBookGuard: BLOCKED %s — spread %.4f exceeds max %.4f",
                str(token_id), spread, self.max_spread,
            )
            return check

        # Gate 4: Enough depth on the side we're trading
        if str(side).upper() == "BUY":
            # For BUY, we need asks (we're taking from the ask side)
            ask_depth_usdc = analysis["ask_volume"] * (analysis["best_ask"] or 0)
            if ask_depth_usdc < self.min_depth_usdc and analysis["ask_volume"] > 0:
                check["warnings"].append(
                    f"thin_ask_depth (${ask_depth_usdc:.2f} < ${self.min_depth_usdc:.2f})"
                )
        else:
            # For SELL, we need bids (we're hitting the bid side)
            bid_depth_usdc = analysis["bid_volume"] * (analysis["best_bid"] or 0)
            if bid_depth_usdc < self.min_depth_usdc and analysis["bid_volume"] > 0:
                check["warnings"].append(
                    f"thin_bid_depth (${bid_depth_usdc:.2f} < ${self.min_depth_usdc:.2f})"
                )

        # Warnings (non-blocking)
        if spread is not None and spread > self.wide_spread_warn:
            check["warnings"].append(
                f"wide_spread ({spread:.4f} > {self.wide_spread_warn}) — expect higher execution cost"
            )

        if abs(analysis.get("imbalance", 0.0)) > self.max_imbalance:
            direction = "bid-heavy" if analysis["imbalance"] > 0 else "ask-heavy"
            check["warnings"].append(
                f"orderbook_imbalanced ({direction}, imbalance={analysis['imbalance']:.2f})"
            )

        # Calculate recommended prices
        # For BUY: you pay the ask. Best execution = post a bid slightly below best ask.
        # For SELL: you hit the bid. Best execution = post an ask slightly above best bid.
        best_bid = analysis.get("best_bid")
        best_ask = analysis.get("best_ask")
        midpoint = analysis.get("midpoint")

        if best_ask is not None:
            # Cap at 0.99 — Polymarket CLOB hard maximum. best_ask can be > 0.99 for
            # near-resolved markets; submitting those prices causes 400 errors and
            # increments the session failed-entry counter toward the kill-switch.
            check["recommended_entry_price"] = min(0.99, float(best_ask))
        elif midpoint is not None:
            check["recommended_entry_price"] = min(0.99, float(midpoint))
        if best_bid is not None:
            # Floor at 0.01 — Polymarket CLOB hard minimum.
            check["recommended_exit_price"] = max(0.01, float(best_bid))
        elif midpoint is not None:
            check["recommended_exit_price"] = max(0.01, float(midpoint))

        # Estimate spread cost for the intended trade
        if spread is not None and intended_size_usdc > 0 and best_ask and best_ask > 0:
            shares = intended_size_usdc / best_ask
            # Half the spread is the "cost" vs midpoint for a market-crossing order
            check["spread_cost_usdc"] = shares * (spread / 2.0)

        # All gates passed
        check["tradable"] = True
        check["reason"] = "ok"

        # Log summary
        logging.info(
            "OrderBookGuard: OK %s | bid=%.4f ask=%.4f mid=%.4f spread=%.4f | "
            "bids=%d asks=%d | imbalance=%.2f | spread_cost=$%.4f%s",
            str(token_id),
            best_bid or 0, best_ask or 0, midpoint or 0, spread or 0,
            analysis["bid_depth"], analysis["ask_depth"],
            analysis.get("imbalance", 0),
            check["spread_cost_usdc"],
            f" | WARNINGS: {check['warnings']}" if check["warnings"] else "",
        )

        return check

    def get_smart_entry_price(self, token_id, side="BUY", aggression="midpoint"):
        """
        Get a smart entry price based on order book state.

        aggression levels:
          "passive"  — post at best bid (maker, may not fill)
          "midpoint" — post at midpoint (balanced)
          "aggressive" — cross to best ask (taker, fills immediately)
        """
        analysis = self.analyze_book(token_id, depth=5)

        best_bid = analysis.get("best_bid")
        best_ask = analysis.get("best_ask")
        midpoint = analysis.get("midpoint")

        if str(side).upper() == "BUY":
            if aggression == "passive" and best_bid is not None:
                return best_bid
            elif aggression == "midpoint" and midpoint is not None:
                return midpoint
            elif aggression == "aggressive" and best_ask is not None:
                return best_ask
            # Fallback
            return best_ask or midpoint or best_bid
        else:  # SELL
            if aggression == "passive" and best_ask is not None:
                return best_ask
            elif aggression == "midpoint" and midpoint is not None:
                return midpoint
            elif aggression == "aggressive" and best_bid is not None:
                return best_bid
            return best_bid or midpoint or best_ask

    def format_book_summary(self, analysis):
        """Human-readable order book summary for logging."""
        if not analysis.get("book_available"):
            return "NO BOOK"

        lines = []
        lines.append(f"  Best Bid: {analysis.get('best_bid', 'N/A')}")
        lines.append(f"  Best Ask: {analysis.get('best_ask', 'N/A')}")
        lines.append(f"  Midpoint: {analysis.get('midpoint', 'N/A')}")
        lines.append(f"  Spread:   {analysis.get('spread', 'N/A')}")
        lines.append(f"  Bid Depth: {analysis.get('bid_depth', 0)} levels, {analysis.get('bid_volume', 0):.1f} shares")
        lines.append(f"  Ask Depth: {analysis.get('ask_depth', 0)} levels, {analysis.get('ask_volume', 0):.1f} shares")
        lines.append(f"  Imbalance: {analysis.get('imbalance', 0):.2f}")

        if analysis.get("top_bids"):
            lines.append("  Top Bids:")
            for b in analysis["top_bids"][:3]:
                lines.append(f"    {b['price']:.4f} x {b['size']:.1f}")
        if analysis.get("top_asks"):
            lines.append("  Top Asks:")
            for a in analysis["top_asks"][:3]:
                lines.append(f"    {a['price']:.4f} x {a['size']:.1f}")

        return "\n".join(lines)
