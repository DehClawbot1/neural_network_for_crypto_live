"""
services/family_pool_state.py
─────────────────────────────
Process-wide snapshot of per-family capital pools, published by the
PortfolioAllocator once per cycle and consumed by sizing/approval sites.

Kept as a tiny module-global instead of a singleton to avoid pulling
RiskService into every hot path that only needs the cap value.

Semantics
---------
- `update(...)` replaces the whole snapshot atomically (no partial state).
- `cap_for(family)` returns the pool ceiling or None when unset (callers
  must fall back to their existing logic when None).
- A pool of 0.0 is a *hard stop* for the family (allocator flagged it out).
"""
from __future__ import annotations

from threading import RLock
from typing import Optional

_LOCK = RLock()
_POOLS: dict[str, float] = {}
_LAST_TOTAL: float = 0.0


def _normalise(family: str) -> str:
    fam = str(family or "").strip().lower()
    if fam.startswith("weather"):
        return "weather_temperature"
    return "btc" if fam in ("", "btc") else fam


def update(pools: dict[str, float] | None, *, total_capital_usdc: float = 0.0) -> None:
    global _LAST_TOTAL
    cleaned: dict[str, float] = {}
    for k, v in (pools or {}).items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        cleaned[_normalise(k)] = max(0.0, f)
    with _LOCK:
        _POOLS.clear()
        _POOLS.update(cleaned)
        _LAST_TOTAL = max(0.0, float(total_capital_usdc or 0.0))


def clear() -> None:
    with _LOCK:
        _POOLS.clear()


def cap_for(family: str) -> Optional[float]:
    with _LOCK:
        if not _POOLS:
            return None
        return _POOLS.get(_normalise(family))


def snapshot() -> dict[str, float]:
    with _LOCK:
        return dict(_POOLS)


def last_total_capital() -> float:
    with _LOCK:
        return _LAST_TOTAL


__all__ = ["update", "clear", "cap_for", "snapshot", "last_total_capital"]
