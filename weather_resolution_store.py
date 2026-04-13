"""
weather_resolution_store.py

Fetches and caches real Polymarket contract resolution outcomes for weather markets.

Strategy (in priority order):
  1. Local cache  logs/weather_resolution_cache.csv
  2. closed_positions.csv  (token_won field already captured there)
  3. markets.csv           (if resolved + resolution_outcome columns are present)
  4. Polymarket REST API   GET /markets/{condition_id}

Returns None (never 0.5) when resolution is genuinely unavailable.
A None return triggers weather_resolution_unavailable = True in contract_target_builder.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_COLS = ["condition_id", "resolved_yes", "resolution_source", "fetched_at"]
_POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"


class WeatherResolutionStore:
    """
    Cache-backed store for weather contract resolution outcomes.

    usage:
        store = WeatherResolutionStore(logs_dir="logs")
        result = store.get(condition_id="0xabc...")
        # result: 1, 0, or None (unresolved / unavailable)
    """

    def __init__(self, logs_dir: str = "logs") -> None:
        self.logs_dir = Path(logs_dir)
        self.cache_file = self.logs_dir / "weather_resolution_cache.csv"
        self._cache: dict[str, Optional[int]] = {}  # condition_id → 1/0/None
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, condition_id: str) -> Optional[int]:
        """
        Return 1 (YES), 0 (NO), or None (unresolved / unavailable).
        Never returns 0.5 or any non-integer proxy.
        """
        if not self._loaded:
            self._load()

        cid = str(condition_id or "").strip()
        if not cid:
            return None

        # 1 — in-memory cache hit
        if cid in self._cache:
            return self._cache[cid]

        # 2 — try local CSVs first (free, always available)
        result = self._lookup_local(cid)
        if result is not None:
            self._write_cache_entry(cid, result, "local_csv")
            return result

        # 3 — try Polymarket API
        result = self._fetch_api(cid)
        if result is not None:
            self._write_cache_entry(cid, result, "polymarket_api")
        else:
            # Store None in memory so we don't re-fetch on every call
            self._cache[cid] = None

        return result

    def get_batch(self, condition_ids: list[str]) -> dict[str, Optional[int]]:
        """Resolve a list of condition_ids. Returns dict of cid → result."""
        return {cid: self.get(cid) for cid in condition_ids}

    def backfill_from_closed_positions(self) -> int:
        """
        Read closed_positions.csv and populate cache from token_won field.
        Returns number of new entries written.
        """
        cp_path = self.logs_dir / "closed_positions.csv"
        if not cp_path.exists():
            return 0
        try:
            df = pd.read_csv(cp_path, engine="python", on_bad_lines="skip")
        except Exception as exc:
            logger.warning("WeatherResolutionStore.backfill: cannot read closed_positions: %s", exc)
            return 0

        if df.empty or "condition_id" not in df.columns:
            return 0

        # Filter to weather markets only
        family_col = df.get("market_family", pd.Series("", index=df.index)).astype(str).str.lower()
        weather_mask = family_col.str.startswith("weather")
        if not weather_mask.any():
            return 0

        df_weather = df[weather_mask].copy()
        count = 0
        for _, row in df_weather.iterrows():
            cid = str(row.get("condition_id") or "").strip()
            if not cid:
                continue
            if cid in self._cache and self._cache[cid] is not None:
                continue
            token_won = row.get("token_won")
            if str(token_won).strip().lower() in {"true", "1", "yes", "yes_wins"}:
                resolved_yes = 1
            elif str(token_won).strip().lower() in {"false", "0", "no", "no_wins"}:
                resolved_yes = 0
            else:
                continue
            self._write_cache_entry(cid, resolved_yes, "closed_positions")
            count += 1

        logger.info("WeatherResolutionStore: backfilled %d entries from closed_positions", count)
        return count

    # ------------------------------------------------------------------
    # Private — load / cache
    # ------------------------------------------------------------------

    def _load(self) -> None:
        self._loaded = True
        if not self.cache_file.exists():
            return
        try:
            df = pd.read_csv(self.cache_file, engine="python", on_bad_lines="skip")
        except Exception as exc:
            logger.warning("WeatherResolutionStore: cannot read cache: %s", exc)
            return

        for _, row in df.iterrows():
            cid = str(row.get("condition_id") or "").strip()
            raw = row.get("resolved_yes")
            if cid:
                try:
                    val = int(float(raw)) if pd.notna(raw) else None
                    self._cache[cid] = val if val in (0, 1) else None
                except Exception:
                    self._cache[cid] = None

    def _write_cache_entry(
        self, condition_id: str, resolved_yes: Optional[int], source: str
    ) -> None:
        self._cache[condition_id] = resolved_yes
        entry = {
            "condition_id": condition_id,
            "resolved_yes": resolved_yes,
            "resolution_source": source,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            if self.cache_file.exists():
                existing = pd.read_csv(self.cache_file, engine="python", on_bad_lines="skip")
                # Upsert — remove old entry for this cid if present
                existing = existing[existing["condition_id"].astype(str) != condition_id]
                merged = pd.concat([existing, pd.DataFrame([entry])], ignore_index=True)
            else:
                merged = pd.DataFrame([entry])
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            merged.to_csv(self.cache_file, index=False)
        except Exception as exc:
            logger.warning("WeatherResolutionStore: cache write failed: %s", exc)

    # ------------------------------------------------------------------
    # Private — local CSV lookup
    # ------------------------------------------------------------------

    def _lookup_local(self, condition_id: str) -> Optional[int]:
        """Check markets.csv and closed_positions.csv for resolution data."""
        # Try markets.csv
        for markets_path in [
            self.logs_dir / "markets.csv",
            self.logs_dir.parent / "markets.csv",
        ]:
            result = self._check_markets_csv(markets_path, condition_id)
            if result is not None:
                return result

        # Try closed_positions.csv
        for cp_path in [
            self.logs_dir / "closed_positions.csv",
            self.logs_dir.parent / "closed_positions.csv",
        ]:
            result = self._check_closed_positions(cp_path, condition_id)
            if result is not None:
                return result

        return None

    def _check_markets_csv(self, path: Path, condition_id: str) -> Optional[int]:
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return None
        if "condition_id" not in df.columns:
            return None
        row = df[df["condition_id"].astype(str) == condition_id]
        if row.empty:
            return None
        row = row.iloc[-1]
        # Check for resolved + outcome fields
        is_resolved = str(row.get("active", "true")).lower() in {"false", "0"} or \
                      str(row.get("closed", "false")).lower() in {"true", "1"}
        if not is_resolved:
            return None  # not yet resolved
        outcome = str(row.get("resolution_outcome", "") or "").strip().upper()
        if outcome in {"YES", "1"}:
            return 1
        if outcome in {"NO", "0"}:
            return 0
        # Some csvs store winner_outcome as YES token or specific text
        winner = str(row.get("winner_outcome", "") or "").strip().upper()
        if winner in {"YES", "1"}:
            return 1
        if winner in {"NO", "0"}:
            return 0
        return None

    def _check_closed_positions(self, path: Path, condition_id: str) -> Optional[int]:
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return None
        if "condition_id" not in df.columns:
            return None
        row = df[df["condition_id"].astype(str) == condition_id]
        if row.empty:
            return None
        row = row.iloc[-1]
        token_won = row.get("token_won")
        if pd.isna(token_won):
            return None
        if str(token_won).strip().lower() in {"true", "1", "yes", "yes_wins"}:
            return 1
        if str(token_won).strip().lower() in {"false", "0", "no", "no_wins"}:
            return 0
        return None

    # ------------------------------------------------------------------
    # Private — Polymarket REST API
    # ------------------------------------------------------------------

    def _fetch_api(self, condition_id: str) -> Optional[int]:
        """
        GET /markets/{condition_id} from Polymarket Gamma API.
        Returns 1 / 0 / None.  Non-blocking — returns None on any error.
        """
        if os.getenv("WEATHER_RESOLUTION_API_DISABLED", "").lower() in {"1", "true", "yes"}:
            return None
        try:
            import requests  # lazy import
            url = f"{_POLYMARKET_GAMMA_BASE}/markets/{condition_id}"
            resp = requests.get(url, timeout=8)
            if resp.status_code != 200:
                return None
            data = resp.json()
            if not isinstance(data, dict):
                return None
            # Check active + resolution fields
            active = data.get("active", True)
            if active:
                return None  # still active = not resolved
            outcome = str(data.get("resolution_outcome", "") or "").strip().upper()
            if outcome in {"YES", "1"}:
                return 1
            if outcome in {"NO", "0"}:
                return 0
            # Fallback: outcomes array
            outcomes = data.get("outcomes", []) or []
            if len(outcomes) >= 2:
                prices = data.get("outcomePrices", []) or []
                if len(prices) >= 2:
                    try:
                        p_yes = float(prices[0])
                        p_no = float(prices[1])
                        if p_yes >= 0.99:
                            return 1
                        if p_no >= 0.99:
                            return 0
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug("WeatherResolutionStore API fetch failed for %s: %s", condition_id, exc)
        return None
