from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)

LEADERBOARD_URL = "https://data-api.polymarket.com/v1/leaderboard"
_CACHE: dict[tuple[str, str, str, int, str], tuple[float, pd.DataFrame]] = {}


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _env_int(name: str, default: int, minimum: int = 1, maximum: int = 10_000) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except Exception:
        value = int(default)
    return max(int(minimum), min(int(maximum), int(value)))


def _build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def leaderboard_wallet_quality(rank_index: int, total_count: int, pnl_value) -> float:
    total_count = max(int(total_count), 1)
    rank_score = 1.0 - (rank_index / max(total_count - 1, 1))
    pnl = _safe_float(pnl_value, 0.0)
    pnl_score = 0.5 + min(0.5, max(-0.25, pnl / 10_000.0))
    return max(0.05, min(0.99, 0.25 + (rank_score * 0.45) + (pnl_score * 0.30)))


class PolymarketLeaderboardService:
    def __init__(self, *, logs_dir: str = "logs", ttl_seconds: int | None = None, session: requests.Session | None = None):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = max(
            60,
            int(ttl_seconds or _env_int("LEADERBOARD_CACHE_TTL_SECONDS", 600, minimum=60, maximum=86_400)),
        )
        self.session = session or _build_session()

    def _audit_file(self, category: str) -> Path:
        suffix = str(category or "unknown").strip().lower()
        if suffix == "crypto":
            suffix = "btc"
        return self.logs_dir / f"leaderboard_wallets_{suffix}.csv"

    def _cache_key(self, *, category: str, time_period: str, order_by: str, limit: int, logs_dir: str) -> tuple[str, str, str, int, str]:
        return (
            str(category or "").strip().upper(),
            str(time_period or "").strip().upper(),
            str(order_by or "").strip().upper(),
            int(limit),
            str(logs_dir),
        )

    def _write_audit(self, category: str, rows: pd.DataFrame) -> None:
        audit_file = self._audit_file(category)
        try:
            rows.to_csv(audit_file, index=False)
        except Exception as exc:
            logger.warning("Failed to write leaderboard audit %s: %s", audit_file, exc)

    def fetch_leaderboard(
        self,
        *,
        category: str,
        limit: int = 100,
        time_period: str = "WEEK",
        order_by: str = "PNL",
        approved_wallets: set[str] | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        category_key = str(category or "").strip().upper()
        limit = max(1, int(limit))
        cache_key = self._cache_key(
            category=category_key,
            time_period=time_period,
            order_by=order_by,
            limit=limit,
            logs_dir=str(self.logs_dir),
        )
        cached = _CACHE.get(cache_key)
        now = time.time()
        if not force_refresh and cached and (now - cached[0]) <= self.ttl_seconds:
            return cached[1].copy()

        params = {
            "category": category_key,
            "timePeriod": str(time_period or "WEEK").strip().upper(),
            "orderBy": str(order_by or "PNL").strip().upper(),
            "limit": limit,
        }
        approved_wallets = {str(wallet or "").strip().lower() for wallet in (approved_wallets or set()) if str(wallet or "").strip()}
        rows: list[dict] = []
        fetched_at = pd.Timestamp.now(tz="UTC").isoformat()
        try:
            response = self.session.get(LEADERBOARD_URL, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("Leaderboard fetch failed for %s: %s", category_key, exc)
            payload = []

        for idx, user in enumerate(payload if isinstance(payload, list) else []):
            wallet = str(user.get("proxyWallet") or "").strip().lower()
            if not wallet:
                continue
            username = str(user.get("username") or user.get("name") or user.get("profileName") or "").strip()
            pnl_value = _safe_float(user.get("pnl", user.get("PnL", 0.0)), 0.0)
            rows.append(
                {
                    "fetched_at": fetched_at,
                    "category": category_key,
                    "time_period": params["timePeriod"],
                    "order_by": params["orderBy"],
                    "rank": idx + 1,
                    "wallet": wallet,
                    "label": username or wallet,
                    "pnl": pnl_value,
                    "quality_score": leaderboard_wallet_quality(idx, len(payload), pnl_value),
                    "approved": (wallet in approved_wallets) if approved_wallets else True,
                    "enabled": True,
                    "min_wallet_score": None,
                    "region_scope": "",
                    "source": "leaderboard_api",
                }
            )

        frame = pd.DataFrame(rows)
        self._write_audit(category_key, frame)
        _CACHE[cache_key] = (now, frame.copy())
        logger.info(
            "Fetched %s leaderboard wallets for %s (%s/%s).",
            len(frame.index),
            category_key,
            params["timePeriod"],
            params["orderBy"],
        )
        return frame.copy()

    def merge_with_overrides(self, *, category: str, dynamic_rows: pd.DataFrame | None, override_rows: pd.DataFrame | None) -> pd.DataFrame:
        dynamic = pd.DataFrame() if dynamic_rows is None else dynamic_rows.copy()
        overrides = pd.DataFrame() if override_rows is None else override_rows.copy()
        for frame in (dynamic, overrides):
            if "wallet" in frame.columns:
                frame["wallet"] = frame["wallet"].fillna("").astype(str).str.strip().str.lower()
        if overrides.empty and dynamic.empty:
            merged = pd.DataFrame(columns=["wallet", "label", "enabled", "min_wallet_score", "region_scope", "source"])
            self._write_audit(category, merged)
            return merged

        if "source" not in overrides.columns and not overrides.empty:
            overrides["source"] = "manual_override"
        if "approved" not in overrides.columns and not overrides.empty:
            overrides["approved"] = True
        if "enabled" not in overrides.columns and not overrides.empty:
            overrides["enabled"] = True

        merged_rows: list[dict] = []
        if not dynamic.empty:
            merged_rows.extend(dynamic.to_dict("records"))
        if not overrides.empty:
            dynamic_lookup = {
                str(row.get("wallet") or "").strip().lower(): row
                for row in dynamic.to_dict("records")
            }
            for override in overrides.to_dict("records"):
                wallet = str(override.get("wallet") or "").strip().lower()
                if not wallet:
                    continue
                base = dict(dynamic_lookup.get(wallet, {}))
                base.update({k: v for k, v in override.items() if v not in [None, ""]})
                base["wallet"] = wallet
                base["source"] = str(override.get("source") or "manual_override")
                merged_rows.append(base)

        merged = pd.DataFrame(merged_rows)
        if not merged.empty:
            if "enabled" in merged.columns:
                merged["enabled"] = merged["enabled"].map(
                    lambda value: value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}
                    if pd.notna(value)
                    else True
                )
            if "approved" in merged.columns:
                merged["approved"] = merged["approved"].map(
                    lambda value: value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}
                    if pd.notna(value)
                    else True
                )
            merged = merged.drop_duplicates(subset=["wallet"], keep="last").reset_index(drop=True)
        self._write_audit(category, merged)
        return merged

    def snapshot_status(self, *, category: str, time_period: str = "WEEK", order_by: str = "PNL", limit: int = 100) -> dict:
        frame = self.fetch_leaderboard(
            category=category,
            limit=limit,
            time_period=time_period,
            order_by=order_by,
        )
        fetched_at = ""
        if not frame.empty and "fetched_at" in frame.columns:
            fetched_at = str(frame["fetched_at"].iloc[-1] or "")
        return {
            "category": str(category or "").strip().upper(),
            "wallet_count": int(len(frame.index)),
            "fetched_at": fetched_at,
            "audit_file": str(self._audit_file(category)),
            "ttl_seconds": int(self.ttl_seconds),
        }
