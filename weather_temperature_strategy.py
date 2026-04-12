from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from brain_log_routing import overwrite_csv_with_brain_mirrors
from brain_paths import WEATHER_FAMILY
from leaderboard_service import PolymarketLeaderboardService
from polymarket_profile_client import PolymarketProfileClient
from strategy_layers import PredictionLayer
from weather_temperature_forecast import WeatherForecastService
from weather_temperature_guard import weather_city_date_cluster_key
from weather_temperature_markets import (
    enrich_weather_temperature_frame,
    fetch_weather_temperature_markets,
    parse_weather_temperature_market_text,
)


logger = logging.getLogger(__name__)


def _safe_float(value, default=0.0):
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _clip01(value) -> float:
    return float(np.clip(_safe_float(value, 0.0), 0.0, 1.0))


def _normalize_wallet(value) -> str:
    return str(value or "").strip().lower()


def _normalize_side(value) -> str:
    side = str(value or "").strip().upper()
    if side in {"YES", "UP", "LONG", "BULLISH"}:
        return "YES"
    if side in {"NO", "DOWN", "SHORT", "BEARISH"}:
        return "NO"
    return side


def _normalize_market_key(source: dict | None) -> str:
    source = source or {}
    for key in ("condition_id", "market_slug", "market_title", "market"):
        value = str(source.get(key) or "").strip()
        if value:
            return value
    return ""


class WeatherTemperatureStrategy:
    def __init__(
        self,
        *,
        logs_dir: str = "logs",
        watchlist_path: str | None = None,
        profile_client: PolymarketProfileClient | None = None,
        forecast_service: WeatherForecastService | None = None,
        leaderboard_service: PolymarketLeaderboardService | None = None,
    ):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.watchlist_path = Path(
            watchlist_path or os.getenv("WEATHER_APPROVED_WATCHLIST_PATH", "config/weather_wallet_watchlist.csv")
        )
        self.profile_client = profile_client or PolymarketProfileClient(timeout=20)
        self.forecast_service = forecast_service or WeatherForecastService(timeout=20)
        self.leaderboard_service = leaderboard_service or PolymarketLeaderboardService(logs_dir=str(self.logs_dir))
        self.snapshot_file = self.logs_dir / "weather_wallet_state_snapshot.csv"
        self.position_epsilon = max(0.001, float(os.getenv("WEATHER_POSITION_EPS", "0.5") or 0.5))
        self.sharp_reduce_threshold = max(0.05, float(os.getenv("SOURCE_WALLET_SHARP_REDUCE_THRESHOLD", "0.55") or 0.55))
        self.fresh_minutes = max(5, int(os.getenv("SOURCE_WALLET_SIGNAL_FRESH_MINUTES", "90") or 90))
        self.min_wallet_score = max(0.0, min(1.0, float(os.getenv("WEATHER_MIN_WALLET_SCORE", "0.60") or 0.60)))
        self.min_forecast_edge = float(os.getenv("WEATHER_MIN_FORECAST_EDGE", "0.08") or 0.08)
        self.max_spread = max(0.01, float(os.getenv("WEATHER_MAX_SPREAD", "0.12") or 0.12))
        self.min_liquidity_score = max(0.0, min(1.0, float(os.getenv("WEATHER_MIN_LIQUIDITY_SCORE", "0.08") or 0.08)))
        self.max_concurrent_positions = max(1, int(os.getenv("WEATHER_MAX_CONCURRENT_POSITIONS", "6") or 6))
        self.cluster_cap = max(1, int(os.getenv("WEATHER_CITY_DATE_CLUSTER_CAP", "1") or 1))
        self._last_watchlist_log_signature: tuple[int, int, int] | None = None

    def _watchlist_rows_from_env(self) -> pd.DataFrame:
        raw = str(
            os.getenv("WEATHER_APPROVED_WALLETS")
            or os.getenv("WEATHER_APPROVED_WALLET")
            or ""
        ).strip()
        if not raw:
            return pd.DataFrame(columns=["wallet", "label", "enabled", "min_wallet_score", "region_scope"])

        rows = []
        separators_normalized = raw.replace("\r", "\n").replace(";", "\n")
        for line in separators_normalized.split("\n"):
            entry = str(line or "").strip()
            if not entry:
                continue
            parts = [part.strip() for part in entry.split("|")]
            wallet = _normalize_wallet(parts[0] if parts else "")
            if not wallet:
                continue
            label = parts[1] if len(parts) > 1 else ""
            enabled_raw = parts[2] if len(parts) > 2 else "true"
            min_score_raw = parts[3] if len(parts) > 3 else self.min_wallet_score
            region_scope = parts[4] if len(parts) > 4 else ""
            enabled = str(enabled_raw).strip().lower() in {"1", "true", "yes", "on", ""}
            rows.append(
                {
                    "wallet": wallet,
                    "label": label,
                    "enabled": enabled,
                    "min_wallet_score": _safe_float(min_score_raw, self.min_wallet_score),
                    "region_scope": region_scope,
                }
            )

        if not rows:
            return pd.DataFrame(columns=["wallet", "label", "enabled", "min_wallet_score", "region_scope"])
        return pd.DataFrame(rows)

    def load_watchlist(self) -> pd.DataFrame:
        columns = ["wallet", "label", "enabled", "min_wallet_score", "region_scope"]
        if not self.watchlist_path.exists():
            watchlist = pd.DataFrame(columns=columns)
        else:
            try:
                watchlist = pd.read_csv(self.watchlist_path, engine="python", on_bad_lines="skip")
            except Exception as exc:
                logger.warning("Failed to read weather wallet watchlist %s: %s", self.watchlist_path, exc)
                watchlist = pd.DataFrame(columns=columns)

        env_watchlist = self._watchlist_rows_from_env()
        if not env_watchlist.empty:
            env_watchlist = env_watchlist.copy()
            env_watchlist["source"] = "manual_override_env"
        if not watchlist.empty:
            watchlist = watchlist.copy()
            watchlist["source"] = "manual_override_csv"
        override_rows = pd.concat(
            [frame for frame in (watchlist, env_watchlist) if frame is not None and not frame.empty],
            ignore_index=True,
            sort=False,
        ) if (not watchlist.empty or not env_watchlist.empty) else pd.DataFrame(columns=columns + ["source"])

        dynamic_enabled = str(os.getenv("WEATHER_USE_DYNAMIC_LEADERBOARD", "true")).strip().lower() in {"1", "true", "yes", "on"}
        dynamic_rows = pd.DataFrame()
        if dynamic_enabled:
            dynamic_limit = max(10, int(os.getenv("WEATHER_LEADERBOARD_LIMIT", "100") or 100))
            dynamic_rows = self.leaderboard_service.fetch_leaderboard(
                category="WEATHER",
                limit=dynamic_limit,
                time_period="WEEK",
                order_by="PNL",
            )
            if not dynamic_rows.empty:
                dynamic_rows = dynamic_rows.copy()
                dynamic_rows["enabled"] = True
                dynamic_rows["approved"] = True
                min_wallet_score_series = dynamic_rows.get(
                    "min_wallet_score",
                    pd.Series([self.min_wallet_score] * len(dynamic_rows.index), index=dynamic_rows.index),
                )
                dynamic_rows["min_wallet_score"] = pd.to_numeric(
                    min_wallet_score_series,
                    errors="coerce",
                ).fillna(self.min_wallet_score)
                dynamic_rows["region_scope"] = dynamic_rows.get("region_scope", pd.Series("", index=dynamic_rows.index)).fillna("")
            else:
                logger.info("Weather leaderboard returned no wallets; falling back to manual weather overrides only.")

        merged_watchlist = self.leaderboard_service.merge_with_overrides(
            category="WEATHER",
            dynamic_rows=dynamic_rows,
            override_rows=override_rows,
        )
        watchlist = merged_watchlist.copy()

        for field, default in (
            ("wallet", ""),
            ("label", ""),
            ("enabled", True),
            ("min_wallet_score", self.min_wallet_score),
            ("region_scope", ""),
            ("source", "leaderboard_api"),
        ):
            if field not in watchlist.columns:
                watchlist[field] = default
        enabled_mask = (
            watchlist["enabled"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "true", "yes", "on"})
        )
        watchlist = watchlist[enabled_mask].copy()
        watchlist["wallet"] = watchlist["wallet"].map(_normalize_wallet)
        watchlist["min_wallet_score"] = pd.to_numeric(watchlist["min_wallet_score"], errors="coerce").fillna(self.min_wallet_score)
        watchlist = watchlist[watchlist["wallet"].astype(str).str.len() > 0].drop_duplicates(subset=["wallet"], keep="last")
        if dynamic_enabled:
            dynamic_count = int(len(dynamic_rows.index))
            override_count = int(len(override_rows.index))
            signature = (dynamic_count, override_count, int(len(watchlist.index)))
            if signature != self._last_watchlist_log_signature:
                logger.info(
                    "Weather wallet source loaded %s live leaderboard wallets and %s manual overrides (%s effective wallets).",
                    dynamic_count,
                    override_count,
                    int(len(watchlist.index)),
                )
                self._last_watchlist_log_signature = signature
            else:
                logger.debug(
                    "Weather wallet source unchanged: %s live leaderboard wallets, %s manual overrides, %s effective wallets.",
                    dynamic_count,
                    override_count,
                    int(len(watchlist.index)),
                )
        return watchlist.reset_index(drop=True)

    def fetch_markets(self) -> pd.DataFrame:
        return fetch_weather_temperature_markets(limit=500, closed=False, max_offset=None)

    def _safe_profile_call(self, method_name: str, **kwargs):
        method = getattr(self.profile_client, method_name, None)
        if not callable(method):
            return []
        try:
            payload = method(**kwargs)
            return payload if isinstance(payload, list) else []
        except Exception as exc:
            logger.warning("Weather profile fetch %s failed for %s: %s", method_name, kwargs.get("user"), exc)
            return []

    def _build_market_lookup(self, markets_df: pd.DataFrame | None) -> tuple[dict, dict, dict]:
        by_condition = {}
        by_slug = {}
        by_title = {}
        if markets_df is None or markets_df.empty:
            return by_condition, by_slug, by_title
        for _, row in enrich_weather_temperature_frame(markets_df).iterrows():
            row_dict = row.to_dict()
            condition_id = str(row_dict.get("condition_id") or "").strip()
            market_slug = str(row_dict.get("market_slug", row_dict.get("slug")) or "").strip()
            market_title = str(row_dict.get("market_title", row_dict.get("question")) or "").strip()
            if condition_id:
                by_condition[condition_id] = row_dict
            if market_slug:
                by_slug[market_slug] = row_dict
            if market_title:
                by_title[market_title.lower()] = row_dict
        return by_condition, by_slug, by_title

    def _match_market_row(self, raw_row: dict, lookups: tuple[dict, dict, dict]) -> dict:
        by_condition, by_slug, by_title = lookups
        condition_id = str(raw_row.get("conditionId") or raw_row.get("condition_id") or "").strip()
        slug = str(raw_row.get("slug") or raw_row.get("eventSlug") or raw_row.get("market_slug") or "").strip()
        title = str(raw_row.get("title") or raw_row.get("market_title") or raw_row.get("question") or "").strip()
        if condition_id and condition_id in by_condition:
            return dict(by_condition[condition_id])
        if slug and slug in by_slug:
            return dict(by_slug[slug])
        if title and title.lower() in by_title:
            return dict(by_title[title.lower()])
        return {}

    def _temperature_rows(self, raw_rows: list[dict], *, source_name: str, lookups: tuple[dict, dict, dict], wallet: str) -> pd.DataFrame:
        rows = []
        for raw_row in raw_rows or []:
            raw = dict(raw_row or {})
            matched_market = self._match_market_row(raw, lookups)
            title = str(raw.get("title") or raw.get("question") or matched_market.get("market_title") or "").strip()
            parsed = parse_weather_temperature_market_text(title, market_slug=raw.get("slug") or matched_market.get("market_slug"))
            if not title or not str(parsed.get("market_family", "")).startswith("weather_temperature"):
                continue
            row = {
                "source_name": source_name,
                "trader_wallet": wallet,
                "market_title": title,
                "market_slug": raw.get("slug") or raw.get("eventSlug") or matched_market.get("market_slug"),
                "condition_id": raw.get("conditionId") or raw.get("condition_id") or matched_market.get("condition_id"),
                "token_id": raw.get("asset") or raw.get("token_id"),
                "outcome_side": _normalize_side(raw.get("outcome")),
                "price": _safe_float(raw.get("price", raw.get("avgPrice", matched_market.get("current_price", 0.0))), 0.0),
                "current_price": _safe_float(raw.get("curPrice", raw.get("price", matched_market.get("current_price", 0.0))), 0.0),
                "size": _safe_float(raw.get("size", raw.get("totalBought", raw.get("currentValue", 0.0))), 0.0),
                "current_value": _safe_float(raw.get("currentValue"), 0.0),
                "realized_pnl": _safe_float(raw.get("realizedPnl", raw.get("cashPnl", 0.0)), 0.0),
                "percent_pnl": _safe_float(raw.get("percentPnl"), 0.0),
                "timestamp": raw.get("timestamp"),
                "end_date": raw.get("endDate") or matched_market.get("end_date"),
                "liquidity": _safe_float(matched_market.get("liquidity"), 0.0),
                "volume": _safe_float(matched_market.get("volume"), 0.0),
                "best_bid": _safe_float(matched_market.get("best_bid"), 0.0),
                "best_ask": _safe_float(matched_market.get("best_ask"), 0.0),
                "last_trade_price": _safe_float(matched_market.get("last_trade_price", matched_market.get("current_price", raw.get("curPrice", raw.get("price", 0.0)))), 0.0),
                "market_url": matched_market.get("url"),
            }
            row.update(parsed)
            rows.append(row)
        return pd.DataFrame(rows)

    def _compute_wallet_skill(self, wallet: str, watch_row: dict, closed_rows: pd.DataFrame) -> dict:
        region_scope = str(watch_row.get("region_scope") or "").strip().lower()
        if closed_rows is None or closed_rows.empty:
            return {
                "wallet_temp_hit_rate_90d": 0.5,
                "wallet_temp_realized_pnl_90d": 0.0,
                "wallet_region_score": 0.5,
                "wallet_temp_range_skill": 0.5,
                "wallet_temp_threshold_skill": 0.5,
                "wallet_quality_score": _clip01(max(self.min_wallet_score, _safe_float(watch_row.get("min_wallet_score"), self.min_wallet_score))),
            }

        work = closed_rows.copy()
        if "timestamp" in work.columns:
            work["timestamp"] = pd.to_datetime(work.get("timestamp"), utc=True, errors="coerce")
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90)
            work = work[(work["timestamp"].isna()) | (work["timestamp"] >= cutoff)]
        if work.empty:
            return self._compute_wallet_skill(wallet, watch_row, pd.DataFrame())

        realized = pd.to_numeric(work.get("realized_pnl", 0.0), errors="coerce").fillna(0.0)
        hit_rate = float((realized > 0).mean()) if len(realized.index) else 0.5
        pnl_total = float(realized.sum())
        pnl_score = _clip01(0.5 + np.tanh(pnl_total / 500.0) * 0.5)

        family_series = work.get("market_family", pd.Series("", index=work.index)).astype(str).str.lower()
        range_skill = float((realized[family_series == "weather_temperature_range"] > 0).mean()) if (family_series == "weather_temperature_range").any() else hit_rate
        threshold_skill = float((realized[family_series == "weather_temperature_threshold"] > 0).mean()) if (family_series == "weather_temperature_threshold").any() else hit_rate

        region_score = 0.5
        if region_scope and "weather_location" in work.columns:
            region_mask = work["weather_location"].astype(str).str.lower().str.contains(region_scope, regex=False, na=False)
            if region_mask.any():
                region_score = float((realized[region_mask] > 0).mean())
        min_wallet_score = max(self.min_wallet_score, _safe_float(watch_row.get("min_wallet_score"), self.min_wallet_score))
        quality = _clip01(
            (hit_rate * 0.35)
            + (pnl_score * 0.20)
            + (_clip01(region_score) * 0.15)
            + (_clip01(range_skill) * 0.15)
            + (_clip01(threshold_skill) * 0.15)
        )
        quality = max(min_wallet_score, quality)
        return {
            "wallet_temp_hit_rate_90d": round(hit_rate, 4),
            "wallet_temp_realized_pnl_90d": round(pnl_total, 4),
            "wallet_region_score": round(_clip01(region_score), 4),
            "wallet_temp_range_skill": round(_clip01(range_skill), 4),
            "wallet_temp_threshold_skill": round(_clip01(threshold_skill), 4),
            "wallet_quality_score": round(_clip01(quality), 4),
        }

    def _last_event_maps(self, trades_df: pd.DataFrame, closed_df: pd.DataFrame) -> tuple[dict, dict, dict]:
        last_add = {}
        last_reduce = {}
        last_close = {}
        if trades_df is not None and not trades_df.empty:
            work = trades_df.copy()
            work["timestamp"] = pd.to_datetime(work.get("timestamp"), utc=True, errors="coerce")
            work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
            for _, row in work.iterrows():
                row_dict = row.to_dict()
                market_key = _normalize_market_key(row_dict)
                side = _normalize_side(row_dict.get("outcome_side"))
                trade_side = str(row_dict.get("side") or "").strip().upper()
                lookup_key = (_normalize_wallet(row_dict.get("trader_wallet")), market_key, side)
                if trade_side == "BUY":
                    last_add[lookup_key] = row_dict["timestamp"].isoformat()
                elif trade_side == "SELL":
                    last_reduce[lookup_key] = row_dict["timestamp"].isoformat()
        if closed_df is not None and not closed_df.empty:
            work = closed_df.copy()
            if "timestamp" in work.columns:
                work["timestamp"] = pd.to_datetime(work.get("timestamp"), utc=True, errors="coerce")
            elif "closed_at" in work.columns:
                work["timestamp"] = pd.to_datetime(work.get("closed_at"), utc=True, errors="coerce")
            work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
            for _, row in work.iterrows():
                row_dict = row.to_dict()
                market_key = _normalize_market_key(row_dict)
                side = _normalize_side(row_dict.get("outcome_side"))
                lookup_key = (_normalize_wallet(row_dict.get("trader_wallet")), market_key, side)
                last_close[lookup_key] = row_dict["timestamp"].isoformat()
        return last_add, last_reduce, last_close

    def _build_current_snapshot(self, markets_df: pd.DataFrame | None) -> pd.DataFrame:
        watchlist = self.load_watchlist()
        if watchlist.empty:
            return pd.DataFrame()

        lookups = self._build_market_lookup(markets_df)
        snapshot_rows: list[dict] = []
        for _, watch_row in watchlist.iterrows():
            wallet = _normalize_wallet(watch_row.get("wallet"))
            if not wallet:
                continue
            open_positions = self._safe_profile_call("get_positions", user=wallet, limit=200, sort_by="CURRENT")
            closed_positions = self._safe_profile_call("get_closed_positions", user=wallet, limit=200)
            trades = self._safe_profile_call("get_trades", user=wallet, limit=200, taker_only=False)
            _ = self._safe_profile_call("get_activity", user=wallet, limit=200)

            open_df = self._temperature_rows(open_positions, source_name="positions", lookups=lookups, wallet=wallet)
            closed_df = self._temperature_rows(closed_positions, source_name="closed_positions", lookups=lookups, wallet=wallet)
            trades_df = self._temperature_rows(trades, source_name="trades", lookups=lookups, wallet=wallet)
            skill = self._compute_wallet_skill(wallet, watch_row.to_dict(), closed_df)
            last_add, last_reduce, last_close = self._last_event_maps(trades_df, closed_df)
            avg_size = float(open_df["size"].replace(0, pd.NA).dropna().mean()) if not open_df.empty and "size" in open_df.columns else 0.0
            if math.isnan(avg_size):
                avg_size = 0.0

            for _, row in open_df.iterrows():
                base = row.to_dict()
                market_key = _normalize_market_key(base)
                side = _normalize_side(base.get("outcome_side"))
                state_key = (wallet, market_key, side)
                size = max(_safe_float(base.get("size"), 0.0), 0.0)
                if size <= self.position_epsilon:
                    continue
                reference_ts = last_add.get(state_key) or base.get("timestamp") or datetime.now(timezone.utc).isoformat()
                normalized_price = _safe_float(base.get("price"), _safe_float(base.get("current_price"), 0.0))
                normalized_current = _safe_float(base.get("current_price"), normalized_price)
                snapshot_rows.append(
                    {
                        **base,
                        **skill,
                        "wallet_label": watch_row.get("label"),
                        "wallet_watchlist_approved": True,
                        "wallet_source": watch_row.get("source", "leaderboard_api"),
                        "wallet_region_scope": watch_row.get("region_scope"),
                        "min_wallet_score": watch_row.get("min_wallet_score"),
                        "source_wallet_reference_ts": reference_ts,
                        "source_wallet_average_entry": normalized_price,
                        "source_wallet_current_net_exposure": size,
                        "source_wallet_position_size": size,
                        "source_wallet_current_direction": side,
                        "source_wallet_last_add": last_add.get(state_key),
                        "source_wallet_last_reduce": last_reduce.get(state_key),
                        "source_wallet_last_close": last_close.get(state_key),
                        "source_wallet_direction_confidence": round(
                            _clip01(
                                (skill["wallet_quality_score"] * 0.55)
                                + (_clip01(size / max(avg_size, size, 1.0)) * 0.20)
                                + (_clip01(abs(normalized_current - normalized_price) + 0.25) * 0.25)
                            ),
                            4,
                        ),
                        "source_wallet_size_delta_ratio": 0.0,
                        "source_wallet_reduce_fraction": 0.0,
                        "weather_cluster_key": weather_city_date_cluster_key(base),
                    }
                )
        snapshot_df = pd.DataFrame(snapshot_rows)
        if snapshot_df.empty:
            return snapshot_df
        snapshot_df["timestamp"] = pd.to_datetime(snapshot_df.get("source_wallet_reference_ts"), utc=True, errors="coerce")
        snapshot_df = snapshot_df.sort_values("timestamp", ascending=False, na_position="last").reset_index(drop=True)
        return snapshot_df

    def _load_previous_snapshot(self) -> pd.DataFrame:
        if not self.snapshot_file.exists():
            return pd.DataFrame()
        try:
            snapshot = pd.read_csv(self.snapshot_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()
        if snapshot.empty:
            return snapshot
        snapshot["timestamp"] = pd.to_datetime(snapshot.get("source_wallet_reference_ts", snapshot.get("timestamp")), utc=True, errors="coerce")
        return snapshot

    def _write_snapshot(self, snapshot_df: pd.DataFrame):
        if snapshot_df is None or snapshot_df.empty:
            overwrite_csv_with_brain_mirrors(
                self.snapshot_file,
                pd.DataFrame(),
                family_hint=WEATHER_FAMILY,
                shared_logs_dir=self.logs_dir,
                include_shared=True,
            )
            return
        overwrite_csv_with_brain_mirrors(
            self.snapshot_file,
            snapshot_df,
            family_hint=WEATHER_FAMILY,
            shared_logs_dir=self.logs_dir,
            include_shared=True,
        )

    def _build_event_signal(
        self,
        row: dict,
        *,
        entry_intent: str,
        position_event: str,
        net_increase: bool,
        size_delta: float,
        reduce_fraction: float = 0.0,
        exit_signal: bool = False,
        reduce_signal: bool = False,
        reversal_signal: bool = False,
    ) -> dict:
        signal = dict(row)
        reference_ts = pd.to_datetime(
            signal.get("source_wallet_last_add")
            or signal.get("source_wallet_reference_ts")
            or signal.get("timestamp"),
            utc=True,
            errors="coerce",
        )
        if pd.isna(reference_ts):
            reference_ts = pd.to_datetime(
                signal.get("source_wallet_last_reduce")
                or signal.get("source_wallet_last_close"),
                utc=True,
                errors="coerce",
            )
        if pd.isna(reference_ts):
            reference_ts = pd.to_datetime(
                signal.get("source_wallet_last_close"),
                utc=True,
                errors="coerce",
            )
        if pd.isna(reference_ts):
            reference_ts = pd.Timestamp.now(tz="UTC")
        freshness_seconds = max(0.0, (pd.Timestamp.now(tz="UTC") - reference_ts).total_seconds())
        freshness_score = _clip01(1.0 - freshness_seconds / max(self.fresh_minutes * 60.0, 1.0))
        min_wallet_score = max(self.min_wallet_score, _safe_float(signal.get("min_wallet_score"), self.min_wallet_score))
        gate_reasons = []
        if entry_intent == "OPEN_LONG":
            if not bool(signal.get("wallet_watchlist_approved", True)):
                gate_reasons.append("wallet_not_approved")
            if _safe_float(signal.get("wallet_quality_score"), 0.0) < min_wallet_score:
                gate_reasons.append("wallet_quality_below_min")
            if not net_increase:
                gate_reasons.append("wallet_net_position_not_increased")
            if freshness_seconds > self.fresh_minutes * 60.0:
                gate_reasons.append("wallet_state_stale")
        signal.update(
            {
                "timestamp": reference_ts.isoformat(),
                "entry_intent": entry_intent,
                "source_wallet_position_event": position_event,
                "source_wallet_net_position_increased": bool(net_increase),
                "source_wallet_size_delta": float(size_delta),
                "source_wallet_size_delta_ratio": _clip01(size_delta / max(_safe_float(signal.get("source_wallet_position_size"), 1.0), 1.0)),
                "source_wallet_reduce_fraction": _clip01(reduce_fraction),
                "source_wallet_state_freshness_seconds": float(freshness_seconds),
                "source_wallet_freshness_score": float(freshness_score),
                "source_wallet_fresh": bool(freshness_seconds <= (self.fresh_minutes * 60.0)),
                "source_wallet_exit_signal": bool(exit_signal),
                "source_wallet_reduce_signal": bool(reduce_signal),
                "source_wallet_reversal_signal": bool(reversal_signal),
                "wallet_state_gate_pass": bool(not gate_reasons),
                "wallet_state_gate_reason": ",".join(gate_reasons),
                "signal_source": "weather_temperature_wallet",
                "analytics_only": False,
                "analytics_only_reason": None,
            }
        )
        return signal

    def _build_event_rows(self, current_snapshot: pd.DataFrame, previous_snapshot: pd.DataFrame) -> pd.DataFrame:
        if current_snapshot is None:
            current_snapshot = pd.DataFrame()
        if previous_snapshot is None:
            previous_snapshot = pd.DataFrame()

        prev_lookup = {}
        for _, row in previous_snapshot.iterrows():
            row_dict = row.to_dict()
            key = (_normalize_wallet(row_dict.get("trader_wallet")), _normalize_market_key(row_dict), _normalize_side(row_dict.get("outcome_side")))
            prev_lookup[key] = row_dict

        emitted_rows = []
        seen_prev_keys = set()

        for _, row in current_snapshot.iterrows():
            row_dict = row.to_dict()
            wallet = _normalize_wallet(row_dict.get("trader_wallet"))
            market_key = _normalize_market_key(row_dict)
            side = _normalize_side(row_dict.get("outcome_side"))
            current_key = (wallet, market_key, side)
            opposite_side = "NO" if side == "YES" else "YES"
            opposite_key = (wallet, market_key, opposite_side)

            prev_same = prev_lookup.get(current_key)
            prev_opposite = prev_lookup.get(opposite_key)
            current_size = _safe_float(row_dict.get("source_wallet_position_size"), 0.0)
            prev_same_size = _safe_float(prev_same.get("source_wallet_position_size"), 0.0) if prev_same else 0.0
            prev_opp_size = _safe_float(prev_opposite.get("source_wallet_position_size"), 0.0) if prev_opposite else 0.0

            if prev_opposite and prev_opp_size > self.position_epsilon and prev_same_size <= self.position_epsilon:
                emitted_rows.append(
                    self._build_event_signal(
                        prev_opposite,
                        entry_intent="CLOSE_LONG",
                        position_event="REVERSAL_EXIT",
                        net_increase=False,
                        size_delta=prev_opp_size,
                        reduce_fraction=1.0,
                        exit_signal=True,
                        reversal_signal=True,
                    )
                )
                seen_prev_keys.add(opposite_key)

            if prev_same is None or prev_same_size <= self.position_epsilon:
                emitted_rows.append(
                    self._build_event_signal(
                        row_dict,
                        entry_intent="OPEN_LONG",
                        position_event="REVERSAL_ENTRY" if prev_opp_size > self.position_epsilon else "NEW_ENTRY",
                        net_increase=True,
                        size_delta=current_size,
                        reversal_signal=prev_opp_size > self.position_epsilon,
                    )
                )
            elif current_size > prev_same_size + self.position_epsilon:
                emitted_rows.append(
                    self._build_event_signal(
                        row_dict,
                        entry_intent="OPEN_LONG",
                        position_event="SCALE_IN",
                        net_increase=True,
                        size_delta=current_size - prev_same_size,
                    )
                )
            elif current_size < prev_same_size - self.position_epsilon:
                reduce_fraction = (prev_same_size - current_size) / max(prev_same_size, 1.0)
                if current_size <= self.position_epsilon:
                    emitted_rows.append(
                        self._build_event_signal(
                            prev_same,
                            entry_intent="CLOSE_LONG",
                            position_event="FULL_EXIT",
                            net_increase=False,
                            size_delta=prev_same_size,
                            reduce_fraction=1.0,
                            exit_signal=True,
                        )
                    )
                elif reduce_fraction >= self.sharp_reduce_threshold:
                    reduced_row = dict(row_dict)
                    reduced_row["source_wallet_last_reduce"] = row_dict.get("source_wallet_reference_ts")
                    emitted_rows.append(
                        self._build_event_signal(
                            reduced_row,
                            entry_intent="CLOSE_LONG",
                            position_event="SHARP_REDUCE",
                            net_increase=False,
                            size_delta=prev_same_size - current_size,
                            reduce_fraction=reduce_fraction,
                            exit_signal=True,
                            reduce_signal=True,
                        )
                    )
            seen_prev_keys.add(current_key)

        for prev_key, prev_row in prev_lookup.items():
            if prev_key in seen_prev_keys:
                continue
            prev_size = _safe_float(prev_row.get("source_wallet_position_size"), 0.0)
            if prev_size <= self.position_epsilon:
                continue
            emitted_rows.append(
                self._build_event_signal(
                    prev_row,
                    entry_intent="CLOSE_LONG",
                    position_event="FULL_EXIT",
                    net_increase=False,
                    size_delta=prev_size,
                    reduce_fraction=1.0,
                    exit_signal=True,
                )
            )
        return pd.DataFrame(emitted_rows)

    def _apply_forecast_context(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        if signals_df is None or signals_df.empty:
            return pd.DataFrame() if signals_df is None else signals_df.copy()

        rows = []
        for _, row in signals_df.iterrows():
            row_dict = row.to_dict()
            forecast = self.forecast_service.fetch_market_forecast(row_dict)
            row_dict.update(forecast)
            fair_yes_probability = _clip01(row_dict.get("forecast_p_hit_interval", 0.5))
            side = _normalize_side(row_dict.get("outcome_side"))
            fair_side_probability = fair_yes_probability if side == "YES" else 1.0 - fair_yes_probability
            market_probability = _clip01(
                row_dict.get("current_price")
                or row_dict.get("last_trade_price")
                or row_dict.get("price")
                or 0.5
            )
            row_dict["weather_fair_probability_yes"] = round(fair_yes_probability, 6)
            row_dict["weather_fair_probability_side"] = round(fair_side_probability, 6)
            row_dict["weather_market_probability"] = round(market_probability, 6)
            row_dict["weather_forecast_edge"] = round(float(fair_side_probability - market_probability), 6)
            row_dict["weather_forecast_confirms_direction"] = bool((fair_side_probability - market_probability) > 0)
            if str(row_dict.get("entry_intent", "")).upper() == "OPEN_LONG":
                if not bool(row_dict.get("weather_parseable", False)):
                    row_dict["analytics_only"] = True
                    row_dict["analytics_only_reason"] = "weather_parse_failed"
                elif not bool(row_dict.get("forecast_ready", False)):
                    row_dict["analytics_only"] = True
                    row_dict["analytics_only_reason"] = str(row_dict.get("forecast_missing_reason") or "weather_forecast_missing")
                elif bool(row_dict.get("forecast_stale", True)):
                    row_dict["analytics_only"] = True
                    row_dict["analytics_only_reason"] = "weather_forecast_stale"
            rows.append(row_dict)
        return pd.DataFrame(rows)

    def _apply_cluster_consensus(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        if signals_df is None or signals_df.empty:
            return pd.DataFrame() if signals_df is None else signals_df.copy()
        out = signals_df.copy()
        if "wallet_agreement_score" not in out.columns:
            out["wallet_agreement_score"] = 0.5
        if "wallet_conflict_with_stronger" not in out.columns:
            out["wallet_conflict_with_stronger"] = False
        if "wallet_stronger_conflict_score" not in out.columns:
            out["wallet_stronger_conflict_score"] = 0.0
        if "wallet_support_strength" not in out.columns:
            out["wallet_support_strength"] = 0.0

        entry_mask = out["entry_intent"].astype(str).str.upper() == "OPEN_LONG"
        if not entry_mask.any():
            return out

        work = out[entry_mask].copy()
        work["_cluster_key"] = work.apply(lambda row: weather_city_date_cluster_key(row.to_dict()), axis=1)
        for cluster_key, group in work.groupby("_cluster_key"):
            if not cluster_key:
                continue
            eligible = group[
                (~group.get("analytics_only", pd.Series(False, index=group.index)).fillna(False).astype(bool))
                & group["wallet_watchlist_approved"].fillna(False).astype(bool)
            ].copy()
            for idx, row in group.iterrows():
                row_side = _normalize_side(row.get("outcome_side"))
                same_side = eligible[eligible["outcome_side"].astype(str).str.upper() == row_side]
                opp_side = eligible[eligible["outcome_side"].astype(str).str.upper() != row_side]
                support_strength = (
                    same_side["wallet_quality_score"].astype(float).fillna(0.0)
                    * same_side["source_wallet_direction_confidence"].astype(float).fillna(0.0)
                ).sum() if not same_side.empty else 0.0
                oppose_strength = (
                    opp_side["wallet_quality_score"].astype(float).fillna(0.0)
                    * opp_side["source_wallet_direction_confidence"].astype(float).fillna(0.0)
                ).max() if not opp_side.empty else 0.0
                total_strength = max(support_strength + oppose_strength, 1e-9)
                agreement_score = support_strength / total_strength
                conflict = oppose_strength > (support_strength + 0.08)
                out.at[idx, "wallet_agreement_score"] = round(_clip01(agreement_score), 4)
                out.at[idx, "wallet_conflict_with_stronger"] = bool(conflict)
                out.at[idx, "wallet_stronger_conflict_score"] = round(float(oppose_strength), 4)
                out.at[idx, "wallet_support_strength"] = round(float(support_strength), 4)
                if conflict:
                    existing_reason = str(out.at[idx, "wallet_state_gate_reason"] or "").strip()
                    parts = [part for part in existing_reason.split(",") if part]
                    if "conflict_with_stronger_wallet" not in parts:
                        parts.append("conflict_with_stronger_wallet")
                    out.at[idx, "wallet_state_gate_reason"] = ",".join(parts)
                    out.at[idx, "wallet_state_gate_pass"] = False
        return out

    def build_cycle_signals(self, markets_df: pd.DataFrame | None = None) -> pd.DataFrame:
        current_snapshot = self._build_current_snapshot(markets_df)
        previous_snapshot = self._load_previous_snapshot()
        signals_df = self._build_event_rows(current_snapshot, previous_snapshot)
        signals_df = self._apply_forecast_context(signals_df)
        signals_df = self._apply_cluster_consensus(signals_df)
        self._write_snapshot(current_snapshot)
        if signals_df is None or signals_df.empty:
            return pd.DataFrame()
        signals_df["timestamp"] = pd.to_datetime(signals_df.get("timestamp"), utc=True, errors="coerce")
        signals_df = signals_df.sort_values("timestamp", ascending=False, na_position="last").reset_index(drop=True)
        return signals_df

    def _time_left_feature(self, end_date, reference_time=None) -> float:
        if not end_date:
            return 0.5
        try:
            end_dt = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
            ref_dt = pd.to_datetime(reference_time, utc=True, errors="coerce") if reference_time is not None else datetime.now(timezone.utc)
            if pd.isna(ref_dt):
                ref_dt = datetime.now(timezone.utc)
            seconds_left = max((end_dt - ref_dt).total_seconds(), 0)
            return _clip01(seconds_left / (7 * 24 * 3600))
        except Exception:
            return 0.5

    def score_candidates(self, signals_df: pd.DataFrame, markets_df: pd.DataFrame | None = None) -> pd.DataFrame:
        if signals_df is None or signals_df.empty:
            return pd.DataFrame()

        rows = []
        for _, row in signals_df.iterrows():
            raw = row.to_dict()
            side = _normalize_side(raw.get("outcome_side"))
            entry_intent = str(raw.get("entry_intent", "OPEN_LONG") or "OPEN_LONG").upper()
            market_prob = _clip01(raw.get("weather_market_probability", raw.get("current_price", raw.get("price", 0.5))))
            fair_yes = _clip01(raw.get("weather_fair_probability_yes", raw.get("forecast_p_hit_interval", 0.5)))
            fair_side = fair_yes if side == "YES" else 1.0 - fair_yes
            edge = float(fair_side - market_prob)
            liquidity = _safe_float(raw.get("liquidity"), 0.0)
            volume = _safe_float(raw.get("volume"), 0.0)
            best_bid = _safe_float(raw.get("best_bid"), market_prob)
            best_ask = _safe_float(raw.get("best_ask"), market_prob)
            spread = abs(best_ask - best_bid)
            liquidity_score = _clip01(raw.get("liquidity_score", liquidity / 50000.0 if liquidity > 0 else 0.0))
            volume_score = _clip01(raw.get("volume_score", volume / 100000.0 if volume > 0 else 0.0))
            uncertainty_c = max(0.25, _safe_float(raw.get("forecast_uncertainty_c"), 1.0))
            drift_c = abs(_safe_float(raw.get("forecast_drift_c"), 0.0))
            stability_score = _clip01(1.0 - min(drift_c / max(uncertainty_c * 2.0, 0.5), 1.0))
            margin_lower = _safe_float(raw.get("forecast_margin_to_lower_c"), 0.0)
            margin_upper = _safe_float(raw.get("forecast_margin_to_upper_c"), 0.0)
            question_type = str(raw.get("weather_question_type") or "").strip().lower()
            if question_type == "threshold":
                margin_c = margin_lower if side == "YES" else -margin_lower
            elif side == "YES":
                margin_c = min(margin_lower, margin_upper)
            else:
                margin_c = max(-margin_lower, -margin_upper)
            margin_score = _clip01(0.5 + (margin_c / max(uncertainty_c * 3.0, 0.75)) * 0.5)
            wallet_quality = _clip01(raw.get("wallet_quality_score", 0.5))
            direction_conf = _clip01(raw.get("source_wallet_direction_confidence", 0.0))
            agreement_score = _clip01(raw.get("wallet_agreement_score", 0.5))
            size_score = _clip01(raw.get("source_wallet_size_delta_ratio", 0.0))
            spread_score = _clip01(1.0 - min(spread / max(self.max_spread, 0.01), 1.0))
            time_left = self._time_left_feature(raw.get("end_date"), raw.get("timestamp"))
            expected_return = edge
            weather_entry_gate_fail = (
                entry_intent == "OPEN_LONG"
                and (
                    bool(raw.get("analytics_only", False))
                    or not bool(raw.get("wallet_state_gate_pass", True))
                    or liquidity_score < self.min_liquidity_score
                    or spread > self.max_spread
                )
            )

            confidence = 0.0
            if entry_intent == "CLOSE_LONG":
                model_confidence = max(0.70, (direction_conf * 0.6) + 0.30)
                confidence = model_confidence
                signal_label = "WEATHER_SOURCE_EXIT"
                reason = f"weather_exit event={raw.get('source_wallet_position_event')} confidence={model_confidence:.2f}"
                action_code = 2
                p_tp = 0.50
            else:
                p_tp = fair_side
                # Profitability-first: edge and expected_return carry 48 %,
                # probability (p_tp, stability) carries 28 %, wallet 24 %.
                _edge_norm = _clip01(max(edge, 0.0) / max(self.min_forecast_edge * 2.0, 0.02))
                model_confidence = _clip01(
                    (_edge_norm * 0.28)
                    + (_clip01(max(expected_return, 0.0) * 4.0) * 0.20)
                    + (p_tp * 0.18)
                    + (stability_score * 0.10)
                    + (wallet_quality * 0.10)
                    + (direction_conf * 0.06)
                    + (agreement_score * 0.04)
                    + (liquidity_score * 0.02)
                    + (spread_score * 0.02)
                )
                if edge <= 0 or not bool(raw.get("forecast_ready", False)):
                    model_confidence = min(model_confidence, 0.44)
                confidence = model_confidence
                # Action code: profitability-weighted score drives the tier
                _weather_profit_score = _clip01(
                    _edge_norm * 0.40
                    + _clip01(max(expected_return, 0.0) * 4.0) * 0.30
                    + confidence * 0.30
                )
                if _weather_profit_score < 0.25:
                    action_code = 0
                elif _weather_profit_score < 0.45:
                    action_code = 1
                elif _weather_profit_score < 0.70:
                    action_code = 2
                else:
                    action_code = 3
                signal_label = {
                    0: "IGNORE",
                    1: "LOW-CONFIDENCE WATCH",
                    2: "STRONG WEATHER OPPORTUNITY",
                    3: "HIGHEST-RANKED WEATHER SIGNAL",
                }[action_code]
                reason = (
                    f"weather_fair={fair_side:.2f}, market_prob={market_prob:.2f}, edge={edge:.4f}, "
                    f"wallet_q={wallet_quality:.2f}, dir_conf={direction_conf:.2f}, "
                    f"stability={stability_score:.2f}, liq={liquidity_score:.2f}, spread={spread:.3f}"
                )

            row_out = {
                **raw,
                "market_family": raw.get("market_family"),
                "entry_model_family": "weather_temperature_hybrid",
                "entry_model_version": "weather_v1",
                "technical_regime_bucket": raw.get(
                    "technical_regime_bucket",
                    "weather_forecast_confirmed" if edge >= self.min_forecast_edge else "weather_forecast_mixed",
                ),
                "trader_win_rate": _clip01(raw.get("wallet_temp_hit_rate_90d", wallet_quality)),
                "normalized_trade_size": size_score,
                "outcome_side": side,
                "side": side,
                "current_price": market_prob,
                "entry_price": market_prob,
                "price": market_prob,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "time_left": time_left,
                "liquidity_score": liquidity_score,
                "volume_score": volume_score,
                "probability_momentum": _clip01(abs(edge) / max(self.min_forecast_edge * 2.0, 0.02)),
                "volatility_score": _clip01(uncertainty_c / 5.0),
                "whale_pressure": _clip01((wallet_quality * 0.7) + (size_score * 0.3)),
                "market_structure_score": _clip01((liquidity_score * 0.35) + (volume_score * 0.20) + (margin_score * 0.25) + (spread_score * 0.20)),
                "volatility_risk": _clip01(uncertainty_c / 5.0),
                "time_decay_score": _clip01(1.0 - time_left),
                "model_confidence": round(float(model_confidence), 4),
                "confidence": round(float(confidence), 4),
                "signal_label": signal_label,
                "action_code": int(action_code),
                "reason": reason,
                "reason_summary": reason,
                "recommended_action": entry_intent,
                "action": entry_intent,
                "edge_score": round(float(edge), 6),
                "expected_return": round(float(expected_return), 6),
                "p_tp_before_sl": round(float(p_tp), 6),
                "risk_adjusted_ev": round(float(expected_return * stability_score * (0.5 + (liquidity_score * 0.5))), 6),
                "entry_ev": round(float(expected_return * confidence), 6),
                "execution_quality_score": round(float((spread_score * 0.45) + (liquidity_score * 0.35) + (volume_score * 0.20)), 6),
                "weather_forecast_margin_score": round(float(margin_score), 6),
                "weather_forecast_stability_score": round(float(stability_score), 6),
                "weather_min_score": 0.45,
                "weather_max_spread": float(self.max_spread),
                "weather_min_liquidity_score": float(self.min_liquidity_score),
                "weather_min_forecast_edge": float(self.min_forecast_edge),
                "weather_entry_gate_fail": bool(weather_entry_gate_fail),
                "weather_entry_allowed_by_forecast": bool(edge >= self.min_forecast_edge and raw.get("forecast_ready", False)),
            }
            decision_score = PredictionLayer.select_signal_score(row_out)
            row_out["decision_score"] = round(float(decision_score), 4)
            row_out["confidence"] = round(float(max(row_out["model_confidence"], decision_score)), 4)
            ranking_score = row_out["confidence"]
            if entry_intent != "CLOSE_LONG":
                if ranking_score < 0.45:
                    action_code = 0
                elif ranking_score < 0.60:
                    action_code = 1
                elif ranking_score < 0.78:
                    action_code = 2
                else:
                    action_code = 3
                row_out["action_code"] = int(action_code)
                row_out["signal_label"] = {
                    0: "IGNORE",
                    1: "LOW-CONFIDENCE WATCH",
                    2: "STRONG WEATHER OPPORTUNITY",
                    3: "HIGHEST-RANKED WEATHER SIGNAL",
                }[action_code]
            rows.append(row_out)
        scored_df = pd.DataFrame(rows)
        if not scored_df.empty:
            scored_df["timestamp"] = pd.to_datetime(scored_df.get("timestamp"), utc=True, errors="coerce")
            scored_df = scored_df.sort_values("timestamp", ascending=False, na_position="last").reset_index(drop=True)
        return scored_df

    def apply_active_exit_rules(self, trade_manager, markets_df: pd.DataFrame | None = None) -> list[dict]:
        if trade_manager is None:
            return []
        lookups = self._build_market_lookup(markets_df)
        exit_events = []
        for trade in list(getattr(trade_manager, "active_trades", {}).values()):
            trade_family = str(getattr(trade, "market_family", "") or "").strip().lower()
            if not trade_family.startswith("weather_temperature"):
                continue
            trade_row = dict(getattr(trade, "__dict__", {}) or {})
            matched_market = self._match_market_row(trade_row, lookups)
            if matched_market:
                for field in ("liquidity", "volume", "best_bid", "best_ask", "last_trade_price", "current_price", "end_date", "market_url"):
                    if field not in trade_row or trade_row.get(field) in [None, "", 0]:
                        trade_row[field] = matched_market.get(field)
                trade_row["market_slug"] = trade_row.get("market_slug") or matched_market.get("market_slug")
            forecast = self.forecast_service.fetch_market_forecast(trade_row)
            trade_row.update(forecast)
            fair_yes = _clip01(trade_row.get("forecast_p_hit_interval", 0.5))
            side = _normalize_side(trade_row.get("outcome_side"))
            fair_side = fair_yes if side == "YES" else 1.0 - fair_yes
            current_price = _clip01(
                trade_row.get("current_price")
                or trade_row.get("last_trade_price")
                or getattr(trade, "current_price", getattr(trade, "entry_price", 0.5))
            )
            spread = abs(
                _safe_float(trade_row.get("best_ask"), current_price)
                - _safe_float(trade_row.get("best_bid"), current_price)
            )
            edge = fair_side - current_price
            close_reason = None
            if not bool(trade_row.get("forecast_ready", False)) or bool(trade_row.get("forecast_stale", True)):
                close_reason = "weather_forecast_stale"
            elif edge < min(self.min_forecast_edge * 0.5, 0.03):
                close_reason = "weather_forecast_edge_collapse"
            elif edge < -0.05:
                close_reason = "weather_market_rich_vs_forecast"
            elif spread > self.max_spread:
                close_reason = "weather_liquidity_stress"
            else:
                end_date = pd.to_datetime(trade_row.get("end_date"), utc=True, errors="coerce")
                if pd.notna(end_date):
                    minutes_left = max(0.0, (end_date - pd.Timestamp.now(tz="UTC")).total_seconds() / 60.0)
                    if minutes_left <= 180:
                        close_reason = "weather_resolution_near"
            if close_reason:
                trade.current_price = float(current_price)
                trade.close(exit_price=float(current_price), reason=close_reason)
                exit_events.append(
                    {
                        "trade_key": f"{getattr(trade, 'token_id', '')}|{getattr(trade, 'condition_id', '')}|{getattr(trade, 'outcome_side', '')}",
                        "reason": close_reason,
                        "market": getattr(trade, "market", None),
                    }
                )
        return exit_events
