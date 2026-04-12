from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BTC_BRAIN_ID = "btc_brain"
WEATHER_BRAIN_ID = "weather_temperature_brain"
BTC_FAMILY = "btc"
WEATHER_FAMILY = "weather_temperature"

_WEATHER_MARKET_KEYWORDS = (
    "highest temperature",
    "high temperature",
    "maximum temperature",
    "temperature will be",
    "temperature be between",
    "temperature be ",
)


@dataclass(frozen=True)
class BrainContext:
    brain_id: str
    market_family: str
    family_prefix: str
    shared_logs_dir: Path
    shared_weights_dir: Path
    logs_dir: Path
    weights_dir: Path


def _normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def normalize_market_family(value: Any) -> str:
    text = _normalize_text(value)
    if text.startswith(WEATHER_FAMILY):
        return WEATHER_FAMILY
    if text in {"", "nan", "none"}:
        return ""
    return BTC_FAMILY


def _looks_like_weather_temperature_text(text: str) -> bool:
    lowered = _normalize_text(text)
    if not lowered:
        return False
    if WEATHER_FAMILY in lowered:
        return True
    if "weather" in lowered and "temperature" in lowered:
        return True
    return any(keyword in lowered for keyword in _WEATHER_MARKET_KEYWORDS)


def infer_market_family_from_row(row: dict[str, Any] | pd.Series | None) -> str:
    if row is None:
        row_dict = {}
    elif hasattr(row, "to_dict"):
        row_dict = dict(row.to_dict())
    else:
        row_dict = dict(row)
    explicit = normalize_market_family(row_dict.get("market_family"))
    if explicit:
        return explicit
    # NOTE: only market-content fields are used for inference.
    # brain_id / artifact_group / model_kind are intentionally EXCLUDED because they
    # can contain the substring "weather_temperature" (e.g. "weather_temperature_brain")
    # and would cause any row ever touched by the weather brain to be mis-classified —
    # a self-reinforcing contamination loop.
    text_parts = [
        row_dict.get("market"),
        row_dict.get("market_title"),
        row_dict.get("question"),
        row_dict.get("slug"),
        row_dict.get("market_slug"),
    ]
    combined = " | ".join("" if part is None or pd.isna(part) else str(part) for part in text_parts)
    if _looks_like_weather_temperature_text(combined):
        return WEATHER_FAMILY
    return BTC_FAMILY


def resolve_brain_context(
    market_family: str | None = None,
    *,
    brain_id: str | None = None,
    shared_logs_dir: str | Path = "logs",
    shared_weights_dir: str | Path = "weights",
) -> BrainContext:
    shared_logs_path = Path(shared_logs_dir)
    shared_weights_path = Path(shared_weights_dir)
    shared_logs_path.mkdir(parents=True, exist_ok=True)
    shared_weights_path.mkdir(parents=True, exist_ok=True)

    explicit_brain = _normalize_text(brain_id)
    family = normalize_market_family(market_family)
    if explicit_brain == WEATHER_BRAIN_ID or family == WEATHER_FAMILY:
        family = WEATHER_FAMILY
        chosen_brain = WEATHER_BRAIN_ID
        logs_dir = shared_logs_path / WEATHER_FAMILY
        weights_dir = shared_weights_path / WEATHER_FAMILY
    else:
        family = BTC_FAMILY
        chosen_brain = BTC_BRAIN_ID
        logs_dir = shared_logs_path / BTC_FAMILY
        weights_dir = shared_weights_path / BTC_FAMILY

    logs_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    return BrainContext(
        brain_id=chosen_brain,
        market_family=family,
        family_prefix=family,
        shared_logs_dir=shared_logs_path,
        shared_weights_dir=shared_weights_path,
        logs_dir=logs_dir,
        weights_dir=weights_dir,
    )


def list_brain_contexts(
    *,
    shared_logs_dir: str | Path = "logs",
    shared_weights_dir: str | Path = "weights",
) -> list[BrainContext]:
    return [
        resolve_brain_context(BTC_FAMILY, shared_logs_dir=shared_logs_dir, shared_weights_dir=shared_weights_dir),
        resolve_brain_context(WEATHER_FAMILY, shared_logs_dir=shared_logs_dir, shared_weights_dir=shared_weights_dir),
    ]


def ensure_market_family_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    if "market_family" not in out.columns:
        out["market_family"] = out.apply(lambda row: infer_market_family_from_row(row.to_dict()), axis=1)
    else:
        normalized = out["market_family"].map(normalize_market_family)
        missing = normalized.eq("")
        if missing.any():
            inferred = out.loc[missing].apply(lambda row: infer_market_family_from_row(row.to_dict()), axis=1)
            normalized.loc[missing] = inferred
        out["market_family"] = normalized
    return out


def filter_frame_for_brain(df: pd.DataFrame, brain_context: BrainContext) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = ensure_market_family_column(df)
    family_series = out["market_family"].fillna("").astype(str).str.lower()
    if brain_context.family_prefix == WEATHER_FAMILY:
        # Primary filter: market_family tag starts with "weather_temperature".
        # Secondary guard: also verify the market title actually looks like weather.
        # This prevents contaminated rows (where a non-weather market was mis-tagged
        # by the brain_id inference loop) from leaking into the weather brain's data.
        tag_mask = family_series.str.startswith(WEATHER_FAMILY)
        def _title_confirms_weather(row) -> bool:
            text_parts = [
                row.get("market"), row.get("market_title"),
                row.get("question"), row.get("slug"), row.get("market_slug"),
            ]
            combined = " | ".join(
                "" if part is None or (isinstance(part, float) and pd.isna(part)) else str(part)
                for part in text_parts
            ).strip(" |")
            if not combined:
                # No market title available — nothing to contradict the tag, keep the row.
                return True
            return _looks_like_weather_temperature_text(combined)
        content_mask = out.apply(_title_confirms_weather, axis=1)
        mask = tag_mask & content_mask
        dropped = int(tag_mask.sum()) - int(mask.sum())
        if dropped > 0:
            import logging as _logging
            _logging.warning(
                "filter_frame_for_brain[weather]: dropped %d row(s) whose market_family "
                "tag was 'weather_temperature' but market title did not confirm it "
                "(likely contamination via brain_id inference loop).",
                dropped,
            )
        out = out[mask].copy()
        out["market_family"] = WEATHER_FAMILY
    else:
        mask = ~family_series.str.startswith(WEATHER_FAMILY)
        out = out[mask].copy()
        out["market_family"] = BTC_FAMILY
    if out.empty:
        return out
    out["brain_id"] = brain_context.brain_id
    return out


def active_runtime_identity(
    brain_context: BrainContext,
    *,
    active_model_group: str,
    active_model_kind: str,
    active_regime: str,
) -> dict[str, str]:
    return {
        "brain_id": brain_context.brain_id,
        "market_family": brain_context.market_family,
        "active_model_group": str(active_model_group or ""),
        "active_model_kind": str(active_model_kind or ""),
        "active_regime": str(active_regime or ""),
    }
