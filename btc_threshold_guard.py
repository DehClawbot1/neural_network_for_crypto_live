from __future__ import annotations

import re
from typing import Iterable, Mapping


_TITLE_PATTERN = re.compile(
    r"will the price of bitcoin be\s+(?:above|greater than)\s+\$?([\d,]+(?:\.\d+)?)\s+on\s+([a-z]+)\s+(\d{1,2})",
    re.IGNORECASE,
)
_SLUG_PATTERN = re.compile(
    r"bitcoin-(?:above|greater-than)-(\d+(?:\.\d+)?)(k)?-on-([a-z]+)-(\d{1,2})",
    re.IGNORECASE,
)


def _get(source, key: str, default=None):
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _parse_threshold_value(number_text: str, has_k_suffix: bool = False) -> float:
    raw = str(number_text or "").strip().replace(",", "")
    value = float(raw)
    if has_k_suffix:
        value *= 1000.0
    return float(value)


def parse_btc_price_threshold_market(market=None, market_slug=None):
    title = str(market or "").strip()
    slug = str(market_slug or "").strip().lower()

    title_match = _TITLE_PATTERN.search(title)
    if title_match:
        threshold = _parse_threshold_value(title_match.group(1), has_k_suffix=False)
        month = title_match.group(2).strip().lower()
        day = int(title_match.group(3))
        return {
            "market_family": "btc_price_threshold",
            "threshold_price": threshold,
            "expiry_key": f"{month}-{day:02d}",
            "market_title": title,
            "market_slug": slug or None,
        }

    slug_match = _SLUG_PATTERN.search(slug)
    if slug_match:
        threshold = _parse_threshold_value(slug_match.group(1), has_k_suffix=bool(slug_match.group(2)))
        month = slug_match.group(3).strip().lower()
        day = int(slug_match.group(4))
        return {
            "market_family": "btc_price_threshold",
            "threshold_price": threshold,
            "expiry_key": f"{month}-{day:02d}",
            "market_title": title or None,
            "market_slug": slug,
        }

    return None


def describe_btc_price_threshold_position(source):
    parsed = parse_btc_price_threshold_market(
        market=_get(source, "market", _get(source, "market_title")),
        market_slug=_get(source, "market_slug"),
    )
    if not parsed:
        return None

    outcome_side = str(_get(source, "outcome_side", _get(source, "side", "")) or "").strip().upper()
    if outcome_side not in {"YES", "NO"}:
        return None

    return {
        **parsed,
        "outcome_side": outcome_side,
        "market": str(_get(source, "market", _get(source, "market_title", "")) or ""),
        "condition_id": str(_get(source, "condition_id", "") or ""),
        "token_id": str(_get(source, "token_id", "") or ""),
    }


def btc_price_threshold_positions_conflict(left, right) -> bool:
    left_desc = describe_btc_price_threshold_position(left)
    right_desc = describe_btc_price_threshold_position(right)
    if not left_desc or not right_desc:
        return False
    if left_desc["expiry_key"] != right_desc["expiry_key"]:
        return False
    if left_desc["outcome_side"] == right_desc["outcome_side"]:
        return False

    yes_threshold = (
        left_desc["threshold_price"]
        if left_desc["outcome_side"] == "YES"
        else right_desc["threshold_price"]
    )
    no_threshold = (
        left_desc["threshold_price"]
        if left_desc["outcome_side"] == "NO"
        else right_desc["threshold_price"]
    )
    return bool(yes_threshold >= no_threshold)


def find_conflicting_btc_price_threshold_position(candidate, open_positions: Iterable):
    candidate_desc = describe_btc_price_threshold_position(candidate)
    if not candidate_desc:
        return None

    for existing in open_positions or []:
        existing_desc = describe_btc_price_threshold_position(existing)
        if not existing_desc:
            continue
        if not btc_price_threshold_positions_conflict(candidate_desc, existing_desc):
            continue
        return {
            "candidate": candidate_desc,
            "existing": existing_desc,
            "expiry_key": candidate_desc["expiry_key"],
            "candidate_threshold_price": candidate_desc["threshold_price"],
            "existing_threshold_price": existing_desc["threshold_price"],
            "candidate_outcome_side": candidate_desc["outcome_side"],
            "existing_outcome_side": existing_desc["outcome_side"],
        }
    return None
