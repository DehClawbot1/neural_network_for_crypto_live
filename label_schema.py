"""
label_schema.py
Single source of truth for every learning target column.

Each entry in LABEL_REGISTRY describes:
  group       — "alpha" | "path" | "execution"
  type        — "binary" | "continuous"
  family      — list of market families that use this label
  formula     — plain-language description
  requires    — upstream columns that must be non-null
  min_reliability — minimum reliability_score for a row to be used
  missing_flag    — column written as True when this label cannot be computed

Rules:
  1. Alpha targets live exclusively in probability space [0, 1].
  2. Path targets live in return space (or direction binary).
  3. Execution targets come from real execution_feedback.csv, not from path data.
  4. Targets from different groups must never be mixed in a single column.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LABEL_REGISTRY: dict[str, dict[str, Any]] = {

    # ══════════════════════════════════════════════════════════════════
    # GROUP A — ALPHA TRUTH
    # Was the market mispriced at entry?  (probability space only)
    # ══════════════════════════════════════════════════════════════════

    "btc_prob_up_15m": {
        "group": "alpha",
        "type": "binary",
        "family": ["btc"],
        "formula": "int(forward_return_15m > 0)",
        "horizon": "15m",
        "requires": ["forward_return_15m"],
        "min_reliability": 1.0,
        "missing_flag": None,
        "note": "Replaces btc_fair_prob_up_15m (which was return, not probability).",
    },
    "btc_prob_up_60m": {
        "group": "alpha",
        "type": "binary",
        "family": ["btc"],
        "formula": "int(end_of_60m_price > entry_price)",
        "horizon": "60m",
        "requires": ["target_up"],
        "min_reliability": 1.0,
        "missing_flag": None,
        "note": "Replaces btc_fair_prob_up_60m (clearer name, same formula).",
    },
    "btc_market_edge": {
        "group": "alpha",
        "type": "continuous",
        "family": ["btc"],
        "formula": "btc_prob_up_15m − market_implied_prob",
        "horizon": "15m",
        "requires": ["forward_return_15m", "market_implied_prob"],
        "min_reliability": 1.0,
        "missing_flag": "btc_implied_prob_missing",
        "note": "Pure probability space. Replaces btc_mispricing_edge which mixed return + prob units.",
    },
    "weather_contract_resolved_yes": {
        "group": "alpha",
        "type": "binary",
        "family": ["weather"],
        "formula": "1 if Polymarket resolution_outcome=='YES' else 0",
        "horizon": "contract resolution",
        "requires": ["condition_id"],
        "min_reliability": 1.0,
        "missing_flag": "weather_resolution_unavailable",
        "note": "Real-world resolution. Replaces weather_resolution_yes_prob (CLOB direction proxy).",
    },
    "weather_market_edge": {
        "group": "alpha",
        "type": "continuous",
        "family": ["weather"],
        "formula": "weather_contract_resolved_yes − market_implied_prob",
        "horizon": "contract resolution",
        "requires": ["weather_contract_resolved_yes", "market_implied_prob"],
        "min_reliability": 1.0,
        "missing_flag": "weather_market_edge_unavailable",
        "note": "Pure probability space. Replaces weather_mispricing_edge.",
    },

    # ══════════════════════════════════════════════════════════════════
    # GROUP B — PATH TRUTH
    # What happened to the price in the hold window?
    # (return / direction space — never mixed with probability space)
    # ══════════════════════════════════════════════════════════════════

    "path_return_15m": {
        "group": "path",
        "type": "continuous",
        "family": ["btc", "weather"],
        "formula": "(price_at_15m − entry_price) / entry_price",
        "horizon": "15m",
        "requires": ["forward_return_15m"],
        "min_reliability": 0.5,
        "missing_flag": None,
        "note": "Renamed from forward_return_15m. Same computation.",
    },
    "path_return_60m_end": {
        "group": "path",
        "type": "continuous",
        "family": ["btc", "weather"],
        "formula": "(end_of_60m_price − entry_price) / entry_price",
        "horizon": "60m",
        "requires": ["entry_price"],
        "min_reliability": 0.5,
        "missing_flag": None,
    },
    "path_tp_hit_60m": {
        "group": "path",
        "type": "binary",
        "family": ["btc"],           # BTC only — % thresholds are meaningless for weather
        "formula": "1 if TP(+4%) hit before SL(-3%) in 60m CLOB path",
        "horizon": "60m",
        "tp": "+4%",
        "sl": "-3%",
        "requires": ["tp_before_sl_60m"],
        "min_reliability": 1.0,
        "missing_flag": None,
        "note": (
            "BTC family only. Polymarket binary contracts trade near 0/1 cents, "
            "so % TP/SL thresholds are invalid and should not be applied to weather."
        ),
    },
    "path_mfe_60m": {
        "group": "path",
        "type": "continuous",
        "family": ["btc"],
        "formula": "max((tick − entry) / entry) over 60m window",
        "horizon": "60m",
        "requires": ["mfe_60m"],
        "min_reliability": 0.5,
        "missing_flag": None,
    },
    "path_mae_60m": {
        "group": "path",
        "type": "continuous",
        "family": ["btc"],
        "formula": "min((tick − entry) / entry) over 60m window",
        "horizon": "60m",
        "requires": ["mae_60m"],
        "min_reliability": 0.5,
        "missing_flag": None,
    },

    # ══════════════════════════════════════════════════════════════════
    # GROUP C — EXECUTION TRUTH
    # Could the trade actually be entered and exited efficiently?
    # Source: execution_feedback.csv (None in contract_targets.csv)
    # ══════════════════════════════════════════════════════════════════

    "exec_fill_prob_5s": {
        "group": "execution",
        "type": "binary",
        "family": ["btc", "weather"],
        "formula": "1 if order filled within 5 seconds of placement",
        "source": "execution_feedback.csv",
        "requires": ["fill_latency_ms", "actual_filled"],
        "min_reliability": 1.0,
        "missing_flag": "exec_cost_unavailable",
    },
    "exec_fill_prob_30s": {
        "group": "execution",
        "type": "binary",
        "family": ["btc", "weather"],
        "formula": "1 if order filled within 30 seconds of placement",
        "source": "execution_feedback.csv",
        "requires": ["fill_latency_ms", "actual_filled"],
        "min_reliability": 1.0,
        "missing_flag": "exec_cost_unavailable",
    },
    "exec_expected_pnl_after_cost": {
        "group": "execution",
        "type": "continuous",
        "family": ["btc", "weather"],
        "formula": "path_return_15m − (spread_at_entry×0.5 + actual_slippage + fee_rate)",
        "source": "execution_feedback.csv JOIN contract_targets.csv",
        "requires": ["actual_slippage", "spread_at_entry", "forward_return_15m"],
        "min_reliability": 1.0,
        "missing_flag": "exec_cost_unavailable",
        "note": "Replaces btc_ev_after_cost (fake return×0.99). Cost from real execution data.",
    },
    "exec_slippage": {
        "group": "execution",
        "type": "continuous",
        "family": ["btc", "weather"],
        "formula": "(actual_fill_price − mid_at_placement) / mid_at_placement",
        "source": "execution_feedback.csv",
        "requires": ["actual_slippage"],
        "min_reliability": 1.0,
        "missing_flag": "exec_cost_unavailable",
    },
    "exec_cancel_risk": {
        "group": "execution",
        "type": "binary",
        "family": ["btc", "weather"],
        "formula": "1 if order cancelled before any fill",
        "source": "execution_feedback.csv",
        "requires": ["actual_filled"],
        "min_reliability": 1.0,
        "missing_flag": "exec_cost_unavailable",
    },
    "exec_liquidity_failure_risk": {
        "group": "execution",
        "type": "binary",
        "family": ["btc", "weather"],
        "formula": "1 if fill_size < 80% of intended OR fill_latency_ms > 30000",
        "source": "execution_feedback.csv",
        "requires": ["actual_filled", "fill_latency_ms"],
        "min_reliability": 1.0,
        "missing_flag": "exec_cost_unavailable",
    },
}

# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def labels_for_group(group: str) -> list[str]:
    """Return all label names for a given group ('alpha'|'path'|'execution')."""
    return [k for k, v in LABEL_REGISTRY.items() if v.get("group") == group]


def labels_for_family(family: str) -> list[str]:
    """Return all label names applicable to a market family."""
    fam = family.lower()
    result = []
    for k, v in LABEL_REGISTRY.items():
        families = v.get("family", [])
        if any(fam.startswith(f) for f in families):
            result.append(k)
    return result


def alpha_labels_for_family(family: str) -> list[str]:
    """Return alpha-group labels for a market family."""
    return [k for k in labels_for_family(family) if LABEL_REGISTRY[k].get("group") == "alpha"]


def path_labels_for_family(family: str) -> list[str]:
    """Return path-group labels for a market family."""
    return [k for k in labels_for_family(family) if LABEL_REGISTRY[k].get("group") == "path"]


def primary_classifier_target(family: str) -> str | None:
    """
    Return the preferred primary classifier target for a family.

    BTC  → 'path_tp_hit_60m'
    Weather → 'weather_contract_resolved_yes'
    Other → None
    """
    fam = family.lower()
    if fam.startswith("btc"):
        return "path_tp_hit_60m"
    if fam.startswith("weather"):
        return "weather_contract_resolved_yes"
    return None


def primary_regressor_target(family: str) -> str:
    """Return the preferred primary regressor target for a family."""
    return "path_return_15m"


def missing_flags() -> set[str]:
    """Return all missing-data flag column names."""
    return {
        v["missing_flag"]
        for v in LABEL_REGISTRY.values()
        if v.get("missing_flag")
    }
