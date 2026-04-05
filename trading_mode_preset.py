"""
trading_mode_preset.py
======================
Startup trading-mode selector (1–4) that tunes risk, sizing, governor,
and entry parameters in one shot.

Mode 1 - Very Aggressive   : max capital deployment, loose guards
Mode 2 - Aggressive        : high capital deployment, moderate guards
Mode 3 - Conservative      : reduced sizing, tighter guards
Mode 4 - Very Conservative : minimal sizing, strictest guards
"""

import logging
import os

from config import TradingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------
# Each preset is a dict of:
#   "config"  → attrs to patch on TradingConfig (class-level)
#   "env"     → env vars to set (only if not already overridden by user)
# ---------------------------------------------------------------------------

PRESETS = {
    1: {
        "name": "Very Aggressive",
        "config": {
            "MAX_RISK_PER_TRADE_PCT": 0.25,
            "CAPITAL_RESERVE_PCT": 0.02,
            "HIGH_CONFIDENCE_BET_PCT": 0.22,
            "MEDIUM_CONFIDENCE_BET_PCT": 0.15,
            "LOW_CONFIDENCE_BET_PCT": 0.08,
            "HARD_MAX_BET_USDC": 500.0,
            "MAX_CONCURRENT_POSITIONS": 30,
            "MIN_ENTRY_USDC": 2.00,
        },
        "env": {
            "ENTRY_MIN_SCORE": "0.06",
            "ENTRY_MAX_SPREAD": "0.45",
            "ENTRY_MIN_LIQUIDITY": "0.3",
            "ENTRY_MIN_LIQUIDITY_SCORE": "0.002",
            "ENTRY_AGGRESSION_TOP_K": "6",
            "TARGET_ENTRY_INTERVAL_MINUTES": "3",
            "ALWAYS_ON_SIGNAL_SIZE": "50",
            "SESSION_MAX_DRAWDOWN_USDC": "100.0",
            "SESSION_MAX_FAILED_ENTRIES": "25",
            # Governor Level 1 — very loose
            "GOV_LEVEL1_MIN_WIN_RATE": "0.28",
            "GOV_LEVEL1_MIN_PROFIT_FACTOR": "0.50",
            "GOV_LEVEL1_MAX_NEGATIVE_AVG_PNL": "-0.20",
            "GOV_LEVEL1_MAX_DRAWDOWN": "60",
            "GOV_LEVEL1_SIZE_MULTIPLIER": "0.70",
            "GOV_LEVEL1_MIN_ENTRY_CONFIDENCE": "0.40",
            "GOV_LEVEL1_MIN_LIQUIDITY_SCORE": "0.20",
            # Governor Level 2 — still loose
            "GOV_LEVEL2_MIN_WIN_RATE": "0.20",
            "GOV_LEVEL2_MIN_PROFIT_FACTOR": "0.40",
            "GOV_LEVEL2_MAX_NEGATIVE_AVG_PNL": "-0.40",
            "GOV_LEVEL2_MAX_DRAWDOWN": "100",
            "GOV_LEVEL2_SIZE_MULTIPLIER": "0.50",
            "GOV_LEVEL2_MIN_ENTRY_CONFIDENCE": "0.30",
            "GOV_LEVEL2_MIN_LIQUIDITY_SCORE": "0.30",
        },
    },
    2: {
        "name": "Aggressive",
        "config": {
            "MAX_RISK_PER_TRADE_PCT": 0.18,
            "CAPITAL_RESERVE_PCT": 0.05,
            "HIGH_CONFIDENCE_BET_PCT": 0.18,
            "MEDIUM_CONFIDENCE_BET_PCT": 0.12,
            "LOW_CONFIDENCE_BET_PCT": 0.06,
            "HARD_MAX_BET_USDC": 350.0,
            "MAX_CONCURRENT_POSITIONS": 25,
            "MIN_ENTRY_USDC": 2.50,
        },
        "env": {
            "ENTRY_MIN_SCORE": "0.10",
            "ENTRY_MAX_SPREAD": "0.38",
            "ENTRY_MIN_LIQUIDITY": "0.4",
            "ENTRY_MIN_LIQUIDITY_SCORE": "0.004",
            "ENTRY_AGGRESSION_TOP_K": "4",
            "TARGET_ENTRY_INTERVAL_MINUTES": "4",
            "ALWAYS_ON_SIGNAL_SIZE": "35",
            "SESSION_MAX_DRAWDOWN_USDC": "70.0",
            "SESSION_MAX_FAILED_ENTRIES": "20",
            # Governor Level 1
            "GOV_LEVEL1_MIN_WIN_RATE": "0.35",
            "GOV_LEVEL1_MIN_PROFIT_FACTOR": "0.70",
            "GOV_LEVEL1_MAX_NEGATIVE_AVG_PNL": "-0.12",
            "GOV_LEVEL1_MAX_DRAWDOWN": "40",
            "GOV_LEVEL1_SIZE_MULTIPLIER": "0.60",
            "GOV_LEVEL1_MIN_ENTRY_CONFIDENCE": "0.50",
            "GOV_LEVEL1_MIN_LIQUIDITY_SCORE": "0.30",
            # Governor Level 2
            "GOV_LEVEL2_MIN_WIN_RATE": "0.25",
            "GOV_LEVEL2_MIN_PROFIT_FACTOR": "0.50",
            "GOV_LEVEL2_MAX_NEGATIVE_AVG_PNL": "-0.30",
            "GOV_LEVEL2_MAX_DRAWDOWN": "70",
            "GOV_LEVEL2_SIZE_MULTIPLIER": "0.40",
            "GOV_LEVEL2_MIN_ENTRY_CONFIDENCE": "0.35",
            "GOV_LEVEL2_MIN_LIQUIDITY_SCORE": "0.40",
        },
    },
    3: {
        "name": "Conservative",
        "config": {
            "MAX_RISK_PER_TRADE_PCT": 0.10,
            "CAPITAL_RESERVE_PCT": 0.10,
            "HIGH_CONFIDENCE_BET_PCT": 0.10,
            "MEDIUM_CONFIDENCE_BET_PCT": 0.06,
            "LOW_CONFIDENCE_BET_PCT": 0.03,
            "HARD_MAX_BET_USDC": 150.0,
            "MAX_CONCURRENT_POSITIONS": 15,
            "MIN_ENTRY_USDC": 3.00,
        },
        "env": {
            "ENTRY_MIN_SCORE": "0.18",
            "ENTRY_MAX_SPREAD": "0.25",
            "ENTRY_MIN_LIQUIDITY": "0.8",
            "ENTRY_MIN_LIQUIDITY_SCORE": "0.01",
            "ENTRY_AGGRESSION_TOP_K": "2",
            "TARGET_ENTRY_INTERVAL_MINUTES": "8",
            "ALWAYS_ON_SIGNAL_SIZE": "15",
            "SESSION_MAX_DRAWDOWN_USDC": "30.0",
            "SESSION_MAX_FAILED_ENTRIES": "10",
            # Governor Level 1 — tight
            "GOV_LEVEL1_MIN_WIN_RATE": "0.42",
            "GOV_LEVEL1_MIN_PROFIT_FACTOR": "0.90",
            "GOV_LEVEL1_MAX_NEGATIVE_AVG_PNL": "-0.06",
            "GOV_LEVEL1_MAX_DRAWDOWN": "20",
            "GOV_LEVEL1_SIZE_MULTIPLIER": "0.40",
            "GOV_LEVEL1_MIN_ENTRY_CONFIDENCE": "0.70",
            "GOV_LEVEL1_MIN_LIQUIDITY_SCORE": "0.50",
            # Governor Level 2 — strict
            "GOV_LEVEL2_MIN_WIN_RATE": "0.35",
            "GOV_LEVEL2_MIN_PROFIT_FACTOR": "0.70",
            "GOV_LEVEL2_MAX_NEGATIVE_AVG_PNL": "-0.15",
            "GOV_LEVEL2_MAX_DRAWDOWN": "40",
            "GOV_LEVEL2_SIZE_MULTIPLIER": "0.25",
            "GOV_LEVEL2_MIN_ENTRY_CONFIDENCE": "0.65",
            "GOV_LEVEL2_MIN_LIQUIDITY_SCORE": "0.60",
        },
    },
    4: {
        "name": "Very Conservative",
        "config": {
            "MAX_RISK_PER_TRADE_PCT": 0.06,
            "CAPITAL_RESERVE_PCT": 0.15,
            "HIGH_CONFIDENCE_BET_PCT": 0.06,
            "MEDIUM_CONFIDENCE_BET_PCT": 0.04,
            "LOW_CONFIDENCE_BET_PCT": 0.02,
            "HARD_MAX_BET_USDC": 80.0,
            "MAX_CONCURRENT_POSITIONS": 8,
            "MIN_ENTRY_USDC": 5.00,
        },
        "env": {
            "ENTRY_MIN_SCORE": "0.25",
            "ENTRY_MAX_SPREAD": "0.15",
            "ENTRY_MIN_LIQUIDITY": "1.5",
            "ENTRY_MIN_LIQUIDITY_SCORE": "0.02",
            "ENTRY_AGGRESSION_TOP_K": "1",
            "TARGET_ENTRY_INTERVAL_MINUTES": "12",
            "ALWAYS_ON_SIGNAL_SIZE": "10",
            "SESSION_MAX_DRAWDOWN_USDC": "15.0",
            "SESSION_MAX_FAILED_ENTRIES": "5",
            # Governor Level 1 — very tight
            "GOV_LEVEL1_MIN_WIN_RATE": "0.48",
            "GOV_LEVEL1_MIN_PROFIT_FACTOR": "1.00",
            "GOV_LEVEL1_MAX_NEGATIVE_AVG_PNL": "-0.04",
            "GOV_LEVEL1_MAX_DRAWDOWN": "12",
            "GOV_LEVEL1_SIZE_MULTIPLIER": "0.30",
            "GOV_LEVEL1_MIN_ENTRY_CONFIDENCE": "0.80",
            "GOV_LEVEL1_MIN_LIQUIDITY_SCORE": "0.60",
            # Governor Level 2 — strictest
            "GOV_LEVEL2_MIN_WIN_RATE": "0.40",
            "GOV_LEVEL2_MIN_PROFIT_FACTOR": "0.85",
            "GOV_LEVEL2_MAX_NEGATIVE_AVG_PNL": "-0.08",
            "GOV_LEVEL2_MAX_DRAWDOWN": "25",
            "GOV_LEVEL2_SIZE_MULTIPLIER": "0.15",
            "GOV_LEVEL2_MIN_ENTRY_CONFIDENCE": "0.82",
            "GOV_LEVEL2_MIN_LIQUIDITY_SCORE": "0.70",
        },
    },
}


def _print_banner():
    """Print the mode selection menu."""
    print("\n" + "=" * 60)
    print("       POLYMARKET BOT - TRADING MODE SELECTION")
    print("=" * 60)
    print()
    print("  [1] Very Aggressive   - max capital, loose guards")
    print("  [2] Aggressive        - high capital, moderate guards")
    print("  [3] Conservative      - reduced sizing, tight guards")
    print("  [4] Very Conservative - minimal sizing, strictest guards")
    print()


def select_trading_mode(timeout_seconds: int = 30, default_mode: int = 2) -> int:
    """
    Prompt the user to pick a trading mode (1-4) at startup.

    Falls back to *default_mode* if the env var ``TRADING_MODE_PRESET`` is
    already set, or if stdin is unavailable / times out.
    """
    env_preset = os.getenv("TRADING_MODE_PRESET", "").strip()
    if env_preset in {"1", "2", "3", "4"}:
        mode = int(env_preset)
        logger.info("Trading mode preset from env: %d (%s)", mode, PRESETS[mode]["name"])
        return mode

    _print_banner()
    try:
        raw = input(f"  Select mode [1-4] (default={default_mode}): ").strip()
    except (EOFError, KeyboardInterrupt):
        raw = ""
    if raw in {"1", "2", "3", "4"}:
        mode = int(raw)
    else:
        mode = default_mode
        if raw:
            print(f"  Invalid input '{raw}' — using default mode {default_mode}.")
    print(f"\n  >>> Mode {mode}: {PRESETS[mode]['name']} <<<\n")
    return mode


def apply_preset(mode: int) -> dict:
    """
    Apply the chosen preset.  Returns the preset dict for logging.

    - Patches TradingConfig class attributes
    - Sets env vars (only those NOT already overridden by the user's .env)
    """
    preset = PRESETS.get(mode)
    if preset is None:
        logger.warning("Unknown trading mode %s — no preset applied.", mode)
        return {}

    # --- Patch TradingConfig ---
    for attr, value in preset["config"].items():
        setattr(TradingConfig, attr, value)
        logger.debug("TradingConfig.%s = %s", attr, value)

    # --- Set env vars (respect user overrides from .env) ---
    applied_env = {}
    for key, value in preset["env"].items():
        existing = os.getenv(key)
        if existing is None:
            os.environ[key] = value
            applied_env[key] = value
        else:
            applied_env[key] = f"{existing} (user override)"

    logger.info(
        "Applied trading mode %d (%s): config=%d overrides, env=%d vars",
        mode,
        preset["name"],
        len(preset["config"]),
        len(preset["env"]),
    )
    return preset
