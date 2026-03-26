"""
apply_tutorial_fixes.py
========================
Patches the codebase to align with the official Polymarket Python tutorial
(https://github.com/RobotTraders/bits_and_bobs/blob/main/polymarket_python.ipynb)

Usage:
    python apply_tutorial_fixes.py

Fixes applied:
  1. execution_client.py — Auth flow aligned with tutorial (derive_api_key),
     USDC balance normalization, tutorial-compatible convenience methods
  2. market_monitor.py — Tutorial-style market discovery, proper clobTokenIds
     parsing, volume24hr sorting, outcomePrices extraction
  3. market_price_service.py — Order book analysis matching tutorial pattern
     (sorted bids/asks, imbalance calculation)
  4. price_tracker.py — NEW: Tutorial BONUS 1 & 2 (real-time price tracking,
     address position tracking, market deep dive)
"""

import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(".")
BACKUP_DIR = PROJECT_ROOT / "backups" / f"tutorial_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

FILES_TO_PATCH = {
    "execution_client.py": {
        "description": "Auth flow: derive_api_key(), USDC normalization, order book methods",
        "fixes": [
            "Uses derive_api_key() as primary credential derivation (tutorial method)",
            "Falls back to create_or_derive_api_creds() for older client versions",
            "Added _normalize_usdc_balance() for proper /1e6 conversion",
            "Added get_order_book(), get_midpoint(), get_price(), get_spread() convenience methods",
            "Added cancel_all() matching tutorial pattern",
            "Fixed get_open_orders() to use OpenOrderParams() (tutorial pattern)",
        ],
    },
    "market_monitor.py": {
        "description": "Market discovery: clobTokenIds, volume sorting, outcomePrices",
        "fixes": [
            "clobTokenIds[0]/[1] preferred over tokens array (tutorial pattern)",
            "Added volume24hr field from Gamma API",
            "Added outcomePrices parsing (tutorial shows this as useful field)",
            "Added fetch_active_markets_by_volume() (tutorial Section 2 pattern)",
            "Better fallback for lastTradePrice from outcomePrices",
        ],
    },
    "market_price_service.py": {
        "description": "Order book analysis: sorted bids/asks, depth, imbalance",
        "fixes": [
            "Added get_order_book_analysis() with tutorial-style bid/ask sorting",
            "Calculates order book imbalance, bid/ask volume",
            "Lazy ClobClient initialization for read-only order book queries",
            "get_midpoint() now tries order book analysis first (more accurate)",
        ],
    },
    "price_tracker.py": {
        "description": "NEW: Tutorial BONUS utilities",
        "fixes": [
            "track_price(): Real-time CLOB midpoint polling (Tutorial BONUS 1)",
            "get_user_positions(): Data API position tracking (Tutorial BONUS 2)",
            "get_market_deep_dive(): Combined order book + price analysis",
        ],
    },
}

FIXES_DIR = Path(__file__).parent


def backup_and_copy():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Backing up originals to {BACKUP_DIR}/\n")

    for filename, info in FILES_TO_PATCH.items():
        source = FIXES_DIR / filename
        target = PROJECT_ROOT / filename
        backup = BACKUP_DIR / filename

        if not source.exists():
            print(f"  [SKIP] {filename} — fix file not found at {source}")
            continue

        if target.exists():
            shutil.copy2(target, backup)
            print(f"  [BACKUP] {filename} → {BACKUP_DIR.name}/")
        else:
            print(f"  [NEW]    {filename} (no original to back up)")

        shutil.copy2(source, target)
        print(f"  [PATCH]  {filename} — {info['description']}")
        for fix in info["fixes"]:
            print(f"           • {fix}")
        print()


def print_summary():
    print("=" * 64)
    print("ALL TUTORIAL-ALIGNED FIXES APPLIED")
    print("=" * 64)
    print()
    print("What changed:")
    print()
    print("  1. AUTHENTICATION (execution_client.py)")
    print("     Tutorial pattern: derive_api_key() → set_api_creds()")
    print("     Now tries derive_api_key() first, falls back gracefully.")
    print("     USDC balance properly normalized (raw int vs float).")
    print()
    print("  2. MARKET DISCOVERY (market_monitor.py)")
    print("     Tutorial pattern: Gamma API with volume24hr sorting")
    print("     Token IDs extracted from clobTokenIds (JSON string).")
    print("     New: fetch_active_markets_by_volume() utility.")
    print()
    print("  3. ORDER BOOK ANALYSIS (market_price_service.py)")
    print("     Tutorial pattern: get_order_book → sorted bids/asks")
    print("     New: Full order book analysis with depth & imbalance.")
    print()
    print("  4. PRICE & POSITION TRACKING (price_tracker.py)")
    print("     Tutorial BONUS 1: Real-time CLOB midpoint polling.")
    print("     Tutorial BONUS 2: Data API wallet position lookup.")
    print("     New: Market deep dive combining all analysis.")
    print()
    print("To restore originals:")
    print(f"  Copy files from {BACKUP_DIR}/ back to project root")
    print()
    print("Restart run_bot.py to apply all changes.")


if __name__ == "__main__":
    backup_and_copy()
    print_summary()
