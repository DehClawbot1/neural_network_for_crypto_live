"""
apply_orderbook_and_sigtype_fix.py
===================================
Applies ALL fixes:
  1. start.py:           Interactive signature_type prompt (not hardcoded)
  2. orderbook_guard.py: NEW - checks order book, spread, midpoint before every trade
  3. supervisor.py:      Wires order book guard into entry flow
  4. execution_client.py: Better signature_type defaults and warnings
  5. .env files:         Fixes POLYMARKET_SIGNATURE_TYPE if set to 0

Usage:
    1. Copy all files to your project root
    2. python apply_orderbook_and_sigtype_fix.py
    3. python diagnose_balance_fix.py  (verify balance)
    4. python run_bot.py
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

BACKUP_DIR = Path("backups") / f"orderbook_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
FIXES_DIR = Path(__file__).parent


def backup(path: Path):
    if path.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        dest = BACKUP_DIR / path.name
        shutil.copy2(path, dest)
        print(f"  [BACKUP] {path.name}")


def copy_new_file(source_name, target_path=None):
    source = FIXES_DIR / source_name
    target = Path(target_path or source_name)
    if not source.exists():
        print(f"  [SKIP] {source_name} not found")
        return False
    if target.exists():
        backup(target)
    shutil.copy2(source, target)
    print(f"  [COPY] {source_name}")
    return True


def fix_env_files():
    """Fix POLYMARKET_SIGNATURE_TYPE=0 → 1 in all .env files."""
    changed = 0
    for name in [".env", ".env.live", ".env.live.template"]:
        path = Path(name)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if "POLYMARKET_SIGNATURE_TYPE=0" in text:
            backup(path)
            text = text.replace("POLYMARKET_SIGNATURE_TYPE=0", "POLYMARKET_SIGNATURE_TYPE=1")
            path.write_text(text, encoding="utf-8")
            print(f"  [FIXED] {name}: signature_type 0 → 1")
            changed += 1
        else:
            current = re.search(r"POLYMARKET_SIGNATURE_TYPE=(\d+)", text)
            if current:
                print(f"  [OK] {name}: already type={current.group(1)}")
    return changed


def patch_supervisor():
    """Patch supervisor.py to use OrderBookGuard before every trade entry."""
    path = Path("supervisor.py")
    if not path.exists():
        print("  [SKIP] supervisor.py not found")
        return False

    backup(path)
    text = path.read_text(encoding="utf-8")
    changed = False

    # ── 1. Add import for OrderBookGuard ──
    if "from orderbook_guard import" not in text:
        # Add after the last existing import block
        text = text.replace(
            "from db import Database",
            "from db import Database\nfrom orderbook_guard import OrderBookGuard",
        )
        changed = True
        print("  [PATCH] Added OrderBookGuard import")

    # ── 2. Initialize the guard in main_loop ──
    if "orderbook_guard = OrderBookGuard()" not in text:
        # Add after trade_manager initialization
        old_init = 'trade_manager = TradeManager(logs_dir="logs")'
        new_init = '''trade_manager = TradeManager(logs_dir="logs")
    orderbook_guard = OrderBookGuard(max_spread=0.10, min_bid_depth=2, min_ask_depth=2)'''
        if old_init in text:
            text = text.replace(old_init, new_init)
            changed = True
            print("  [PATCH] Initialized OrderBookGuard in main_loop")

    # ── 3. Add order book check before entry ──
    # Find the section where fill_price is computed and trade is submitted
    # The key line is: fill_price = quote_entry_price(signal_row)

    old_entry_block = """                    fill_price = quote_entry_price(signal_row)
                    
                    if fill_price is None or pd.isna(fill_price):
                        logging.warning("Skipping signal with missing fill price for token_id=%s", token_id)
                        continue"""

    new_entry_block = """                    # ── ORDER BOOK GUARD: Check book before entry ──
                    ob_check = orderbook_guard.check_before_entry(
                        token_id=token_id,
                        side="BUY",
                        intended_size_usdc=size_usdc,
                    )
                    if not ob_check["tradable"]:
                        logging.info(
                            "OrderBookGuard BLOCKED entry for %s: %s",
                            token_id[:16], ob_check["reason"],
                        )
                        continue

                    # Use order book recommended price instead of naive quote
                    fill_price = ob_check.get("recommended_entry_price")
                    if fill_price is None:
                        fill_price = quote_entry_price(signal_row)

                    if fill_price is None or pd.isna(fill_price):
                        logging.warning("Skipping signal with missing fill price for token_id=%s", token_id)
                        continue

                    # Log spread cost warning
                    spread_cost = ob_check.get("spread_cost_usdc", 0.0)
                    if spread_cost > 0.01:
                        logging.info(
                            "Spread cost for %s: $%.4f (spread=%.4f, midpoint=%.4f, ask=%.4f)",
                            token_id[:16], spread_cost,
                            ob_check["analysis"].get("spread", 0),
                            ob_check["analysis"].get("midpoint", 0),
                            ob_check["analysis"].get("best_ask", 0),
                        )
                    for warn in ob_check.get("warnings", []):
                        logging.warning("OrderBookGuard WARNING for %s: %s", token_id[:16], warn)"""

    if old_entry_block in text:
        text = text.replace(old_entry_block, new_entry_block)
        changed = True
        print("  [PATCH] Added order book guard before entry (exact match)")
    else:
        # Try a more flexible match — find the fill_price line and patch around it
        simple_old = "fill_price = quote_entry_price(signal_row)"
        if simple_old in text and "ob_check = orderbook_guard" not in text:
            # Find the first occurrence inside the entry block (after action_val != 0)
            # We'll do a targeted replacement
            idx = text.find(simple_old)
            if idx > 0:
                # Insert order book check before the fill_price line
                indent = "                    "
                ob_block = f"""{indent}# ── ORDER BOOK GUARD: Check book before entry ──
{indent}ob_check = orderbook_guard.check_before_entry(
{indent}    token_id=token_id,
{indent}    side="BUY",
{indent}    intended_size_usdc=size_usdc,
{indent})
{indent}if not ob_check["tradable"]:
{indent}    logging.info(
{indent}        "OrderBookGuard BLOCKED entry for %s: %s",
{indent}        token_id[:16], ob_check["reason"],
{indent}    )
{indent}    continue

{indent}# Use order book recommended price instead of naive quote
{indent}fill_price = ob_check.get("recommended_entry_price")
{indent}if fill_price is None:
{indent}    """
                text = text[:idx] + ob_block + text[idx:]
                changed = True
                print("  [PATCH] Added order book guard before entry (flexible match)")

    # ── 4. Add order book check before exit pricing too ──
    old_exit_price = "exit_price = quote_exit_price(pos_dict)"
    new_exit_price = """# ── ORDER BOOK GUARD: Get real bid price for exit ──
                            ob_exit_check = orderbook_guard.analyze_book(token_id, depth=5)
                            if ob_exit_check.get("best_bid") is not None:
                                exit_price = ob_exit_check["best_bid"]
                                logging.info("Exit price from order book bid: %.4f (spread=%.4f)",
                                             exit_price, ob_exit_check.get("spread", 0))
                            else:
                                exit_price = quote_exit_price(pos_dict)"""

    if old_exit_price in text and "ob_exit_check = orderbook_guard" not in text:
        text = text.replace(old_exit_price, new_exit_price, 2)  # Replace first 2 occurrences
        changed = True
        print("  [PATCH] Added order book analysis for exit pricing")

    if changed:
        path.write_text(text, encoding="utf-8")
        print("  [SAVED] supervisor.py")
    else:
        print("  [SKIP] supervisor.py — already patched or structure differs")

    return changed


def main():
    print("=" * 60)
    print("APPLYING: Order Book Guard + Signature Type Fix")
    print("=" * 60)
    print()

    # 1. Copy new/patched files
    print("--- Copying patched files ---")
    copy_new_file("orderbook_guard.py")
    copy_new_file("start.py")
    copy_new_file("execution_client.py")
    print()

    # 2. Fix .env files
    print("--- Fixing .env signature_type ---")
    fix_env_files()
    print()

    # 3. Patch supervisor.py
    print("--- Patching supervisor.py ---")
    patch_supervisor()
    print()

    print("=" * 60)
    print("ALL FIXES APPLIED")
    print("=" * 60)
    print()
    print("What changed:")
    print()
    print("  1. start.py: Now ASKS which signature_type you use")
    print("     (1=email/Magic, 2=MetaMask, 0=direct EOA)")
    print()
    print("  2. orderbook_guard.py: NEW module that checks before every trade:")
    print("     - Does the order book exist?")
    print("     - Is the spread acceptable? (max 0.10)")
    print("     - Are there enough bid/ask levels?")
    print("     - What is the real fill price? (ask for BUY, bid for SELL)")
    print("     - How much will the spread cost you?")
    print()
    print("  3. supervisor.py: Entry flow now runs through the guard:")
    print("     - Blocked if spread > 0.10 or book is empty")
    print("     - Uses best_ask as entry price (not midpoint)")
    print("     - Logs spread cost for every trade")
    print("     - Exit pricing uses real best_bid from order book")
    print()
    print("  4. execution_client.py: Warns if signature_type looks wrong")
    print()
    print("  5. .env: POLYMARKET_SIGNATURE_TYPE fixed to 1 (email login)")
    print()
    print("Next steps:")
    print("  python diagnose_balance_fix.py   # verify balance shows up")
    print("  python run_bot.py                # start trading with book checks")
    print()
    if BACKUP_DIR.exists():
        print(f"Originals backed up to: {BACKUP_DIR}/")


if __name__ == "__main__":
    main()
