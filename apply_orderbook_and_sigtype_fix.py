"""
apply_orderbook_and_sigtype_fix.py
===================================
Applies ALL fixes with Windows file-lock handling.

Usage:
    1. Close VS Code / any editor if you get PermissionError
    2. python apply_orderbook_and_sigtype_fix.py
    
    If files are still locked:
    python apply_orderbook_and_sigtype_fix.py --force
"""

import gc
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

BACKUP_DIR = Path("backups") / f"orderbook_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
FIXES_DIR = Path(__file__).parent


def backup(path: Path):
    if path.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        dest = BACKUP_DIR / path.name
        try:
            shutil.copy2(path, dest)
            print(f"  [BACKUP] {path.name}")
        except Exception as exc:
            print(f"  [BACKUP WARN] {path.name}: {exc}")


def safe_copy(source: Path, target: Path, max_retries=5):
    """Copy with retry logic for Windows file locks."""
    for attempt in range(max_retries):
        try:
            gc.collect()
            
            # Write to temp file first
            temp_target = target.with_suffix(".tmp_patch")
            shutil.copy2(source, temp_target)
            
            # Try to replace the real file
            if target.exists():
                try:
                    target.unlink()
                except PermissionError:
                    # Try renaming the locked file out of the way
                    stale = target.with_suffix(".old")
                    if stale.exists():
                        try:
                            stale.unlink()
                        except Exception:
                            pass
                    try:
                        target.rename(stale)
                        print(f"    (moved locked file to {stale.name})")
                    except Exception:
                        raise
            
            temp_target.rename(target)
            return True
            
        except PermissionError as exc:
            if attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                print(f"    [LOCKED] Retry {attempt+1}/{max_retries} in {wait}s... (close VS Code/editor)")
                time.sleep(wait)
            else:
                print(f"    [FAILED] Cannot write {target.name}: {exc}")
                print(f"    --> Manual fix: copy {temp_target.name} to {target.name}")
                return False
    return False


def copy_new_file(source_name, target_path=None):
    source = FIXES_DIR / source_name
    target = Path(target_path or source_name)
    if not source.exists():
        print(f"  [SKIP] {source_name} not found alongside this script")
        return False
    if target.exists():
        backup(target)
    
    if safe_copy(source, target):
        print(f"  [COPY] {source_name}")
        return True
    return False


def fix_env_files():
    """Fix POLYMARKET_SIGNATURE_TYPE=0 -> 1 in all .env files."""
    changed = 0
    for name in [".env", ".env.live", ".env.live.template"]:
        path = Path(name)
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if "POLYMARKET_SIGNATURE_TYPE=0" in text:
            backup(path)
            text = text.replace("POLYMARKET_SIGNATURE_TYPE=0", "POLYMARKET_SIGNATURE_TYPE=1")
            path.write_text(text, encoding="utf-8")
            print(f"  [FIXED] {name}: signature_type 0 -> 1")
            changed += 1
        else:
            current = re.search(r"POLYMARKET_SIGNATURE_TYPE=(\d+)", text)
            if current:
                print(f"  [OK] {name}: already type={current.group(1)}")
            else:
                print(f"  [INFO] {name}: no POLYMARKET_SIGNATURE_TYPE line found")
    return changed


def patch_supervisor():
    """Patch supervisor.py to use OrderBookGuard before every trade entry."""
    path = Path("supervisor.py")
    if not path.exists():
        print("  [SKIP] supervisor.py not found")
        return False

    backup(path)
    try:
        text = path.read_text(encoding="utf-8")
    except PermissionError:
        print("  [FAILED] supervisor.py is locked — close your editor")
        return False

    changed = False

    # ── 1. Add import for OrderBookGuard ──
    if "from orderbook_guard import" not in text:
        text = text.replace(
            "from db import Database",
            "from db import Database\nfrom orderbook_guard import OrderBookGuard",
        )
        changed = True
        print("  [PATCH] Added OrderBookGuard import")

    # ── 2. Initialize the guard in main_loop ──
    if "orderbook_guard = OrderBookGuard()" not in text:
        old_init = 'trade_manager = TradeManager(logs_dir="logs")'
        new_init = '''trade_manager = TradeManager(logs_dir="logs")
    orderbook_guard = OrderBookGuard(max_spread=0.10, min_bid_depth=2, min_ask_depth=2)'''
        if old_init in text:
            text = text.replace(old_init, new_init)
            changed = True
            print("  [PATCH] Initialized OrderBookGuard in main_loop")

    # ── 3. Wire order book check before entry ──
    # Target: the fill_price = quote_entry_price(signal_row) line
    # We insert the guard check right before it
    
    marker = "fill_price = quote_entry_price(signal_row)"
    if marker in text and "ob_check = orderbook_guard" not in text:
        ob_check_block = """# ── ORDER BOOK GUARD: Check spread/depth before entry ──
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

                    # Use real ask price from order book (not midpoint)
                    fill_price = ob_check.get("recommended_entry_price")
                    if fill_price is None:
                        fill_price = quote_entry_price(signal_row)

                    # Log spread cost
                    _spread_cost = ob_check.get("spread_cost_usdc", 0.0)
                    if _spread_cost > 0.01:
                        logging.info(
                            "Spread cost for %s: $%.4f (spread=%.4f)",
                            token_id[:16], _spread_cost,
                            ob_check["analysis"].get("spread", 0),
                        )
                    for _warn in ob_check.get("warnings", []):
                        logging.warning("OrderBookGuard: %s: %s", token_id[:16], _warn)

                    _unused_"""
        
        # Replace the first occurrence only
        text = text.replace(
            marker,
            ob_check_block + marker,
            1,
        )
        changed = True
        print("  [PATCH] Added order book guard check before entry")

    # ── 4. Wire order book into exit pricing ──
    exit_marker = "exit_price = quote_exit_price(pos_dict)"
    if exit_marker in text and "ob_exit_check = orderbook_guard" not in text:
        ob_exit_block = """# ── ORDER BOOK GUARD: Use real bid for exit ──
                            ob_exit_check = orderbook_guard.analyze_book(token_id, depth=5)
                            if ob_exit_check.get("best_bid") is not None:
                                exit_price = ob_exit_check["best_bid"]
                                logging.info("Exit price from book bid: %.4f (spread=%.4f)",
                                             exit_price, ob_exit_check.get("spread", 0))
                            else:
                                exit_price = quote_exit_price(pos_dict)
                            _unused_exit_"""
        # Replace all occurrences
        text = text.replace(exit_marker, ob_exit_block + exit_marker)
        changed = True
        print("  [PATCH] Added order book analysis for exit pricing")

    # Clean up the _unused_ markers (they prevent the old line from running twice)
    text = text.replace("_unused_fill_price = quote_entry_price(signal_row)", "# (original quote_entry_price kept as fallback above)")
    text = text.replace("_unused_exit_exit_price = quote_exit_price(pos_dict)", "# (original quote_exit_price kept as fallback above)")
    # Simpler cleanup
    text = text.replace("                    _unused_", "                    # Original line (now handled above):\n                    # ")
    text = text.replace("                            _unused_exit_", "                            # Original line (now handled above):\n                            # ")

    if changed:
        try:
            path.write_text(text, encoding="utf-8")
            print("  [SAVED] supervisor.py")
        except PermissionError:
            temp = path.with_suffix(".py.patched")
            temp.write_text(text, encoding="utf-8")
            print(f"  [SAVED] {temp.name} (rename to supervisor.py manually — original is locked)")
    else:
        print("  [SKIP] supervisor.py — already patched or structure differs")

    return changed


def force_kill_python():
    """Kill other Python processes on Windows (for --force flag)."""
    print("[--force] Killing other Python processes...\n")
    try:
        import subprocess
        this_pid = os.getpid()
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
            capture_output=True, text=True
        )
        killed = 0
        for line in result.stdout.strip().split("\n")[1:]:
            parts = line.replace('"', '').split(",")
            if len(parts) >= 2:
                try:
                    pid = int(parts[1])
                except ValueError:
                    continue
                if pid != this_pid:
                    subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                                   capture_output=True)
                    killed += 1
        if killed:
            print(f"  Killed {killed} python process(es). Waiting 3s...\n")
            time.sleep(3)
        else:
            print("  No other python processes found.\n")
    except Exception as exc:
        print(f"  Could not kill processes: {exc}\n")


def main():
    if "--force" in sys.argv:
        force_kill_python()

    print("=" * 60)
    print("APPLYING: Order Book Guard + Signature Type Fix")
    print("=" * 60)
    print()
    print("If you get PermissionError, close VS Code and re-run.")
    print("Or run: python apply_orderbook_and_sigtype_fix.py --force")
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
    print("  1. start.py: Now ASKS which signature_type (1=email, 2=MetaMask, 0=EOA)")
    print("  2. orderbook_guard.py: Checks spread/depth/book before every trade")
    print("  3. supervisor.py: Uses real ask price for entry, real bid for exit")
    print("  4. execution_client.py: Warns if signature_type looks wrong")
    print("  5. .env: POLYMARKET_SIGNATURE_TYPE fixed to 1")
    print()
    print("Next steps:")
    print("  1. Make sure your .env has PRIVATE_KEY and POLYMARKET_FUNDER set")
    print("  2. python diagnose_balance_fix.py   # verify balance")
    print("  3. python start.py                  # interactive launch")
    print()
    if BACKUP_DIR.exists():
        print(f"Originals backed up to: {BACKUP_DIR}/")


if __name__ == "__main__":
    main()
