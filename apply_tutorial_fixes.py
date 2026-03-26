"""
apply_tutorial_fixes.py
========================
Patches the codebase to align with the official Polymarket Python tutorial.

Usage:
    1. CLOSE your IDE / stop run_bot.py / stop any Python processes first
    2. python apply_tutorial_fixes.py

If you still get PermissionError, run:
    python apply_tutorial_fixes.py --force
"""

import os
import sys
import shutil
import time
import gc
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(".")
BACKUP_DIR = PROJECT_ROOT / "backups" / f"tutorial_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

FILES_TO_PATCH = {
    "execution_client.py": "Auth flow: derive_api_key(), USDC normalization, order book methods",
    "market_monitor.py": "Market discovery: clobTokenIds, volume sorting, outcomePrices",
    "market_price_service.py": "Order book analysis: sorted bids/asks, depth, imbalance",
    "price_tracker.py": "NEW: Tutorial BONUS utilities (price tracking, positions)",
}

FIXES_DIR = Path(__file__).parent


def safe_copy(source, target, max_retries=5):
    """Copy with retry logic for Windows file locks."""
    for attempt in range(max_retries):
        try:
            gc.collect()

            temp_target = target.with_suffix(".tmp_patch")
            shutil.copy2(source, temp_target)

            if target.exists():
                try:
                    target.unlink()
                except PermissionError:
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
                print(f"    [LOCKED] Retry {attempt+1}/{max_retries} in {wait}s... (close IDE/bot)")
                time.sleep(wait)
            else:
                print(f"    [FAILED] Cannot write {target.name}: {exc}")
                print(f"    --> Manual fix: copy the .tmp_patch file yourself")
                return False
    return False


def show_tips():
    print("Before running, close these if files are locked:")
    print("  - VS Code / PyCharm / any editor with the project open")
    print("  - Any running python.exe (run_bot.py, streamlit, etc.)")
    print("  - Task Manager > Details > python.exe > End Task")
    print()


def force_kill_python():
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


def backup_and_copy():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Backing up originals to {BACKUP_DIR}/\n")

    success_count = 0
    fail_count = 0

    for filename, description in FILES_TO_PATCH.items():
        source = FIXES_DIR / filename
        target = PROJECT_ROOT / filename
        backup = BACKUP_DIR / filename

        if not source.exists():
            print(f"  [SKIP] {filename} -- fix file not found")
            continue

        if target.exists():
            try:
                shutil.copy2(target, backup)
                print(f"  [BACKUP] {filename}")
            except Exception as exc:
                print(f"  [BACKUP WARN] {filename}: {exc}")
        else:
            print(f"  [NEW]    {filename}")

        if safe_copy(source, target):
            print(f"  [PATCHED] {filename} -- {description}")
            success_count += 1
        else:
            fail_count += 1
        print()

    return success_count, fail_count


def print_summary(success, failed):
    print("=" * 60)
    if failed == 0:
        print(f"ALL {success} FIXES APPLIED SUCCESSFULLY")
    else:
        print(f"APPLIED {success}/{success+failed} FIXES ({failed} FAILED)")
    print("=" * 60)
    print()

    if failed > 0:
        print("For failed files, either:")
        print("  1. Close all editors/processes and re-run this script")
        print("  2. Run: python apply_tutorial_fixes.py --force")
        print("  3. Manually rename .tmp_patch files to their real names")
        print()
        return

    print("Key changes:")
    print("  - execution_client.py: derive_api_key() auth + USDC normalization")
    print("  - market_monitor.py: clobTokenIds parsing + volume sorting")
    print("  - market_price_service.py: order book depth analysis")
    print("  - price_tracker.py: real-time price + position tracking")
    print()
    print("Restart run_bot.py to apply all changes.")
    print(f"\nTo restore originals: copy from {BACKUP_DIR}/")


if __name__ == "__main__":
    show_tips()

    if "--force" in sys.argv:
        force_kill_python()

    success, failed = backup_and_copy()
    print_summary(success, failed)
