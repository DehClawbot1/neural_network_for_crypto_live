"""
apply_dashboard_fixes.py
========================
Copies all dashboard bug fixes into place.

Usage:
    1. CLOSE Streamlit dashboard if running
    2. python apply_dashboard_fixes.py

If files are locked:
    python apply_dashboard_fixes.py --force
"""

import gc
import os
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(".")
BACKUP_DIR = PROJECT_ROOT / "backups" / f"dashboard_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
FIXES_DIR = Path(__file__).parent

FILES = {
    "dashboard_auth.py": "NEW: shared auth utility for all dashboard pages",
    "dashboard.py": "FIX: conditional dotenv, cached client, 10 bugs fixed",
    "pages/1_Account_Profile.py": "FIX: auth flow, paper mode support, cached client",
    "pages/2_Polymarket_Portfolio.py": "FIX: no module-level auth crash, deferred client",
    "start.py": "FIX: dotenv patching before imports, dashboard launch option",
    "run_bot_and_dashboard.py": "FIX: explicit env propagation to subprocesses",
}


def show_tips():
    print("Before running, close these if files are locked:")
    print("  - Streamlit dashboard (streamlit run dashboard.py)")
    print("  - VS Code / PyCharm / any editor with the project open")
    print("  - Any running python.exe (run_bot.py, etc.)")
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
                print(f"    [LOCKED] Retry {attempt+1}/{max_retries} in {wait}s... (close Streamlit/IDE)")
                time.sleep(wait)
            else:
                print(f"    [FAILED] Cannot write {target.name}: {exc}")
                print(f"    --> Manual fix: rename {temp_target.name} to {target.name}")
                return False
    return False


def backup_and_copy():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Backing up originals to {BACKUP_DIR}/\n")

    success = 0
    failed = 0

    for filename, description in FILES.items():
        source = FIXES_DIR / filename
        target = PROJECT_ROOT / filename
        backup = BACKUP_DIR / filename

        if not source.exists():
            print(f"  [SKIP] {filename} -- fix file not found at {source}")
            continue

        if target.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(target, backup)
                print(f"  [BACKUP] {filename}")
            except Exception as exc:
                print(f"  [BACKUP WARN] {filename}: {exc}")
        else:
            print(f"  [NEW]    {filename}")

        target.parent.mkdir(parents=True, exist_ok=True)

        if safe_copy(source, target):
            print(f"  [PATCHED] {filename} -- {description}")
            success += 1
        else:
            failed += 1
        print()

    return success, failed


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
        print("  1. Close all editors/Streamlit and re-run this script")
        print("  2. Run: python apply_dashboard_fixes.py --force")
        print("  3. Manually rename .tmp_patch files to their real names")
        print()
        return

    print("What changed:")
    print()
    print("  1. AUTH FLOW (Bugs 1, 2, 3, 5, 7)")
    print("     - load_dotenv() skipped when _INTERACTIVE_MODE=1")
    print("     - ExecutionClient cached with @st.cache_resource")
    print("     - Portfolio pages no longer crash on import")
    print("     - start.py patches dotenv BEFORE importing bot modules")
    print()
    print("  2. DATA WIRING (Bugs 4, 6, 9, 10)")
    print("     - Empty DataFrame guards on .max(), .mean()")
    print("     - Auto-refresh warning when package missing")
    print("     - File freshness handles missing files")
    print()
    print("  3. UX (Bug 8)")
    print("     - Account profile works in paper mode")
    print("     - Auth mode indicator in sidebar")
    print()
    print("To test:")
    print("  python start.py             # interactive auth + bot + dashboard")
    print("  streamlit run dashboard.py   # standalone (uses .env)")
    print()
    print(f"To restore originals: copy from {BACKUP_DIR}/")


if __name__ == "__main__":
    show_tips()

    if "--force" in sys.argv:
        force_kill_python()

    success, failed = backup_and_copy()
    print_summary(success, failed)
