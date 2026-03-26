"""
apply_dashboard_fixes.py
========================
Copies all dashboard bug fixes into place.

Usage:
    python apply_dashboard_fixes.py

Fixes applied:
  1. dashboard_auth.py — shared auth utility (conditional load_dotenv, cached client)
  2. dashboard.py — main dashboard with all 10 bugs fixed
  3. pages/1_Account_Profile.py — auth flow + paper mode support
  4. pages/2_Polymarket_Portfolio.py — no module-level auth, deferred client
  5. start.py — dotenv patching before imports, dashboard launch option
  6. run_bot_and_dashboard.py — explicit env propagation
"""

import shutil
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


def backup_and_copy():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Backing up originals to {BACKUP_DIR}/\n")

    success = 0
    for filename, description in FILES.items():
        source = FIXES_DIR / filename
        target = PROJECT_ROOT / filename
        backup = BACKUP_DIR / filename

        if not source.exists():
            print(f"  [SKIP] {filename} — fix file not found at {source}")
            continue

        if target.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            print(f"  [BACKUP] {filename}")
        else:
            print(f"  [NEW]    {filename}")

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"  [PATCH]  {filename} — {description}")
        success += 1
        print()

    return success


def print_summary(count):
    print("=" * 60)
    print(f"APPLIED {count}/{len(FILES)} DASHBOARD FIXES")
    print("=" * 60)
    print()
    print("What changed:")
    print()
    print("  1. AUTH FLOW (Bugs 1, 2, 3, 5, 7)")
    print("     - load_dotenv() is now conditional: skipped when _INTERACTIVE_MODE=1")
    print("     - ExecutionClient is cached with @st.cache_resource")
    print("     - Portfolio pages no longer crash on import")
    print("     - start.py patches dotenv BEFORE importing bot modules")
    print()
    print("  2. DATA WIRING (Bugs 4, 6, 9, 10)")
    print("     - Schema normalization applied consistently")
    print("     - Empty DataFrame guards on .max(), .mean(), etc.")
    print("     - Auto-refresh shows warning when package missing")
    print("     - File freshness handles missing files gracefully")
    print()
    print("  3. UX IMPROVEMENTS (Bug 8)")
    print("     - Account profile works in paper mode (shows available data)")
    print("     - Auth mode indicator in sidebar")
    print("     - Interactive vs .env credential source shown")
    print()
    print("To test:")
    print("  python start.py          # interactive auth + bot + dashboard")
    print("  streamlit run dashboard.py  # standalone dashboard (uses .env)")
    print()
    print(f"To restore originals: copy from {BACKUP_DIR}/")


if __name__ == "__main__":
    count = backup_and_copy()
    print_summary(count)
