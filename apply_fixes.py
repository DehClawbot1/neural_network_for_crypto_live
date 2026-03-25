"""
APPLY ALL FIXES
===============
Run this script from your project root to install all bug fixes.

Usage:
    python apply_fixes.py

This will:
  1. Back up original files to backups/
  2. Copy fixed files into place
  3. Print a summary of what changed

Fixes applied:
  A - RL weights now persist (bootstrap training on first run)
  B - Available USDC balance printed clearly at startup
  C - Confidence + signal_label flow from signals → positions.csv → dashboard
  D - Close reasons flow correctly (not hardcoded "policy_exit")
  E - TradeLifecycle.close() accepts reason parameter
  F - Both realized_pnl AND net_realized_pnl written to closed_positions.csv
  G - All model decisions logged (including IGNOREs)
  H - Consistent ISO timestamp format in positions.csv
  I - All CPU cores used for sklearn/LightGBM training (14 threads on your 5700X)
"""

import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(".")
BACKUP_DIR = PROJECT_ROOT / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
FIXES_DIR = Path(__file__).parent

FILES_TO_PATCH = {
    "trade_lifecycle.py": "BUG FIX C/D/E: close_reason + confidence_at_entry fields",
    "trade_manager.py": "BUG FIX C/D/E/F/H: dashboard wiring + column names",
    "run_bot.py": "BUG FIX A/B: RL persistence + balance display",
    "stage1_models.py": "BUG FIX I: CPU parallelism for LightGBM",
    "supervised_models.py": "BUG FIX I: CPU parallelism for RandomForest",
    "hardware_config.py": "NEW: GPU/CPU detection and optimization",
}


def backup_and_copy():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Backing up originals to {BACKUP_DIR}/\n")

    for filename, description in FILES_TO_PATCH.items():
        source = FIXES_DIR / filename
        target = PROJECT_ROOT / filename
        backup = BACKUP_DIR / filename

        if target.exists():
            shutil.copy2(target, backup)
            print(f"  [BACKUP] {filename} → backups/")
        else:
            print(f"  [NEW]    {filename} (no original to back up)")

        shutil.copy2(source, target)
        print(f"  [PATCH]  {filename} — {description}")
        print()


def print_summary():
    print("=" * 60)
    print("ALL FIXES APPLIED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("What changed:")
    print()
    print("  1. BALANCE DISPLAY (Bug B)")
    print("     run_bot.py now prints a clear table showing:")
    print("     - CLOB/API Balance")
    print("     - On-chain USDC")
    print("     - AVAILABLE TO TRADE (max of both)")
    print()
    print("  2. RL WEIGHTS PERSIST (Bug A)")
    print("     First run bootstraps 1000-step RL weights.")
    print("     Future runs resume from saved weights instead of retraining.")
    print("     Retrain only triggers after 5 new closed trades.")
    print()
    print("  3. DASHBOARD DATA FLOW (Bugs C/D/E/F/H)")
    print("     - confidence + signal_label now flow to positions.csv")
    print("     - Close reasons are actual (take_profit, stop_loss, etc.)")
    print("     - net_realized_pnl column written for dashboard PnL charts")
    print("     - Timestamps use consistent ISO format")
    print()
    print("  4. CPU/GPU OPTIMIZATION (Bug I)")
    print("     - sklearn uses 14 threads (16 - 2 for OS)")
    print("     - LightGBM uses all available cores")
    print("     - OMP/MKL/OpenBLAS thread count set to 8")
    print("     - DirectML probed for AMD GPU acceleration")
    print()
    print("Optional: Install torch-directml for AMD GPU acceleration:")
    print("  pip install torch-directml")
    print()
    print("To restore originals:")
    print(f"  Copy files from {BACKUP_DIR}/ back to project root")
    print()


if __name__ == "__main__":
    backup_and_copy()
    print_summary()
