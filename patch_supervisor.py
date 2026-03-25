#!/usr/bin/env python3
"""
Apply targeted patches to supervisor.py:
  1. TradeManager() → TradeManager(logs_dir="logs")
  2. Add trade_manager.persist_open_positions() after write_status()

Run from repo root:
  python patch_supervisor.py

Then delete this file — it's a one-time patcher.
"""
import sys
from pathlib import Path


def patch():
    path = Path("supervisor.py")
    if not path.exists():
        print("ERROR: supervisor.py not found in current directory.")
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    changes = 0

    # ── Fix 1: TradeManager constructor needs logs_dir ──
    old1 = "trade_manager = TradeManager()"
    new1 = 'trade_manager = TradeManager(logs_dir="logs")'
    if old1 in text:
        text = text.replace(old1, new1, 1)
        changes += 1
        print("[+] Fix 1: TradeManager() -> TradeManager(logs_dir='logs')")
    elif new1 in text:
        print("[=] Fix 1: Already applied (TradeManager logs_dir)")
    else:
        print("[!] Fix 1: Could not find TradeManager() initialization line")

    # ── Fix 2: Add persist_open_positions() after write_status() ──
    marker = "autonomous_monitor.write_status(trader_signals_df, trades_df, alerts_df, open_positions_df_for_status)"
    persist_line = "trade_manager.persist_open_positions()"

    if persist_line in text:
        print("[=] Fix 2: Already applied (persist_open_positions)")
    elif marker in text:
        # Find the marker and add the persist call right after it
        idx = text.index(marker)
        # Walk backwards to find the indentation of the marker line
        line_start = text.rfind("\n", 0, idx)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1
        indent = ""
        for ch in text[line_start:idx]:
            if ch in (" ", "\t"):
                indent += ch
            else:
                break

        replacement = marker + "\n" + indent + persist_line
        text = text.replace(marker, replacement, 1)
        changes += 1
        print("[+] Fix 2: Added trade_manager.persist_open_positions() after write_status()")
    else:
        print("[!] Fix 2: Could not find write_status() marker line")

    if changes > 0:
        path.write_text(text, encoding="utf-8")
        print(f"\nDone: applied {changes} fix(es) to supervisor.py")
    else:
        print("\nNo changes needed — supervisor.py is already patched")


if __name__ == "__main__":
    patch()
