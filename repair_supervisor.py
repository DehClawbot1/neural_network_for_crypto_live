"""
repair_supervisor.py
=====================
Fixes the broken supervisor.py indentation from the previous patch.
Restores from backup if available, then applies the orderbook guard cleanly.

Usage:
    python repair_supervisor.py
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


def find_backup():
    """Find the most recent supervisor.py backup."""
    backups_dir = Path("backups")
    if not backups_dir.exists():
        return None
    # Search all backup subdirectories
    candidates = list(backups_dir.rglob("supervisor.py"))
    if not candidates:
        return None
    # Return the most recently modified one
    return max(candidates, key=lambda p: p.stat().st_mtime)


def clean_and_patch(text):
    """
    Remove any broken orderbook guard insertions, then re-insert cleanly.
    """
    lines = text.split("\n")
    cleaned_lines = []
    skip_until_original = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip any previously inserted orderbook guard blocks
        if any(marker in stripped for marker in [
            "ob_check = orderbook_guard.check_before_entry",
            "ORDER BOOK GUARD: Check spread/depth before entry",
            "ORDER BOOK GUARD: Check book before entry",
            "ob_check = orderbook_guard",
            "ob_exit_check = orderbook_guard",
            "ORDER BOOK GUARD: Use real bid for exit",
            "ORDER BOOK GUARD: Get real bid price for exit",
            "_unused_",
            "# Original line (now handled above):",
            "# (original quote_entry_price kept as fallback above)",
            "# (original quote_exit_price kept as fallback above)",
        ]):
            # Skip this line and any continuation block
            i += 1
            # Skip the rest of the inserted block (indented lines that follow)
            while i < len(lines):
                next_stripped = lines[i].strip()
                if next_stripped == "" or next_stripped.startswith("#"):
                    i += 1
                    continue
                # If we hit a line that belongs to the original code, stop skipping
                if any(orig in next_stripped for orig in [
                    "fill_price = quote_entry_price",
                    "fill_price = ob_check",
                    "if fill_price is None",
                    "exit_price = quote_exit_price",
                    "exit_price = ob_exit_check",
                    "for _warn in",
                    "_spread_cost",
                    "ob_check[",
                    "orderbook_guard.",
                    "logging.info(",
                    "logging.warning(",
                    "continue",
                ]):
                    if "ob_check" in next_stripped or "orderbook_guard" in next_stripped or "_spread_cost" in next_stripped or "_warn" in next_stripped or "_unused_" in next_stripped:
                        i += 1
                        continue
                    else:
                        break
                else:
                    break
            continue

        # Also remove the orderbook_guard import if we'll re-add it
        if stripped == "from orderbook_guard import OrderBookGuard":
            i += 1
            continue

        # Remove the orderbook_guard initialization line
        if "orderbook_guard = OrderBookGuard(" in stripped:
            i += 1
            continue

        cleaned_lines.append(line)
        i += 1

    text = "\n".join(cleaned_lines)

    # ── Now insert the orderbook guard cleanly ──

    # 1. Add import
    if "from orderbook_guard import OrderBookGuard" not in text:
        text = text.replace(
            "from db import Database",
            "from db import Database\nfrom orderbook_guard import OrderBookGuard",
        )

    # 2. Initialize in main_loop
    old_init = 'trade_manager = TradeManager(logs_dir="logs")'
    new_init = 'trade_manager = TradeManager(logs_dir="logs")\n    orderbook_guard = OrderBookGuard(max_spread=0.10, min_bid_depth=2, min_ask_depth=2)'
    if "orderbook_guard = OrderBookGuard" not in text and old_init in text:
        text = text.replace(old_init, new_init)

    # 3. Insert order book check before entry
    # Find: fill_price = quote_entry_price(signal_row)
    # Replace with: order book check + fill_price from book
    old_entry = "                    fill_price = quote_entry_price(signal_row)"
    new_entry = """                    # ── Order book guard: check spread/depth before entry ──
                    try:
                        ob_check = orderbook_guard.check_before_entry(
                            token_id=token_id, side="BUY", intended_size_usdc=size_usdc,
                        )
                        if not ob_check["tradable"]:
                            logging.info("OrderBookGuard BLOCKED %s: %s", token_id[:16], ob_check["reason"])
                            continue
                        fill_price = ob_check.get("recommended_entry_price") or quote_entry_price(signal_row)
                        for _w in ob_check.get("warnings", []):
                            logging.warning("OrderBookGuard %s: %s", token_id[:16], _w)
                    except Exception as _ob_exc:
                        logging.warning("OrderBookGuard failed for %s: %s (using fallback price)", token_id[:16], _ob_exc)
                        fill_price = quote_entry_price(signal_row)"""

    if old_entry in text and "ob_check = orderbook_guard" not in text:
        # Only replace the FIRST occurrence (inside the entry block)
        text = text.replace(old_entry, new_entry, 1)

    # 4. Insert order book check for exit pricing (both occurrences)
    old_exit = "                            exit_price = quote_exit_price(pos_dict)"
    new_exit = """                            try:
                                _ob_exit = orderbook_guard.analyze_book(token_id, depth=5)
                                if _ob_exit.get("best_bid") is not None:
                                    exit_price = _ob_exit["best_bid"]
                                else:
                                    exit_price = quote_exit_price(pos_dict)
                            except Exception:
                                exit_price = quote_exit_price(pos_dict)"""

    if old_exit in text and "_ob_exit = orderbook_guard" not in text:
        text = text.replace(old_exit, new_exit)

    return text


def main():
    print("=" * 55)
    print("REPAIRING supervisor.py")
    print("=" * 55)
    print()

    supervisor_path = Path("supervisor.py")
    if not supervisor_path.exists():
        print("[!] supervisor.py not found in current directory.")
        return

    # Try to restore from backup first
    backup_path = find_backup()
    if backup_path:
        print(f"[+] Found backup: {backup_path}")
        print(f"    Restoring original supervisor.py from backup...")
        shutil.copy2(backup_path, supervisor_path)
        print(f"    [OK] Restored.")
    else:
        print("[~] No backup found. Will clean the current file.")

    # Read the (restored or current) file
    text = supervisor_path.read_text(encoding="utf-8")

    # Verify it can be parsed before patching
    try:
        compile(text, "supervisor.py", "exec")
        print("[+] supervisor.py syntax is valid (before patch).")
    except SyntaxError as exc:
        print(f"[!] supervisor.py has syntax errors even after restore: {exc}")
        print("    Attempting to clean anyway...")

    # Apply clean patch
    patched = clean_and_patch(text)

    # Verify the patched version compiles
    try:
        compile(patched, "supervisor.py", "exec")
        print("[+] Patched supervisor.py syntax is valid!")
    except SyntaxError as exc:
        print(f"[!] Patch created a syntax error at line {exc.lineno}: {exc.msg}")
        print(f"    Line: {exc.text}")
        print()
        print("    Saving broken version as supervisor.py.broken for inspection.")
        Path("supervisor.py.broken").write_text(patched, encoding="utf-8")
        print("    Original supervisor.py was NOT changed.")
        return

    # Save
    # Backup current broken version just in case
    broken_backup = Path("backups") / f"supervisor_broken_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    broken_backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(supervisor_path, broken_backup)

    supervisor_path.write_text(patched, encoding="utf-8")
    print(f"[+] supervisor.py saved with clean orderbook guard patch.")
    print()
    print("What was added:")
    print("  - import OrderBookGuard at top")
    print("  - Initialize orderbook_guard in main_loop")
    print("  - Before every BUY entry: check spread, depth, get real ask price")
    print("  - Before every SELL exit: get real bid price from order book")
    print("  - All wrapped in try/except so failures fall back to old pricing")
    print()
    print("Run your bot:")
    print("  python launch.py")


if __name__ == "__main__":
    main()
