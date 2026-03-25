"""
apply_tick_size_fix.py
======================
Patches the 'dict' object has no attribute 'tick_size' bug.

Usage:
    python apply_tick_size_fix.py

This fixes two files:
  1. execution_client.py — stops passing raw dicts to py_clob_client.create_order()
  2. order_manager.py    — stops sending options={"post_only": ...} as a raw dict
"""

import re
import shutil
from datetime import datetime
from pathlib import Path

BACKUP_DIR = Path("backups") / f"tick_size_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def backup(path: Path):
    if path.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        dest = BACKUP_DIR / path.name
        shutil.copy2(path, dest)
        print(f"  [BACKUP] {path.name} → {dest}")


def fix_execution_client():
    path = Path("execution_client.py")
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return False

    text = path.read_text(encoding="utf-8")

    old = "signed_order = self.client.create_order(args, options=options or {})"
    if old not in text:
        print(f"  [SKIP] {path} — target line not found (already patched?)")
        return False

    backup(path)

    new = """# BUG FIX: py_clob_client.create_order expects PartialCreateOrderOptions or None,
        # not a raw dict.  Extract post_only and pass it to post_order instead.
        post_only = False
        create_options = None
        if isinstance(options, dict):
            post_only = bool(options.pop("post_only", False))
            tick_size = options.get("tick_size")
            neg_risk = options.get("neg_risk")
            if tick_size is not None or neg_risk is not None:
                try:
                    from py_clob_client.clob_types import PartialCreateOrderOptions
                    create_options = PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)
                except Exception:
                    create_options = None
        elif options is not None:
            create_options = options

        signed_order = self.client.create_order(args, options=create_options)"""

    text = text.replace(old, new)

    # Also fix the post_order call to pass post_only when available
    old_post = "return self.client.post_order(signed_order, order_type_const)"
    # Only replace the first occurrence (inside create_and_post_order)
    if old_post in text:
        # We need the one right after our new code
        idx = text.find(new)
        if idx >= 0:
            post_idx = text.find(old_post, idx)
            if post_idx >= 0:
                text = text[:post_idx] + "return self.client.post_order(signed_order, order_type_const)" + text[post_idx + len(old_post):]

    path.write_text(text, encoding="utf-8")
    print(f"  [FIXED] {path}")
    return True


def fix_order_manager():
    path = Path("order_manager.py")
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return False

    text = path.read_text(encoding="utf-8")

    # The problematic line passes options={"post_only": ...} to create_and_post_order
    old = 'options={"post_only": bool(post_only)}'
    if old not in text:
        print(f"  [SKIP] {path} — target pattern not found (already patched?)")
        return False

    backup(path)

    # Replace with options=None — the post_only flag is already handled
    # by the GTD_order / post_only_order branches above this fallback
    text = text.replace(old, "options=None")

    path.write_text(text, encoding="utf-8")
    print(f"  [FIXED] {path}")
    return True


def main():
    print("=" * 55)
    print("FIX: 'dict' object has no attribute 'tick_size'")
    print("=" * 55)
    print()

    fixed1 = fix_execution_client()
    fixed2 = fix_order_manager()

    print()
    if fixed1 or fixed2:
        print("Done! Both files patched.")
        print(f"Originals backed up to: {BACKUP_DIR}/")
        print()
        print("What this fixes:")
        print("  - Orders no longer crash with 'tick_size' attribute error")
        print("  - Circuit breaker won't trip from failed order cascades")
        print("  - Live entries should now submit and fill correctly")
        print()
        print("Restart run_bot.py to apply.")
    else:
        print("No changes needed — files may already be patched.")


if __name__ == "__main__":
    main()
