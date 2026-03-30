import os
import re
import shutil
from datetime import datetime

FILES_TO_PATCH = [
    "live_position_book.py",
    "position_manager.py",
    "live_pnl.py",
    "reconciliation_service.py"
]

def backup_file(filepath):
    if not os.path.exists(filepath):
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.{timestamp}.bak"
    shutil.copy2(filepath, backup_path)
    print(f"[+] Backed up {filepath} -> {backup_path}")

def patch_file(filepath, patch_func):
    if not os.path.exists(filepath):
        print(f"[-] {filepath} not found. Skipping.")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        original_content = f.read()

    patched_content = patch_func(original_content)

    if patched_content != original_content:
        backup_file(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(patched_content)
        print(f"[+] Successfully fixed Phase 5 bugs in {filepath}")
    else:
        print(f"[-] No changes needed for {filepath} (or patterns didn't match)")

# --- Patching Functions ---

def patch_live_position_book(content):
    # BUG 4: The NaN DataFrame Empty Columns Crash
    content = re.sub(
        r'return pd\.DataFrame\(list\(books\.values\(\)\)\)',
        r'return pd.DataFrame(list(books.values())) if books else pd.DataFrame(columns=["position_key", "token_id", "condition_id", "outcome_side", "shares", "avg_entry_price", "realized_pnl", "last_fill_at", "source", "status"]) # BUG FIX 4',
        content
    )

    # BUG 1: API Rate Limit Database Wipe
    content = re.sub(
        r'if not isinstance\(payload, dict\):\n\s*return 0\.0',
        r'if not isinstance(payload, dict): return None\n        if "error" in payload or "message" in payload: return None # BUG FIX 1: Prevent API errors from wiping DB',
        content
    )
    content = re.sub(
        r'if available_shares <= 1e-9:',
        r'if available_shares is not None and available_shares <= 1e-9: # BUG FIX 1',
        content
    )
    return content

def patch_position_manager(content):
    # BUG 6: Falsy $0.00 Midpoint Poisoning
    content = re.sub(
        r'current_price = quote\.get\("midpoint"\) or quote\.get\("last_trade_price"\) or quote\.get\("price"\)',
        r'current_price = quote.get("midpoint") if quote.get("midpoint") is not None else quote.get("last_trade_price", quote.get("price")) # BUG FIX 6',
        content
    )

    # BUG 3: Zero Liquidity Trap Bypass
    content = re.sub(
        r'bid_size = float\(row\.get\("bid_size", self\.min_bid_size_to_exit\) or self\.min_bid_size_to_exit\)',
        r'bs = row.get("bid_size"); bid_size = float(bs) if bs is not None else self.min_bid_size_to_exit # BUG FIX 3',
        content
    )

    # BUG 2: Timestamp Timezone Subtraction Crash
    content = re.sub(
        r'minutes_open = \(pd\.Timestamp\.now\(\) - opened_at\)\.total_seconds\(\) / 60\.0 if pd\.notna\(opened_at\) else 0\.0',
        r'minutes_open = (pd.Timestamp.now(tz=opened_at.tz) - opened_at).total_seconds() / 60.0 if pd.notna(opened_at) else 0.0 # BUG FIX 2',
        content
    )

    # BUG 7 & 10: Incomplete Reduce Closures & Dust
    content = re.sub(
        r'positions\.at\[idx, "shares"\] = shares_remaining',
        r'positions.at[idx, "shares"] = round(shares_remaining, 6)\n        if round(shares_remaining, 6) <= 0: positions.at[idx, "status"] = "CLOSED" # BUG FIX 7 & 10',
        content
    )
    return content

def patch_live_pnl(content):
    # BUG 5: Non-DataFrame Crashes
    content = re.sub(
        r'if positions_df is None or positions_df\.empty:',
        r'if not isinstance(positions_df, pd.DataFrame) or positions_df.empty: # BUG FIX 5',
        content
    )
    return content

def patch_reconciliation_service(content):
    # BUG 8: CSV String ID Coercion Corruption
    content = re.sub(
        r'try:\n\s*return pd\.read_csv\(path, engine="python", on_bad_lines="skip"\)',
        r'try:\n            return pd.read_csv(path, engine="python", on_bad_lines="skip", dtype=str) # BUG FIX 8',
        content
    )

    # BUG 9: API String Iteration Crash
    content = re.sub(
        r'if isinstance\(payload, list\):\n\s*return payload',
        r'if isinstance(payload, str): return [] # BUG FIX 9\n        if isinstance(payload, list):\n            return payload',
        content
    )
    return content

if __name__ == "__main__":
    print("=== Commencing Phase 5 Deep Hunt Bug Fixes ===")
    patch_file("live_position_book.py", patch_live_position_book)
    patch_file("position_manager.py", patch_position_manager)
    patch_file("live_pnl.py", patch_live_pnl)
    patch_file("reconciliation_service.py", patch_reconciliation_service)
    print("=== Done! ===")