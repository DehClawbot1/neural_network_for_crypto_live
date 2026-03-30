import os
import re
import shutil
from datetime import datetime

FILES_TO_PATCH = [
    "execution_client.py",
    "trade_manager.py",
    "pnl_engine.py",
    "signal_engine.py"
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
        print(f"[+] Successfully fixed bugs in {filepath}")
    else:
        print(f"[-] No changes needed for {filepath} (or already patched)")

# --- Patching Logic ---

def patch_execution_client(content):
    # BUG 1: The < $1.00 Microdollar Blackhole
    content = re.sub(
        r'(if is_micro and val >= )1_000_000( and val == int\(val\):)',
        r'\1 100: # BUG FIX 1: Support balances under $1.00',
        content
    )
    
    # BUG 7: Silent Options Erasure
    content = re.sub(
        r'(except Exception:)\n(\s*create_options = None)',
        r'\1\n\2 # BUG FIX 7: Log the failure to prevent silent erasure\n                    logging.warning("Failed to map PartialCreateOrderOptions. V5 tick sizing may be dropped.")',
        content
    )
    
    # BUG 10: Falsy String Casting
    content = re.sub(
        r'val = float\(raw_balance\)',
        r'val = float(raw_balance) if str(raw_balance).strip() else 0.0 # BUG FIX 10: Handle empty string API drops safely',
        content
    )
    return content

def patch_trade_manager(content):
    # BUG 2: API Timeout Erasing Trades
    content = re.sub(
        r'(if bal_val < 10000:)',
        r'if raw_bal is not None and bal_val < 10000: # BUG FIX 2: Protect against NoneType API drops',
        content
    )
    
    # BUG 3: "Sell" Entry Duplication Bug
    content = re.sub(
        r'(market = signal_row\.get\("market_title"\) or signal_row\.get\("market"\))',
        r'if str(signal_row.get("action", "BUY")).upper() != "BUY":\n            return None # BUG FIX 3: Prevent opening new trades on EXIT signals\n\n        \1',
        content
    )
    
    # BUG 5: Active Trades Overwritten
    overwrite_pattern = r'self\.active_trades\s*=\s*rebuilt_trades\n\s*logger\.info\("\[~\] Reconciled %s live positions into TradeManager\.", len\(self\.active_trades\)\)'
    merge_fix = """# BUG FIX 5: Merge incoming positions instead of blindly overwriting
        for key, r_trade in rebuilt_trades.items():
            if key not in self.active_trades: self.active_trades[key] = r_trade
        logger.info("[~] Reconciled %s live positions into TradeManager.", len(self.active_trades))"""
    content = re.sub(overwrite_pattern, merge_fix, content)
    
    # BUG 8: Naive Timestamp Crashing
    content = re.sub(
        r'current_ts = current_timestamp\.replace\(tzinfo=None\) if current_timestamp\.tzinfo is not None else current_timestamp',
        r'current_ts = current_timestamp.replace(tzinfo=timezone.utc) if current_timestamp.tzinfo is None else current_timestamp # BUG FIX 8: Enforce UTC safely\n            if opened_dt.tzinfo is None: opened_dt = opened_dt.replace(tzinfo=timezone.utc)',
        content
    )
    
    # BUG 9: Spread Collapse Trailing Stop Trap
    content = re.sub(
        r'(trade\.peak_price = max\(trade\.peak_price, current_price\))',
        r'if current_price > trade.entry_price: # BUG FIX 9: Only raise peak on real profit, ignoring wide spreads at entry\n                \1',
        content
    )
    return content

def patch_signal_engine(content):
    # BUG 4: Heuristic-Only Deprecation Trap
    content = re.sub(
        r'(confidence = float\(np\.clip\(\(heuristic_confidence \* 0\.45\) \+ \(model_confidence \* 0\.55\), 0\.0, 1\.0\)\))',
        r'\1\n        if expected_return == 0.0 and p_tp == 0.0: confidence = heuristic_confidence # BUG FIX 4: Restore 100% heuristic weight if AI is offline',
        content
    )
    return content

def patch_pnl_engine(content):
    # BUG 6: Phantom Fractional Dust
    content = re.sub(
        r'(return float\(capital_usdc\) / float\(entry_price\))',
        r'return int((\1) * 1e6) / 1e6 # BUG FIX 6: Truncate precisely to 6 decimals for conditional tokens',
        content
    )
    return content

if __name__ == "__main__":
    print("=== Commencing Phase 2 Deep Hunt Bug Fixes ===")
    patch_file("execution_client.py", patch_execution_client)
    patch_file("trade_manager.py", patch_trade_manager)
    patch_file("pnl_engine.py", patch_pnl_engine)
    patch_file("signal_engine.py", patch_signal_engine)
    print("=== Done! ===")