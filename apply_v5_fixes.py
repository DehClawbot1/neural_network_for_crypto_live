import os
import re
import shutil
from datetime import datetime

FILES_TO_PATCH = [
    "trade_lifecycle.py",
    "supervisor.py",
    "order_manager.py",
    "trade_manager.py"
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

def patch_trade_lifecycle(content):
    # BUG 1: Do not zero shares so supervisor can actually sell them
    content = re.sub(
        r'self\.shares\s*=\s*0\.0\s*\n\s*self\.size_usdc\s*=\s*0\.0',
        '# self.shares = 0.0 # BUG 1 FIX: Keep shares intact so supervisor knows how much to sell\n        # self.size_usdc = 0.0',
        content
    )
    return content

def patch_supervisor(content):
    # BUG 6: Restore dropped metadata
    content = re.sub(
        r'(trade\.enter\(size_usdc=size_usdc, entry_price=actual_fill_price\))',
        r'trade.confidence_at_entry = confidence # BUG 6 FIX\n                        trade.signal_label = signal_row.get("signal_label", "UNKNOWN")\n                        \1',
        content
    )
    
    # BUG 7: Reject $0.00 bids
    content = re.sub(
        r'if _ob_exit\.get\("best_bid"\) is not None:',
        'if _ob_exit.get("best_bid") is not None and float(_ob_exit.get("best_bid")) > 0: # BUG 7 FIX: Prevent $0 limit sells',
        content
    )

    # BUG 8: Fix string falsiness
    content = re.sub(
        r'actual_fill_size\s*=\s*float\(fill_payload\.get\("size", ([a-zA-Z0-9_]+)\) or ([a-zA-Z0-9_]+)\)',
        r'sz = fill_payload.get("size", \1)\n                                        actual_fill_size = float(sz) if (sz is not None and str(sz).strip() and float(sz) > 0) else \2 # BUG 8 FIX',
        content
    )

    # BUG 9: Prevent sync_summary NoneType crash
    content = re.sub(
        r'(sync_summary\s*=\s*reconciliation_service\.sync_orders_and_fills\(\))',
        r'\1\n                    if not sync_summary: sync_summary = {} # BUG 9 FIX',
        content
    )
    return content

def patch_order_manager(content):
    # BUG 3: Only apply $0.99 limit to BUYS to prevent trapping depreciated bags
    content = re.sub(
        r'(if notional_val < 0\.99:)',
        r'if notional_val < 0.99 and normalized_side == "BUY": # BUG 3 FIX: Allow liquidating depreciated bags',
        content
    )

    # BUG 4: Prevent 0.001 dust stranding
    content = re.sub(
        r'max_sell_shares\s*=\s*self\._round_down_shares\(max\(0\.0, available_shares \* 0\.999\)\)',
        r'max_sell_shares = self._round_down_shares(max(0.0, available_shares)) # BUG 4 FIX: Clear entire bag, no dust',
        content
    )

    # BUG 5: Prevent double-dividing already normalized tokens
    content = re.sub(
        r'if is_micro:\n(\s*)return val / 1e6',
        r'if is_micro:\n\1return (val / 1e6) if val > 100 else val # BUG 5 FIX: Protect conditional token sizes',
        content
    )

    # BUG 10: Prevent NoneType crash on order_id
    content = re.sub(
        r'order_id\s*=\s*response\.get\("orderID"\)',
        r'order_id = (response or {}).get("orderID") # BUG 10 FIX',
        content
    )
    return content

def patch_trade_manager(content):
    # BUG 2: Fix API lag erasure by merging instead of blindly overwriting active_trades
    old_recon = r'self\.active_trades\s*=\s*rebuilt_trades\n\s*logger\.info\("\[~\] Reconciled %s live positions into TradeManager\.", len\(self\.active_trades\)\)'
    new_recon = """# BUG 2 FIX: Merge to avoid erasing newly opened un-indexed trades
        cutoff = datetime.now(timezone.utc).timestamp() - 60
        for key, rebuilt_trade in rebuilt_trades.items():
            if key in self.active_trades:
                self.active_trades[key].current_price = rebuilt_trade.current_price
                self.active_trades[key].unrealized_pnl = rebuilt_trade.unrealized_pnl
            else:
                self.active_trades[key] = rebuilt_trade
        for key in list(self.active_trades.keys()):
            if key not in rebuilt_trades:
                try:
                    open_ts = datetime.fromisoformat(self.active_trades[key].opened_at).timestamp()
                    if open_ts < cutoff:
                        self.active_trades.pop(key)
                except Exception:
                    pass
        logger.info("[~] Reconciled %s live positions into TradeManager.", len(self.active_trades))"""
    
    content = re.sub(old_recon, new_recon, content)
    return content

if __name__ == "__main__":
    print("=== Commencing Deep Hunt Bug Fixes ===")
    patch_file("trade_lifecycle.py", patch_trade_lifecycle)
    patch_file("supervisor.py", patch_supervisor)
    patch_file("order_manager.py", patch_order_manager)
    patch_file("trade_manager.py", patch_trade_manager)
    print("=== Done! ===")