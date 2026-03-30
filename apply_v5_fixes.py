import os
import re
import shutil
from datetime import datetime

FILES_TO_PATCH = [
    "money_manager.py",
    "polytrade_env.py",
    "live_risk_manager.py"
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
        print(f"[+] Successfully fixed Phase 3 bugs in {filepath}")
    else:
        print(f"[-] No changes needed for {filepath} (or patterns didn't match)")

# --- Patching Functions ---

def patch_money_manager(content):
    # BUG 2: Fix Total Exposure math to use full capital stack
    content = re.sub(
        r'(max_total_exposure\s*=\s*)available_balance(\s*\*\s*TradingConfig\.MAX_TOTAL_EXPOSURE_PCT)',
        r'\1(available_balance + current_exposure)\2 # BUG FIX 2: Compute against total portfolio',
        content
    )

    # BUG 3: Prevent Dynamic Min overriding Max Risk
    fix_2_pattern = r'(if adjusted_bet < dynamic_min:.*?)(adjusted_bet = round\(adjusted_bet, 2\))'
    fix_2_replacement = r"""if dynamic_max < absolute_floor:
            logging.info("MoneyManager: dynamic_max ($%.2f) < absolute_floor ($%.2f). Rejecting trade.", dynamic_max, absolute_floor)
            return 0.0
            
        if adjusted_bet < dynamic_min:
            if remaining_capacity >= dynamic_min:
                adjusted_bet = min(dynamic_min, dynamic_max) # BUG FIX 3: Never break max risk for the sake of minimums
            else:
                return 0.0

        \2"""
    content = re.sub(fix_2_pattern, fix_2_replacement, content, flags=re.DOTALL)
    
    return content

def patch_polytrade_env(content):
    # BUG 4: Stop Phantom Short Selling spam on 0 inventory
    content = re.sub(
        r'size = max\(1\.0, min\(max\(float\(self\.shares\), 1\.0\), max\(float\(self\.shares\), 50\.0\) \* abs\(action_value\)\)\)',
        r'size = max(0.1, min(float(self.shares), float(self.shares) * abs(action_value))) if self.shares > 0 else 0.0\n                if size <= 0: return {"status": "NO_POSITION_TO_EXIT"} # BUG FIX 4: Prevent short-sell loops',
        content
    )

    # BUG 5: Stop Blind Partial Fill Erasure
    content = re.sub(
        r'reduce_size = max\(float\(self\.shares\) \* 0\.5, 0\.0\)\n(\s*)self\.shares = max\(float\(self\.shares\) - reduce_size, 0\.0\)',
        r'reduce_size = float(filled_size) if filled_size > 0 else max(float(self.shares) * 0.5, 0.0)\n\1self.shares = max(float(self.shares) - reduce_size, 0.0) # BUG FIX 5: Use actual fill amount',
        content
    )

    # BUG 8: Support both Dictionary and Object orderbook responses
    content = re.sub(
        r'bids = getattr\(orderbook, "bids", \[\]\)\[:5\]\n\s*asks = getattr\(orderbook, "asks", \[\]\)\[:5\]',
        r'bids = (orderbook.get("bids", []) if isinstance(orderbook, dict) else getattr(orderbook, "bids", []))[:5]\n            asks = (orderbook.get("asks", []) if isinstance(orderbook, dict) else getattr(orderbook, "asks", []))[:5] # BUG FIX 8',
        content
    )

    # BUG 9: Support Dictionary Level volume sums
    content = re.sub(
        r'bid_vol = sum\(float\(getattr\(level, "size", 0\.0\) or 0\.0\) for level in bids\)\n\s*ask_vol = sum\(float\(getattr\(level, "size", 0\.0\) or 0\.0\) for level in asks\)',
        r'bid_vol = sum(float((level.get("size", 0) if isinstance(level, dict) else getattr(level, "size", 0)) or 0) for level in bids)\n            ask_vol = sum(float((level.get("size", 0) if isinstance(level, dict) else getattr(level, "size", 0)) or 0) for level in asks) # BUG FIX 9',
        content
    )

    # BUG 1: Add a 10-second cache to the Binance Ticker to prevent Rate Limit bans
    cache_block = """
    _cached_btc_price = 0.0
    _cached_btc_time = 0.0
    def _safe_correlated_price(self):
        import time
        if time.time() - getattr(self, '_cached_btc_time', 0) < 10:
            return getattr(self, '_cached_btc_price', 0.0)
        try:
            response = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": "BTCUSDT"}, timeout=5)
            if response.ok:
                price = float(response.json().get("price") or 0.0)
                self._cached_btc_price = price
                self._cached_btc_time = time.time()
                return price
        except Exception:
            pass
        return getattr(self, '_cached_btc_price', 0.0) # BUG FIX 1: 10-second TTL cache to prevent IP bans"""
    
    content = re.sub(
        r'def _safe_correlated_price\(self\):.*?return 0\.0',
        cache_block,
        content,
        flags=re.DOTALL
    )

    return content

def patch_live_risk_manager(content):
    # BUG 10: Unsafe NoneType size casts
    content = re.sub(
        r'if float\(size\) > self\.max_position_size:',
        r'if float(size or 0.0) > self.max_position_size: # BUG FIX 10',
        content
    )

    # BUG 6: Missing DB Commits
    content = re.sub(
        r'(self\.db\.execute\([\s\S]*?\)\s*\n)',
        r'\1                if hasattr(self.db, "commit"): self.db.commit() # BUG FIX 6: Ensure audit logs save\n',
        content
    )

    # BUG 7: Add auto-reset for Circuit Breakers
    content = re.sub(
        r'def record_failed_order\(self\):\n\s*self\.failed_orders \+= 1',
        r'def record_failed_order(self):\n        self.failed_orders += 1\n\n    def record_successful_order(self):\n        self.failed_orders = 0 # BUG FIX 7: Decay circuit breaker on success',
        content
    )

    return content

if __name__ == "__main__":
    print("=== Commencing Phase 3 Deep Hunt Bug Fixes ===")
    patch_file("money_manager.py", patch_money_manager)
    patch_file("polytrade_env.py", patch_polytrade_env)
    patch_file("live_risk_manager.py", patch_live_risk_manager)
    print("=== Done! ===")