import os
import re
import shutil
from datetime import datetime

FILES_TO_PATCH = [
    "orderbook_guard.py",
    "money_manager.py",
    "execution_client.py",
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
        print(f"[+] Successfully fixed Phase 8 Risk/Execution bugs in {filepath}")
    else:
        print(f"[-] No changes needed for {filepath} (or patterns didn't match)")

# --- Patching Functions ---

def patch_orderbook_guard(content):
    # BUG 1: The "Crossed Book" Slippage Trap
    content = re.sub(
        r'(spread\s*=\s*float\(best_ask\) - float\(best_bid\))',
        r'\1\n        if spread < 0: return False, "Orderbook is crossed (spread < 0). Liquidity is unstable." # BUG FIX 1',
        content
    )
    
    # BUG 5: Orderbook Depth IndexError Protection
    content = re.sub(
        r'liquidity\s*=\s*sum\(float\(level\["size"\]\) for level in orderbook\["asks"\]\[:10\]\)',
        r'liquidity = sum(float(level.get("size", 0)) for level in orderbook.get("asks", [])[:10] if level) # BUG FIX 5',
        content
    )
    return content

def patch_money_manager(content):
    # BUG 2: Kelly Criterion Zero-Division Crash
    content = re.sub(
        r'(win_rate\s*=\s*len\(wins\)\s*/\s*total_trades)',
        r'win_rate = len(wins) / total_trades if total_trades > 0 else 0.0 # BUG FIX 2\n        if total_trades == 0: return self.default_bet_size',
        content
    )
    
    # BUG 10: Truncation vs. Rounding Dust
    content = re.sub(
        r'shares_to_buy\s*=\s*int\(capital_allocated / price\)',
        r'shares_to_buy = round(capital_allocated / price, 6) # BUG FIX 10: Allow fractional shares',
        content
    )
    return content

def patch_execution_client(content):
    # BUG 3: Infinite Price Precision Rejections
    content = re.sub(
        r'(\s*)"price": price,',
        r'\1"price": round(float(price), 3), # BUG FIX 3: Enforce strict Clob Tick Size limits',
        content
    )
    content = re.sub(
        r'(\s*)"price": str\(price\),',
        r'\1"price": str(round(float(price), 3)), # BUG FIX 3: Enforce strict Clob Tick Size limits',
        content
    )

    # BUG 4: The Null-Nonce Concurrency Drop
    content = re.sub(
        r'nonce\s*=\s*int\(time\.time\(\) \* 1000\)',
        r'nonce = int(time.time() * 1000) + getattr(self, "_nonce_offset", 0)\n        self._nonce_offset = (getattr(self, "_nonce_offset", 0) + 1) % 1000 # BUG FIX 4: Ensure unique nonces',
        content
    )
    
    # BUG 7: "False" String Bypassing Status Checks
    content = re.sub(
        r'if not order\.get\("is_active"\):',
        r'if str(order.get("is_active")).lower() in ["false", "none", "0", ""]: # BUG FIX 7: Safely parse API boolean strings',
        content
    )

    # BUG 8: Order Polling Rate-Limit Spam
    content = re.sub(
        r'time\.sleep\(\s*0\.1\s*\)',
        r'time.sleep(1.0) # BUG FIX 8: Prevent Cloudflare Ratelimit IP Bans',
        content
    )
    return content

def patch_live_risk_manager(content):
    # BUG 6: The "Daily Loss" Absolute Math Trap
    content = re.sub(
        r'if daily_pnl < -self\.max_daily_loss:',
        r'if daily_pnl < -(self.max_daily_loss if self.max_daily_loss > 1 else self.max_daily_loss * self.total_capital): # BUG FIX 6: Handle both absolute and percentage inputs',
        content
    )
    
    # BUG 9: Missing Timezone in Risk DB
    content = re.sub(
        r'datetime\.now\(\)',
        r'datetime.now(timezone.utc) # BUG FIX 9: Enforce UTC for DB audit joins',
        content
    )
    return content

if __name__ == "__main__":
    print("=== Commencing Phase 8 Risk, Execution & Orderbook Bug Fixes ===")
    patch_file("orderbook_guard.py", patch_orderbook_guard)
    patch_file("money_manager.py", patch_money_manager)
    patch_file("execution_client.py", patch_execution_client)
    patch_file("live_risk_manager.py", patch_live_risk_manager)
    print("=== Done! ===")