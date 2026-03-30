import os
import re
import shutil
from datetime import datetime

FILES_TO_PATCH = [
    "run_live_test.py",
    "market_monitor.py",
    "contract_target_builder.py",
    "model_inference.py"
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
        print(f"[+] Successfully fixed Phase 4 bugs in {filepath}")
    else:
        print(f"[-] No changes needed for {filepath} (or patterns didn't match)")

# --- Patching Functions ---

def patch_run_live_test(content):
    # BUG 3: Environment Variable Race Condition
    new_content = """import os

if __name__ == "__main__":
    os.environ["TRADING_MODE"] = os.getenv("TRADING_MODE", "live")
    from run_bot import main as run_main # BUG FIX 3: Import AFTER env var is set
    run_main()
"""
    return new_content

def patch_market_monitor(content):
    # BUG 4: Pagination Early Exit
    content = re.sub(
        r'if len\(markets\) < int\(limit\):\n\s*break',
        r'if not markets: break # BUG FIX 4: Prevent arbitrary page-size limits from halting fetch',
        content
    )

    # BUG 2: Infinite Snapshot Bloat
    content = re.sub(
        r'dedupe_cols = \[c for c in \["market_id", "condition_id", "slug", "timestamp"\] if c in merged\.columns\]',
        r'dedupe_cols = [c for c in ["market_id", "condition_id", "slug"] if c in merged.columns] # BUG FIX 2: Stop deduplicating on timestamp',
        content
    )

    # BUG 9: Safe-Float Fails on Empty Strings
    content = re.sub(
        r'midpoint = \(float\(best_bid\) \+ float\(best_ask\)\) / 2\.0\n\s*spread = abs\(float\(best_ask\) - float\(best_bid\)\)',
        r'midpoint = (_safe_float(best_bid, 0.0) + _safe_float(best_ask, 0.0)) / 2.0\n            spread = abs(_safe_float(best_ask, 0.0) - _safe_float(best_bid, 0.0)) # BUG FIX 9: Handle empty strings gracefully',
        content
    )

    # BUG 10: Falsy Zero Overrides
    content = re.sub(
        r'best_bid = market\.get\("bestBid"\) or market\.get\("best_bid"\) or market\.get\("bid"\)\n\s*best_ask = market\.get\("bestAsk"\) or market\.get\("best_ask"\) or market\.get\("ask"\)',
        r'best_bid = market.get("bestBid") if market.get("bestBid") is not None else market.get("best_bid", market.get("bid"))\n    best_ask = market.get("bestAsk") if market.get("bestAsk") is not None else market.get("best_ask", market.get("ask")) # BUG FIX 10: Do not drop explicit 0.0s',
        content
    )
    return content

def patch_contract_target_builder(content):
    # BUG 1: O(N*M) Pipeline Freeze
    content = re.sub(
        r'(rows = \[\]\n\s*for _, signal_row in signals_df\.iterrows\(\):)',
        r'rows = []\n        history_groups = dict(tuple(history_df.groupby("token_id"))) # BUG FIX 1: O(1) lookups\n        for _, signal_row in signals_df.iterrows():',
        content
    )
    content = re.sub(
        r'token_history = history_df\[history_df\["token_id"\]\.astype\(str\) == str\(token_id\)\]\.copy\(\)',
        r'token_history = history_groups.get(str(token_id), pd.DataFrame()).copy() # BUG FIX 1: Prevent pipeline freeze',
        content
    )

    # BUG 5: Inconsistent Move Normalization (Applies to both _path_stats and build)
    content = re.sub(
        r'moves = \[float\(price\) - float\(entry_price\) for price in path_prices\]',
        r'moves = [(float(price) - float(entry_price)) / float(entry_price) for price in path_prices] # BUG FIX 5: Normalize to ROI',
        content
    )
    content = re.sub(
        r'moves = \[float\(price\) - entry_price for price in path_moves\]',
        r'moves = [(float(price) - entry_price) / entry_price for price in path_moves] # BUG FIX 5: Normalize to ROI',
        content
    )
    return content

def patch_model_inference(content):
    # BUG 6: Sigmoid Overflow Poisoning
    content = re.sub(
        r'probs = 1\.0 / \(1\.0 \+ np\.exp\(-raw\)\)',
        r'probs = 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500))) # BUG FIX 6: Prevent NaN poisoning',
        content
    )

    # BUG 7: 2D Array Flattening
    content = re.sub(
        r'pd\.Series\(preds, index=out\.index\)',
        r'pd.Series(np.array(preds).ravel(), index=out.index) # BUG FIX 7: Support 2D (N,1) regressor arrays',
        content
    )

    # BUG 8: DataFrame Fragmentation Warnings
    content = re.sub(
        r'for col in feature_names:\n\s*if col not in work\.columns:\n\s*work\[col\] = 0\.0',
        r'missing = {col: 0.0 for col in feature_names if col not in work.columns}\n        if missing: work = work.assign(**missing) # BUG FIX 8: Prevent DF Fragmentation',
        content
    )
    return content

if __name__ == "__main__":
    print("=== Commencing Phase 4 Deep Hunt Bug Fixes ===")
    patch_file("run_live_test.py", patch_run_live_test)
    patch_file("market_monitor.py", patch_market_monitor)
    patch_file("contract_target_builder.py", patch_contract_target_builder)
    patch_file("model_inference.py", patch_model_inference)
    print("=== Done! ===")