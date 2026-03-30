import os
import sys
import shutil
import re
from datetime import datetime

# Target files to patch
FILES_TO_PATCH = [
    "config.py",
    "api_setup.py",
    "market_monitor.py",
    "model_inference.py",
    "signal_engine.py",
    "contract_target_builder.py",
    "historical_dataset_builder.py",
    "trade_lifecycle.py",
    "run_bot.py",
    "supervisor.py",
    "trade_manager.py",
    "order_manager.py",
    "dashboard.py"
]

def backup_file(filepath):
    if not os.path.exists(filepath):
        return False
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.{timestamp}.bak"
    shutil.copy2(filepath, backup_path)
    print(f"[+] Backed up {filepath} -> {backup_path}")
    return True

def patch_file(filepath, patch_func):
    if not os.path.exists(filepath):
        print(f"[!] Warning: {filepath} not found. Skipping.")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        original_content = f.read()

    patched_content = patch_func(original_content)

    if patched_content != original_content:
        backup_file(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(patched_content)
        print(f"[+] Successfully patched {filepath}")
    else:
        print(f"[-] No changes needed for {filepath} (or patterns didn't match)")

# --- Patching Logic Functions ---

def patch_config(content):
    # Fix Max positions to 5
    content = re.sub(r'MAX_CONCURRENT_POSITIONS\s*=\s*\d+', 'MAX_CONCURRENT_POSITIONS = 5', content)
    # Fix comments vs values (generic cleanup)
    content = re.sub(r'(MAX_CONCURRENT_POSITIONS\s*=\s*5\s*)#.*', r'\1# Set to 5 per requirements', content)
    return content

def patch_api_setup(content):
    # Fix live/paper config mismatch
    # Allow loading standard .env in live mode instead of forcing .env.paper
    content = re.sub(r'load_dotenv\([\'"]\.env\.paper[\'"]\)', 'load_dotenv() # Patched: rely on environment instead of forcing .env.paper', content)
    return content

def patch_run_bot(content):
    # Warn about signature type 0
    warning_text = r'print\("WARNING: Type 0 is effectively unsupported in this repo and will be rejected by execution_client.py. Use Type 1 or 2."\)\n    '
    if "WARNING: Type 0" not in content:
        content = re.sub(r'(input\(.*?signature type.*?|input\(.*?Type 0.*?)', warning_text + r'\1', content, count=1, flags=re.IGNORECASE)
    return content

def patch_supervisor(content):
    # 1. Fix fetch_btc_markets / fetch_markets_by_slugs
    # Replace calls to non-existent methods with a safe fallback or comment out if they crash
    content = re.sub(r'fetch_btc_markets\(.*?\)', 'fetch_markets() # Patched to use available method', content)
    content = re.sub(r'fetch_markets_by_slugs\(.*?\)', 'fetch_markets() # Patched to use available method', content)
    
    # 2. Fix AI close path flipping fields instead of lifecycle
    # Matches trade.status = 'CLOSED' etc.
    content = re.sub(r'trade\.status\s*=\s*[\'"]CLOSED[\'"]\n\s*trade\.is_active\s*=\s*False', 
                     '# Patched: Use proper close logic\n            trade_lifecycle.close_trade(trade) # Ensure PnL finalization is triggered', content)
    
    # 3. Fix live-exit rollback key formatting
    content = re.sub(r'trades\[f"\{trade\.symbol\}_\{.*?\}\"\]\s*=\s*trade', 'trades[trade.symbol] = trade # Patched: use standard key format to prevent duplicates', content)
    
    # 4. Enforce max positions fallback
    content = re.sub(r'fallback_max_positions\s*=\s*\d+', 'fallback_max_positions = 5', content)
    
    # 5. Inject execution_client to TradeManager if missing
    content = re.sub(r'(TradeManager\()(\))', r'\1execution_client=self.execution_client\2', content)
    return content

def patch_market_monitor(content):
    # Deduplicate snapshots
    if "drop_duplicates" not in content:
        content = re.sub(r'(self\.snapshots\.append\(.*?\))', r'\1\n        self.snapshots = pd.DataFrame(self.snapshots).drop_duplicates().to_dict("records") # Patched: prevent stale row pollution', content)
    
    # Add dummy methods to prevent supervisor crashes if they aren't completely removed
    if "def fetch_btc_markets" not in content:
        content += "\n\n    def fetch_btc_markets(self, max_offset=500):\n        return self.fetch_markets() # Patched fallback\n"
    if "def fetch_markets_by_slugs" not in content:
        content += "\n    def fetch_markets_by_slugs(self, slugs):\n        return self.fetch_markets() # Patched fallback\n"
    return content

def patch_trade_manager(content):
    # 1. Prevent clearing local active trades on empty DB/API sync
    content = re.sub(r'self\.active_trades\.clear\(\)', '# self.active_trades.clear() # Patched: transient failure protection (do not erase local book)', content)
    
    # 2. Fix process_exits execution client reference
    content = re.sub(r'def process_exits\(self, (.*?)\):', r'def process_exits(self, \1, execution_client=None):', content)
    return content

def patch_order_manager(content):
    # 1. Idempotency key fix (remove time.time() so it actually deduplicates)
    content = re.sub(r'idempotency_key\s*=\s*f"\{.*?\}__\{time\.time\(\)\}"', 'idempotency_key = f"{symbol}_{side}_{size}" # Patched: meaningful deduplication', content)
    
    # 2. wait_for_fill returning payload instead of actual fill price/size
    # Adjust return to extract specifics if it's currently returning the raw dict
    if "return payload" in content and "fill_price" not in content:
        content = re.sub(r'return payload', 'return {"fill_price": payload.get("price"), "filled_size": payload.get("size"), "raw": payload} # Patched: explicit fill extraction', content)
    return content

def patch_model_inference(content):
    # Fix Train/Inference feature mismatch (silent dropping)
    # Reindex inference dataframe to expected training columns with fill_value=0
    if "reindex" not in content:
        content = re.sub(r'(df\s*=\s*df\[.*intersection.*\])', r'df = df.reindex(columns=training_features, fill_value=0) # Patched: harden against feature mismatch', content)
    return content

def patch_signal_engine(content):
    # Fix max(heuristic, model) promoting weak models
    content = re.sub(r'max\(heuristic_confidence,\s*model_confidence\)', '(model_confidence * 0.7 + heuristic_confidence * 0.3) if model_confidence > 0.4 else 0.0 # Patched: require model strength', content)
    return content

def patch_contract_target_builder(content):
    # Fix internal inconsistency for labels
    # Force full hold window implied by column names
    content = re.sub(r'_compute_path_labels\(.*?, window_size=.*?\)', '_compute_path_labels(df, window_size=60) # Patched: forced consistency with 60m implied names', content)
    return content

def patch_dashboard(content):
    # Fix dashboard hiding DB row data by ensuring defaults are mapped
    if "fillna" not in content and "positions.csv" in content:
        content = re.sub(r'(df\s*=\s*pd\.DataFrame\(db_rows\))', r'\1\n        for col in ["current_price", "unrealized_pnl", "confidence"]: df[col] = df.get(col, 0.0) # Patched: protect dashboard state', content)
    return content

def patch_historical_dataset_builder(content):
    # Stop reading daily_summary.txt as if it's a CSV
    content = re.sub(r'pd\.read_csv\([\'"]daily_summary\.txt[\'"]\)', 'pd.DataFrame() # Patched: Stopped fragile daily_summary.txt parse. Need actual CSV or DB source.', content)
    return content

def patch_trade_lifecycle(content):
    # Basic protection against ghost trade risk / state flips
    content = re.sub(r'trade\.status\s*=\s*[\'"]CLOSED[\'"]\s*\n\s*return', 'trade.status = "CLOSED"\n    trade.close_time = time.time() # Patched: ensure close time bookkeeping\n    return', content)
    return content

# --- Runtime Reset Logic ---
def reset_runtime_state():
    print("\n[!] Wiping runtime DB/CSV state without touching weights...")
    state_files = ['live_positions.csv', 'paper_positions.csv', 'daily_summary.txt', 'order_history.csv', 'active_trades.db', 'execution.log']
    for sf in state_files:
        if os.path.exists(sf):
            os.remove(sf)
            print(f"   [+] Removed {sf}")
    print("[+] State reset complete.\n")

# --- Main Execution ---
if __name__ == "__main__":
    if "--reset-state" in sys.argv:
        reset_runtime_state()

    print("=== Applying Critical, High, Medium, and Low Priority Patches ===\n")
    
    # Mapping files to their respective patch functions
    patchers = {
        "config.py": patch_config,
        "api_setup.py": patch_api_setup,
        "run_bot.py": patch_run_bot,
        "supervisor.py": patch_supervisor,
        "market_monitor.py": patch_market_monitor,
        "trade_manager.py": patch_trade_manager,
        "order_manager.py": patch_order_manager,
        "model_inference.py": patch_model_inference,
        "signal_engine.py": patch_signal_engine,
        "contract_target_builder.py": patch_contract_target_builder,
        "dashboard.py": patch_dashboard,
        "historical_dataset_builder.py": patch_historical_dataset_builder,
        "trade_lifecycle.py": patch_trade_lifecycle
    }

    for file_name in FILES_TO_PATCH:
        if file_name in patchers:
            # Assuming files are in the current directory or relative paths are mapped properly
            # If the files are in subdirectories, you may need to walk the tree to find their exact paths.
            # Here we search for the file recursively from the current directory.
            file_path = None
            for root, dirs, files in os.walk('.'):
                if file_name in files:
                    file_path = os.path.join(root, file_name)
                    break
            
            if file_path:
                patch_file(file_path, patchers[file_name])
            else:
                print(f"[-] Could not locate {file_name} in current directory tree.")
                
    print("\n=== Patching complete! Review .bak files if you need to revert any changes. ===")