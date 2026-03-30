import os
import re
import shutil
from datetime import datetime

FILES_TO_PATCH = [
    "alerts_engine.py",
    "backtester.py",
    "wallet_alpha_builder.py",
    "rl_trainer.py",
    "dashboard.py"
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
        print(f"[+] Successfully fixed Phase 6 bugs in {filepath}")
    else:
        print(f"[-] No changes needed for {filepath} (or patterns didn't match)")

# --- Patching Functions ---

def patch_alerts_engine(content):
    # Fix import for timezone
    if "from datetime import timezone" not in content and "timezone" not in content:
        content = re.sub(r'from datetime import datetime', 'from datetime import datetime, timezone', content)
    
    # BUG 3: Tz-Naive Timestamp Subtraction Crash
    content = re.sub(
        r'datetime\.utcnow\(\)\.strftime\("%Y-%m-%d %H:%M:%S"\)',
        r'datetime.now(timezone.utc).isoformat() # BUG FIX 3: Prevent Tz-Naive crash',
        content
    )

    # BUG 9: Missing Level Fallback
    content = re.sub(
        r'severity = record\.get\("severity"\)',
        r'severity = record.get("severity", record.get("level")) # BUG FIX 9: Support alternative severity keys',
        content
    )
    return content

def patch_backtester(content):
    # BUG 5: String Coercion Crash
    content = re.sub(
        r'avg_hold = float\(df\[hold_col\]\.astype\(float\)\.mean\(\)\)',
        r'avg_hold = float(pd.to_numeric(df[hold_col], errors="coerce").mean()) # BUG FIX 5: Protect against string annotations',
        content
    )

    # BUG 8: Zero-Loss Infinity Masking
    content = re.sub(
        r'profit_factor = float\(gross_profit / gross_loss\) if gross_loss > 0 else np\.nan',
        r'profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0) # BUG FIX 8',
        content
    )
    return content

def patch_wallet_alpha_builder(content):
    # BUG 6: Non-Numeric Hit Rate Erasure
    content = re.sub(
        r'df\[hit_col\] = pd\.to_numeric\(df\[hit_col\], errors="coerce"\)',
        r'df[hit_col] = pd.to_numeric(df[hit_col].replace({"True": 1, "False": 0, "true": 1, "false": 0}), errors="coerce") # BUG FIX 6: Prevent erasure of boolean strings',
        content
    )

    # BUG 7: Expanding Mean NaN Bleed
    content = re.sub(
        r'(yes_returns = group\[return_col\]\.where\(yes_mask\)\.expanding\(min_periods=1\)\.mean\(\)\.shift\(1\))',
        r'\1.fillna(0.0) # BUG FIX 7',
        content
    )
    content = re.sub(
        r'(no_returns = group\[return_col\]\.where\(no_mask\)\.expanding\(min_periods=1\)\.mean\(\)\.shift\(1\))',
        r'\1.fillna(0.0) # BUG FIX 7',
        content
    )
    return content

def patch_rl_trainer(content):
    # BUG 2: Blind API Instantiation Crash
    content = re.sub(
        r'expected_dim = int\(PolyTradeEnv\(\)\.observation_space\.shape\[0\]\)',
        r'expected_dim = int(LiveReplayDatasetEnv(df).observation_space.shape[0]) # BUG FIX 2: Use dummy offline env to prevent API crash',
        content
    )

    # BUG 1: RL Thread ZipFile Fatality (Race condition)
    bad_loop = r'(model = PPO\.load\(model_path, env=env\)\n\s*print\(f"\[\+\] Fine-tuning.*?\n\s*model\.learn\(.*?\)\n\s*model\.save\(.*?\)\n\s*print\(.*?\))'
    safe_loop = r"""try:
            \1
        except Exception as e:
            print(f"[!] Fine-tuning interrupted (possible file lock race condition): {e}") # BUG FIX 1"""
    
    content = re.sub(bad_loop, safe_loop, content, flags=re.DOTALL)
    return content

def patch_dashboard(content):
    # BUG 4: The Paper Book Erasure Bug
    content = re.sub(
        r'(else:\n\s*)pdf = live_pdf',
        r'\1pdf = pd.concat([live_pdf, pdf], ignore_index=True) # BUG FIX 4: Append instead of wiping paper book',
        content
    )

    # BUG 10: Float Format NaN Exception
    content = re.sub(r'c5\.metric\("Avg Slippage \(bps\)", f"\{asl:\.1f\}"\)', r'c5.metric("Avg Slippage (bps)", f"{0.0 if pd.isna(asl) else asl:.1f}") # BUG FIX 10', content)
    content = re.sub(r'c6\.metric\("Avg EV_adj", f"\{aev:\+\.2\%\}"\)', r'c6.metric("Avg EV_adj", f"{0.0 if pd.isna(aev) else aev:+.2%}")', content)
    content = re.sub(r'c7\.metric\("Avg Meta Prob", f"\{amp:\.2\%\}"\)', r'c7.metric("Avg Meta Prob", f"{0.0 if pd.isna(amp) else amp:.2%}")', content)

    return content

if __name__ == "__main__":
    print("=== Commencing Phase 6 Deep Hunt Bug Fixes ===")
    patch_file("alerts_engine.py", patch_alerts_engine)
    patch_file("backtester.py", patch_backtester)
    patch_file("wallet_alpha_builder.py", patch_wallet_alpha_builder)
    patch_file("rl_trainer.py", patch_rl_trainer)
    patch_file("dashboard.py", patch_dashboard)
    print("=== Done! ===")