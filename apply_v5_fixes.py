import os
import re

def fix_duplicates():
    file_path = "supervisor.py"
    if not os.path.exists(file_path):
        print(f"[-] {file_path} not found.")
        return
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # 1. Deduplicate features_df right after creation
    content = re.sub(
        r'(features_df\s*=\s*feature_builder\.build_features\(signals_df,\s*markets_df\))',
        r'\1\n            if features_df is not None: features_df = features_df.loc[:, ~features_df.columns.duplicated()].copy()',
        content
    )

    # 2. Deduplicate scored_df right after creation
    content = re.sub(
        r'(scored_df\s*=\s*signal_engine\.score_features\(inferred_df\))',
        r'\1\n            if scored_df is not None: scored_df = scored_df.loc[:, ~scored_df.columns.duplicated()].copy()',
        content
    )

    # 3. Guard the exact line that crashes the bot during the analytics write
    content = re.sub(
        r'(trader_signals_df\s*=\s*scored_df\.rename\()',
        r'if scored_df is not None: scored_df = scored_df.loc[:, ~scored_df.columns.duplicated()].copy()\n            \1',
        content
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("[+] Successfully applied bulletproof duplicate column fixes to supervisor.py!")

if __name__ == "__main__":
    fix_duplicates()