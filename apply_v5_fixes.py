import os

def fix_duplicate_columns():
    file_path = "supervisor.py"
    if not os.path.exists(file_path):
        print(f"[-] {file_path} not found.")
        return
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Fix 1: Protect signals_df and features_df
    target1 = "features_df = feature_builder.build_features(signals_df, markets_df)"
    replacement1 = """if signals_df is not None and not signals_df.empty: signals_df = signals_df.loc[:, ~signals_df.columns.duplicated()]
            features_df = feature_builder.build_features(signals_df, markets_df)
            if features_df is not None and not features_df.empty: features_df = features_df.loc[:, ~features_df.columns.duplicated()]"""
    if "signals_df.loc[:, ~signals_df.columns.duplicated()]" not in content:
        content = content.replace(target1, replacement1)

    # Fix 2: Protect inferred_df and scored_df before processing
    target2 = "scored_df = signal_engine.score_features(inferred_df)"
    replacement2 = """if inferred_df is not None and not inferred_df.empty: inferred_df = inferred_df.loc[:, ~inferred_df.columns.duplicated()]
            scored_df = signal_engine.score_features(inferred_df)
            if scored_df is not None and not scored_df.empty: scored_df = scored_df.loc[:, ~scored_df.columns.duplicated()]"""
    if "inferred_df.loc[:, ~inferred_df.columns.duplicated()]" not in content:
        content = content.replace(target2, replacement2)

    # Fix 3: Protect markets_df before merging
    target3 = "save_market_snapshot(markets_df)"
    replacement3 = """if markets_df is not None and not markets_df.empty: markets_df = markets_df.loc[:, ~markets_df.columns.duplicated()]
            save_market_snapshot(markets_df)"""
    if "markets_df.loc[:, ~markets_df.columns.duplicated()]" not in content:
        content = content.replace(target3, replacement3)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("[+] Successfully patched supervisor.py to prevent duplicate column crashes!")

if __name__ == "__main__":
    fix_duplicate_columns()