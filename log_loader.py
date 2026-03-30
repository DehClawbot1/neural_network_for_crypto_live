from pathlib import Path

import pandas as pd


def load_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def load_execution_history(logs_dir="logs"):
    logs_dir = Path(logs_dir)
    current_df = load_csv(logs_dir / "execution_log.csv")
    legacy_df = load_csv(logs_dir / "daily_summary.txt")

    if current_df.empty:
        return legacy_df if not legacy_df.empty else pd.DataFrame()

    # Prefer the canonical execution log. Only merge legacy rows if they actually look compatible.
    if legacy_df.empty:
        return current_df

    compatible_cols = {"timestamp", "market", "wallet_copied", "fill_price", "size_usdc", "action_type"}
    if not compatible_cols.intersection(set(legacy_df.columns)):
        return current_df

    combined = pd.concat([legacy_df, current_df], ignore_index=True, sort=False)
    dedupe_cols = [c for c in ["timestamp", "market", "wallet_copied", "fill_price", "size_usdc", "action_type", "token_id", "order_id"] if c in combined.columns]
    if dedupe_cols:
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
    return combined

