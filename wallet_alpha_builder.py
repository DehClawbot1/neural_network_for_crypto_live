from pathlib import Path

import pandas as pd


class WalletAlphaBuilder:
    """
    Estimate wallet quality from contract-level historical outcome labels.
    Research/paper-trading only.
    """

    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.contract_targets_file = self.logs_dir / "contract_targets.csv"
        self.output_file = self.logs_dir / "wallet_alpha.csv"
        self.history_file = self.logs_dir / "wallet_alpha_history.csv"

    def _safe_read(self, path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def build(self):
        df = self._safe_read(self.contract_targets_file)
        wallet_col = "wallet_copied" if "wallet_copied" in df.columns else "trader_wallet" if "trader_wallet" in df.columns else None
        return_col = "future_return" if "future_return" in df.columns else "forward_return_15m" if "forward_return_15m" in df.columns else None
        hit_col = "target_up" if "target_up" in df.columns else "tp_before_sl_60m" if "tp_before_sl_60m" in df.columns else None
        if df.empty or wallet_col is None or return_col is None:
            return pd.DataFrame()

        alpha = (
            df.groupby(wallet_col)
            .agg(
                observations=(wallet_col, "size"),
                avg_future_return=(return_col, "mean"),
                hit_rate=(hit_col, "mean") if hit_col else (return_col, "mean"),
                avg_confidence=("confidence", "mean") if "confidence" in df.columns else (return_col, "mean"),
            )
            .reset_index()
            .sort_values(by=["avg_future_return", "hit_rate"], ascending=[False, False])
        )
        alpha = alpha.rename(columns={wallet_col: "wallet_copied"})
        return alpha

    def build_history(self):
        df = self._safe_read(self.contract_targets_file)
        wallet_col = "wallet_copied" if "wallet_copied" in df.columns else "trader_wallet" if "trader_wallet" in df.columns else None
        return_col = "future_return" if "future_return" in df.columns else "forward_return_15m" if "forward_return_15m" in df.columns else None
        hit_col = "target_up" if "target_up" in df.columns else "tp_before_sl_60m" if "tp_before_sl_60m" in df.columns else None
        if df.empty or wallet_col is None or return_col is None or "timestamp" not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values([wallet_col, "timestamp"]).reset_index(drop=True)

        rows = []
        for wallet, group in df.groupby(wallet_col):
            group = group.reset_index(drop=True)
            for idx in range(len(group)):
                past = group.iloc[max(0, idx - 200):idx]
                if past.empty:
                    rows.append({"wallet_copied": wallet, "timestamp": group.iloc[idx]["timestamp"]})
                    continue
                rows.append(
                    {
                        "wallet_copied": wallet,
                        "timestamp": group.iloc[idx]["timestamp"],
                        "wallet_trade_count_30d": len(past),
                        "wallet_avg_forward_return_15m": float(past[return_col].mean()),
                        "wallet_winrate_30d": float((past[return_col] > 0).mean()),
                        "wallet_alpha_30d": float(past[return_col].mean()),
                        "wallet_signal_precision_tp": float(past[hit_col].mean()) if hit_col else None,
                        "wallet_recent_streak": int((past[return_col].tail(5) > 0).sum()),
                    }
                )
        history_df = pd.DataFrame(rows)
        if not history_df.empty and "outcome_side" in df.columns and wallet_col in df.columns:
            side_stats = (
                df.groupby([wallet_col, "outcome_side"])[return_col]
                .mean()
                .unstack(fill_value=0.0)
                .reset_index()
                .rename(columns={wallet_col: "wallet_copied", "YES": "yes_side_avg_return", "NO": "no_side_avg_return"})
            )
            history_df = history_df.merge(side_stats, on="wallet_copied", how="left")
        return history_df

    def write(self):
        alpha = self.build()
        history = self.build_history()
        if not alpha.empty:
            alpha.to_csv(self.output_file, index=False)
        if not history.empty:
            history.to_csv(self.history_file, index=False)
        return alpha
