from pathlib import Path

import numpy as np
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

        # ── BUG FIX (BUG 2): Use vectorized rolling/expanding windows instead
        #    of O(n²) per-row slicing. This reduces wallet alpha history
        #    from ~30-60 min to seconds. ──

        # Ensure numeric types for rolling computations
        df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
        df["_return_positive"] = (df[return_col] > 0).astype(float)
        if hit_col and hit_col in df.columns:
            df[hit_col] = pd.to_numeric(df[hit_col], errors="coerce")

        parts = []
        for wallet, group in df.groupby(wallet_col):
            group = group.reset_index(drop=True)
            n = len(group)
            if n == 0:
                continue

            # Use expanding window (capped at 200 via min_periods=1, window=200 rolling)
            # expanding() is equivalent to rolling with window=len, but rolling(200, min_periods=1)
            # gives us the "last 200 rows" behavior efficiently
            window = min(200, n)
            roll = group[return_col].rolling(window=window, min_periods=1)

            wallet_rows = pd.DataFrame({
                "wallet_copied": wallet,
                "timestamp": group["timestamp"].values,
                "wallet_trade_count_30d": range(n),  # count of prior rows
                "wallet_avg_forward_return_15m": roll.mean().shift(1).values,
                "wallet_winrate_30d": group["_return_positive"].rolling(window=window, min_periods=1).mean().shift(1).values,
                "wallet_alpha_30d": roll.mean().shift(1).values,
                "wallet_recent_streak": group["_return_positive"].rolling(window=5, min_periods=1).sum().shift(1).values,
            })

            if hit_col and hit_col in group.columns:
                wallet_rows["wallet_signal_precision_tp"] = group[hit_col].rolling(window=window, min_periods=1).mean().shift(1).values
            else:
                wallet_rows["wallet_signal_precision_tp"] = np.nan

            # YES/NO side avg returns (vectorized)
            if "outcome_side" in group.columns:
                yes_mask = group["outcome_side"].astype(str).str.upper() == "YES"
                no_mask = group["outcome_side"].astype(str).str.upper() == "NO"
                # For side-specific averages, use expanding mean (simpler, still fast)
                yes_returns = group[return_col].where(yes_mask).expanding(min_periods=1).mean().shift(1)
                no_returns = group[return_col].where(no_mask).expanding(min_periods=1).mean().shift(1)
                wallet_rows["yes_side_avg_return"] = yes_returns.values
                wallet_rows["no_side_avg_return"] = no_returns.values
            else:
                wallet_rows["yes_side_avg_return"] = 0.0
                wallet_rows["no_side_avg_return"] = 0.0

            parts.append(wallet_rows)

        if not parts:
            return pd.DataFrame()

        history_df = pd.concat(parts, ignore_index=True)
        return history_df

    def write(self):
        alpha = self.build()
        history = self.build_history()
        if not alpha.empty:
            alpha.to_csv(self.output_file, index=False)
        if not history.empty:
            history.to_csv(self.history_file, index=False)
        return alpha
