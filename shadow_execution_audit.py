import os
import pandas as pd


def run_execution_audit(log_path="logs/shadow_results.csv", limit=20):
    if not os.path.exists(log_path):
        print("❌ Shadow log not found.")
        return

    df = pd.read_csv(log_path, engine="python", on_bad_lines="skip")
    resolved = df[df["outcome"] != "PENDING"].head(limit).copy() if "outcome" in df.columns else pd.DataFrame()

    if len(resolved) < 1:
        print("⏳ No resolved trades to audit yet. Purgatory is still processing...")
        return

    resolved["slippage_pct"] = resolved["entry_slippage_bps"].fillna(0) / 10000
    resolved["theoretical_return"] = resolved["realized_return"].fillna(0) + resolved["slippage_pct"]

    theoretical_win_rate = (resolved["theoretical_return"] >= 0.04).mean()
    actual_win_rate = (resolved["outcome"] == "TP").mean()

    avg_slippage = resolved["entry_slippage_bps"].mean() if "entry_slippage_bps" in resolved.columns else 0.0
    avg_delay = resolved["entry_delay_sec"].mean() if "entry_delay_sec" in resolved.columns else 0.0
    thin_markets = (resolved["trades_in_window"].fillna(0) < 5).sum() if "trades_in_window" in resolved.columns else 0

    mix = resolved["outcome"].value_counts(normalize=True) * 100

    print("-" * 50)
    print(f"📊 SHADOW EXECUTION AUDIT (First {len(resolved)} Trades)")
    print("-" * 50)
    print(f"🏆 Win Rate (Theoretical): {theoretical_win_rate:.2%}")
    print(f"👻 Win Rate (Adjusted): {actual_win_rate:.2%}")
    print(f"📉 Execution Tax (Gap): {theoretical_win_rate - actual_win_rate:.2%}")

    print("\n--- 🏎️ LATENCY & SLIPPAGE ---")
    print(f"⏱️ Avg Entry Delay: {avg_delay:.2f}s")
    print(f"💸 Avg Slippage: {avg_slippage:.1f} BPS")
    print(f"🧊 Thin Market Rows: {thin_markets} (trades_in_window < 5)")

    print("\n--- 📈 OUTCOME MIX ---")
    for outcome, pct in mix.items():
        print(f"{outcome:.<15} {pct:.1f}%")

    toxic_trades = resolved[resolved["entry_slippage_bps"].fillna(0) > 100] if "entry_slippage_bps" in resolved.columns else pd.DataFrame()
    if not toxic_trades.empty:
        print(f"\n⚠️ WARNING: {len(toxic_trades)} trades had >100bps slippage.")
        print("Check if high-conviction whales are clearing the book before you can enter.")

    print("-" * 50)


if __name__ == "__main__":
    run_execution_audit()
