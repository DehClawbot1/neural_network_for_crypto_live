import os
import pandas as pd
import scipy.stats as stats

from config import TradingConfig


def run_impact_correlation(df):
    resolved = df[df['outcome'] != "PENDING"].copy()
    if len(resolved) < 10:
        print("📊 Correlation Audit: Waiting for N >= 10 for statistical relevance.")
        return

    metrics = [
        ('meta_prob', 'Symmetry Check'),
        ('whale_trade_size', 'Liquidity Depth Check'),
        ('normalized_trade_size', 'Market Impact Check'),
    ]

    print("\n--- 🕵️ WHALE IMPACT CORRELATIONS (Spearman Rho) ---")
    for col, description in metrics:
        if col in resolved.columns:
            rho, p_val = stats.spearmanr(resolved[col], resolved['entry_slippage_bps'])
            significance = "✅" if p_val > 0.05 else "⚠️"
            print(f"{col:<22} | Rho: {rho:+.3f} | P-Val: {p_val:.3f} | {significance} {description}")

    high_prob = resolved[resolved['meta_prob'] > 0.90]
    low_prob = resolved[resolved['meta_prob'] < 0.70]
    high_prob_slip = high_prob['entry_slippage_bps'].mean() if not high_prob.empty else float('nan')
    low_prob_slip = low_prob['entry_slippage_bps'].mean() if not low_prob.empty else float('nan')
    print(f"\n💡 Conviction Tax: Avg Slippage for Prob > 0.90 is {high_prob_slip:.1f} BPS")
    print(f"💡 Noise Tax: Avg Slippage for Prob < 0.70 is {low_prob_slip:.1f} BPS")


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

    theoretical_win_rate = (resolved["theoretical_return"] >= TradingConfig.SHADOW_TP_DELTA).mean()
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
    run_impact_correlation(resolved)


if __name__ == "__main__":
    run_execution_audit()
