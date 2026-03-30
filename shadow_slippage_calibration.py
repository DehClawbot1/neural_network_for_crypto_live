import os
import pandas as pd

from config import TradingConfig


def run_slippage_calibration(log_path="logs/shadow_results.csv"):
    if not os.path.exists(log_path):
        print("❌ Shadow log not found.")
        return

    df = pd.read_csv(log_path, engine="python", on_bad_lines="skip")
    active = df[df["outcome"] != "DOA"].dropna(subset=["expected_slip_bps", "entry_slippage_bps"]).copy() if "outcome" in df.columns else pd.DataFrame()

    if len(active) < 5:
        print(f"⏳ Only {len(active)} samples found. Need more resolutions for calibration.")
        return

    active["signed_error"] = active["entry_slippage_bps"] - active["expected_slip_bps"]
    active["abs_error"] = active["signed_error"].abs()

    mae = active["abs_error"].mean()
    med_ae = active["abs_error"].median()
    mse = active["signed_error"].mean()

    print("-" * 50)
    print(f"🎯 SLIPPAGE CALIBRATION AUDIT (N={len(active)})")
    print("-" * 50)
    print(f"📏 Mean Absolute Error: {mae:.1f} BPS")
    print(f"📏 Median Abs Error: {med_ae:.1f} BPS")
    print(f"⚖️ Mean Signed Error: {mse:+.1f} BPS")

    active["prob_bucket"] = pd.cut(active["meta_prob"], bins=TradingConfig.PROB_BUCKETS)
    bucket_bias = active.groupby("prob_bucket")["signed_error"].mean()

    print("\n--- 🪣 BUCKET-LEVEL BIAS (Actual - Predicted) ---")
    for bucket, bias in bucket_bias.items():
        label = "UNDERESTIMATING" if bias > 0 else "OVERESTIMATING"
        print(f"{str(bucket):<15} | Bias: {bias:+.1f} BPS | {label}")

    total_intents = len(df)
    doa_count = len(df[df["outcome"] == "DOA"]) if "outcome" in df.columns else 0
    print(f"\n--- 🚫 VETO INTEGRITY ---")
    print(f"Total Intents: {total_intents}")
    print(f"DOA (Vetoed): {doa_count} ({(doa_count / total_intents):.1%})" if total_intents else "DOA (Vetoed): 0 (0.0%)")

    threshold = TradingConfig.CALIBRATION_BIAS_THRESHOLD
    if mse > threshold:
        print(f"\n⚠️ CRITICAL: Systematic Underestimation (> {threshold} BPS).")
    elif mse < -threshold:
        print(f"\n⚠️ CAUTION: Systematic Overestimation (< -{threshold} BPS).")
    else:
        print(f"\n✅ Healthy Calibration: Bias is within +/- {threshold} BPS.")


if __name__ == "__main__":
    run_slippage_calibration()
