"""Diagnostic script to understand root causes of the 5.1% win rate."""
import pandas as pd
import numpy as np

df = pd.read_csv("logs/closed_positions.csv", engine="python", on_bad_lines="skip")
for col in ["net_realized_pnl", "realized_pnl", "confidence", "entry_price", "exit_price", "size_usdc"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

pnl_col = "net_realized_pnl" if "net_realized_pnl" in df.columns else "realized_pnl"
closed = df[df["status"].astype(str).str.upper() == "CLOSED"].copy() if "status" in df.columns else df.copy()
closed["win"] = closed[pnl_col] > 0

print("=" * 60)
print("TOTAL CLOSED:", len(closed))
print("WIN RATE: {:.1f}%".format(closed["win"].mean() * 100))
print("NET PNL:", round(closed[pnl_col].sum(), 4))

# 1. Family breakdown with median entry price
print("\n=== Market Family ===")
for fam, g in closed.groupby("market_family", dropna=False):
    ep = g["entry_price"].median()
    print("  {:35s} n={:4d}  win={:.1f}%  pnl={:8.2f}  median_entry={:.3f}".format(
        str(fam), len(g), g["win"].mean()*100, g[pnl_col].sum(), ep))

# 2. btc_other: are these short-duration binary markets where entry price was already near 0?
print("\n=== btc_other: entry price distribution ===")
btcother = closed[closed["market_family"].astype(str) == "btc_other"]
print("  Median entry price:", btcother["entry_price"].median())
print("  <0.10 entry:", (btcother["entry_price"] < 0.10).sum())
print("  <0.20 entry:", (btcother["entry_price"] < 0.20).sum())
print("  >0.80 entry:", (btcother["entry_price"] > 0.80).sum())
if "signal_label" in btcother.columns:
    print("\n  Signal labels:")
    print(btcother["signal_label"].value_counts().to_string())

# 3. rl_exit vs take_profit comparison
print("\n=== Exit reason deep-dive ===")
if "close_reason" in closed.columns:
    for reason, g in closed.groupby("close_reason", dropna=False):
        ep = g["entry_price"].median()
        xp = g["exit_price"].median() if "exit_price" in g.columns else float("nan")
        print("  {:35s} n={:4d}  win={:.1f}%  pnl={:8.2f}  entry={:.3f}  exit={:.3f}".format(
            str(reason), len(g), g["win"].mean()*100, g[pnl_col].sum(), ep, xp))

# 4. Confidence → win rate calibration
print("\n=== Confidence thresholds ===")
thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
for thr in thresholds:
    sub = closed[closed["confidence"] >= thr]
    if len(sub) > 0:
        print("  conf >= {:.2f}: n={:4d}  win={:.1f}%  pnl={:8.2f}".format(
            thr, len(sub), sub["win"].mean()*100, sub[pnl_col].sum()))
    else:
        print("  conf >= {:.2f}: no trades".format(thr))

# 5. Profitability by confidence bucket
print("\n=== Confidence buckets ===")
buckets = [0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 1.01]
labels  = ["<0.40","0.40-0.45","0.45-0.50","0.50-0.55","0.55-0.60","0.60-0.70",">0.70"]
closed["cband"] = pd.cut(closed["confidence"], bins=buckets, labels=labels, include_lowest=True)
for band, g in closed.groupby("cband", observed=True):
    print("  {:12s}  n={:4d}  win={:.1f}%  pnl={:8.2f}".format(
        str(band), len(g), g["win"].mean()*100, g[pnl_col].sum()))

# 6. Sample of recent losing rl_exit trades
print("\n=== Sample losing rl_exit trades (last 10) ===")
rl_loses = closed[(closed["close_reason"].astype(str) == "rl_exit") & (~closed["win"])]
cols_to_show = [c for c in ["market_title","entry_price","exit_price",pnl_col,"confidence","signal_label"] if c in rl_loses.columns]
print(rl_loses[cols_to_show].tail(10).to_string())

# 7. What if we filter out btc_other and below 0.45 confidence?
print("\n=== Simulated entry gate: exclude btc_other + conf < 0.45 ===")
filtered = closed[
    (closed["market_family"].astype(str) != "btc_other") &
    (closed["confidence"] >= 0.45)
]
if len(filtered):
    print("  Surviving trades: {:d}".format(len(filtered)))
    print("  Win rate: {:.1f}%".format(filtered["win"].mean()*100))
    print("  Net PnL: {:.2f}".format(filtered[pnl_col].sum()))
else:
    print("  No trades survive filter")
