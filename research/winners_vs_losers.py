"""
Deeper analysis of the 41 filtered trades (no btc_other, conf>=0.45).
All entries are 0.50-0.70. All 13 winners are at exactly 0.64.
Find what differentiates winners from losers.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("logs/closed_positions.csv", engine="python", on_bad_lines="skip")
for col in df.select_dtypes(include="object").columns:
    pass  # keep strings
for col in ["net_realized_pnl", "confidence", "entry_price", "exit_price",
            "size_usdc", "max_adverse_excursion_pct", "max_favorable_excursion_pct",
            "drawdown_from_peak", "runup_from_entry"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

pnl_col = "net_realized_pnl"
closed = df[df["status"].astype(str).str.upper() == "CLOSED"].copy()
closed["win"] = closed[pnl_col] > 0

filtered = closed[
    (closed["market_family"].astype(str) != "btc_other") &
    (closed["confidence"] >= 0.45)
].copy()

print("=" * 70)
print("FILTERED SET: n={}, win={:.1f}%, pnl={:.2f}".format(
    len(filtered), filtered["win"].mean()*100, filtered[pnl_col].sum()))

# 1. Family breakdown
print("\n=== By market family ===")
for fam, g in filtered.groupby("market_family", dropna=False):
    print("  {:35s}  n={:3d}  win={:.1f}%  pnl={:.2f}".format(
        str(fam), len(g), g["win"].mean()*100, g[pnl_col].sum()))

# 2. Close reason
print("\n=== By close reason ===")
if "close_reason" in filtered.columns:
    for reason, g in filtered.groupby("close_reason", dropna=False):
        print("  {:30s}  n={:3d}  win={:.1f}%  pnl={:8.2f}  entry={:.3f}".format(
            str(reason), len(g), g["win"].mean()*100,
            g[pnl_col].sum(), g["entry_price"].median()))

# 3. Signal label
print("\n=== By signal label ===")
if "signal_label" in filtered.columns:
    for label, g in filtered.groupby("signal_label", dropna=False):
        print("  {:35s}  n={:3d}  win={:.1f}%  pnl={:.2f}".format(
            str(label), len(g), g["win"].mean()*100, g[pnl_col].sum()))

# 4. Market title patterns for winners
print("\n=== Winner market titles ===")
winners = filtered[filtered["win"]]
title_col = next((c for c in ["market_title","market","market_slug"] if c in filtered.columns), None)
if title_col:
    for t in winners[title_col].dropna().unique():
        print("  WIN:", str(t)[:80])

print("\n=== Loser market titles (sample) ===")
losers = filtered[~filtered["win"]]
if title_col:
    for t in losers[title_col].dropna().head(10):
        print("  LOSE:", str(t)[:80])

# 5. BTC forecast features at entry
print("\n=== BTC forecast at entry (winners vs losers) ===")
btc_cols = [c for c in ["entry_btc_predicted_direction","entry_btc_predicted_return",
                          "entry_btc_forecast_confidence","entry_btc_mtf_agreement"] if c in filtered.columns]
for col in btc_cols:
    fcol = pd.to_numeric(filtered[col], errors="coerce")
    w = pd.to_numeric(winners[col], errors="coerce").mean()
    l = pd.to_numeric(losers[col], errors="coerce").mean()
    print("  {:40s}  winners={:.3f}  losers={:.3f}".format(col, w, l))

# 6. Outcome side
print("\n=== By outcome side ===")
if "outcome_side" in filtered.columns:
    for side, g in filtered.groupby("outcome_side", dropna=False):
        print("  {:10s}  n={:3d}  win={:.1f}%  pnl={:.2f}".format(
            str(side), len(g), g["win"].mean()*100, g[pnl_col].sum()))

# 7. Entry price bands
print("\n=== Entry price bands (winners vs losers) ===")
print("  Winners: min={:.3f}  max={:.3f}  med={:.3f}".format(
    winners["entry_price"].min(), winners["entry_price"].max(), winners["entry_price"].median()))
print("  Losers:  min={:.3f}  max={:.3f}  med={:.3f}".format(
    losers["entry_price"].min(), losers["entry_price"].max(), losers["entry_price"].median()))

# 8. Simulation: action_code >= 2 filter using signal_label tiers
print("\n=== Simulation: only STRONG PAPER or HIGHEST signals ===")
if "signal_label" in filtered.columns:
    strong = filtered[filtered["signal_label"].isin(
        ["STRONG PAPER OPPORTUNITY", "HIGHEST-RANKED PAPER SIGNAL"])]
    print("  Trades: {}  Win rate: {:.1f}%  PnL: {:.2f}".format(
        len(strong), strong["win"].mean()*100 if len(strong) else 0, strong[pnl_col].sum()))

# 9. Does size affect outcome?
print("\n=== Position size vs outcomes ===")
if "size_usdc" in filtered.columns:
    print("  Winners median size: {:.4f}".format(winners["size_usdc"].median()))
    print("  Losers  median size: {:.4f}".format(losers["size_usdc"].median()))

# 10. Key question: what tags the take_profit vs rl_exit for the same markets?
print("\n=== take_profit_roi vs rl_exit in filtered set ===")
if "close_reason" in filtered.columns:
    tp = filtered[filtered["close_reason"] == "take_profit_roi"]
    rl = filtered[filtered["close_reason"] == "rl_exit"]
    print("  take_profit_roi: n={}, win={:.1f}%, pnl={:.2f}".format(
        len(tp), tp["win"].mean()*100 if len(tp) else 0, tp[pnl_col].sum()))
    print("  rl_exit:         n={}, win={:.1f}%, pnl={:.2f}".format(
        len(rl), rl["win"].mean()*100 if len(rl) else 0, rl[pnl_col].sum()))
    
    # Can we identify if the rl_exit trades were profitable at peak?
    if "max_favorable_excursion_pct" in rl.columns:
        mfe = pd.to_numeric(rl["max_favorable_excursion_pct"], errors="coerce")
        print("  rl_exit max_favorable_excursion: mean={:.3f}  median={:.3f}".format(
            mfe.mean(), mfe.median()))
        if "max_adverse_excursion_pct" in rl.columns:
            mae = pd.to_numeric(rl["max_adverse_excursion_pct"], errors="coerce")
            print("  rl_exit max_adverse_excursion:   mean={:.3f}  median={:.3f}".format(
                mae.mean(), mae.median()))
