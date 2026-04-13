"""Determine optimal entry price floor based on actual win rates."""
import pandas as pd

df = pd.read_csv("logs/closed_positions.csv", engine="python", on_bad_lines="skip")
for col in ["net_realized_pnl", "confidence", "entry_price", "exit_price"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

pnl_col = "net_realized_pnl"
closed = df[df["status"].astype(str).str.upper() == "CLOSED"].copy()
closed["win"] = closed[pnl_col] > 0

# Exclude btc_other (already agreed to block it), apply conf >= 0.45
base = closed[
    (closed["market_family"].astype(str) != "btc_other") &
    (closed["confidence"] >= 0.45)
].copy()

print("Base (no btc_other, conf>=0.45): n={}, win={:.1f}%, pnl={:.2f}".format(
    len(base), base["win"].mean()*100, base[pnl_col].sum()))

print("\n=== Price floor sensitivity (entry price >= floor) ===")
print("{:>10}  {:>6}  {:>9}  {:>9}  {:>9}".format(
    "Floor", "Trades", "Win Rate", "Net PnL", "Avg PnL"))
print("-" * 55)
for floor in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40]:
    sub = base[base["entry_price"] >= floor]
    if len(sub):
        avg = sub[pnl_col].mean()
        print("{:>10.2f}  {:>6d}  {:>8.1f}%  {:>9.2f}  {:>9.4f}".format(
            floor, len(sub), sub["win"].mean()*100, sub[pnl_col].sum(), avg))

print("\n=== Price ceiling sensitivity (entry price <= ceiling) ===")
print("{:>10}  {:>6}  {:>9}  {:>9}".format("Ceiling", "Trades", "Win Rate", "Net PnL"))
print("-" * 45)
for ceil in [0.90, 0.87, 0.85, 0.82, 0.80, 0.75]:
    sub = base[base["entry_price"] <= ceil]
    if len(sub):
        print("{:>10.2f}  {:>6d}  {:>8.1f}%  {:>9.2f}".format(
            ceil, len(sub), sub["win"].mean()*100, sub[pnl_col].sum()))

print("\n=== Combined floor + ceiling sweep ===")
print("{:>8}  {:>8}  {:>6}  {:>9}  {:>9}".format(
    "Floor", "Ceiling", "Trades", "Win Rate", "Net PnL"))
print("-" * 50)
for floor in [0.15, 0.20, 0.25]:
    for ceil in [0.85, 0.87, 0.90]:
        sub = base[(base["entry_price"] >= floor) & (base["entry_price"] <= ceil)]
        if len(sub):
            print("{:>8.2f}  {:>8.2f}  {:>6d}  {:>8.1f}%  {:>9.2f}".format(
                floor, ceil, len(sub), sub["win"].mean()*100, sub[pnl_col].sum()))

print("\n=== Entry price distribution for winning trades ===")
winners = base[base["win"]]
print("  n wins:", len(winners))
if len(winners):
    print("  Min entry:", winners["entry_price"].min())
    print("  Max entry:", winners["entry_price"].max())
    print("  Median entry:", winners["entry_price"].median())
    hist = pd.cut(winners["entry_price"], bins=[0, 0.15, 0.20, 0.30, 0.50, 0.70, 0.85, 1.01],
                  labels=["<0.15","0.15-0.20","0.20-0.30","0.30-0.50","0.50-0.70","0.70-0.85",">0.85"])
    print(hist.value_counts().sort_index().to_string())

print("\n=== Entry price distribution for LOSING trades ===")
losers = base[~base["win"]]
print("  n losses:", len(losers))
if len(losers):
    hist2 = pd.cut(losers["entry_price"], bins=[0, 0.15, 0.20, 0.30, 0.50, 0.70, 0.85, 1.01],
                   labels=["<0.15","0.15-0.20","0.20-0.30","0.30-0.50","0.50-0.70","0.70-0.85",">0.85"])
    print(hist2.value_counts().sort_index().to_string())
