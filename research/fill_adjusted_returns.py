"""
research/fill_adjusted_returns.py

Reads logs/closed_positions.csv and computes fill-adjusted PnL:
  raw_return   = (exit_price - entry_price) / entry_price  (mid-price fantasy)
  slippage_adj = realized slippage from exit_realized_slippage_bps
  net_return   = raw minus cost

Also breaks down by market_family, close_reason, and signal_label.

Run:
    python research/fill_adjusted_returns.py
    python research/fill_adjusted_returns.py --logs logs --out logs/research/fill_adjusted_returns.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _pct(num: float, denom: float) -> str:
    if denom == 0:
        return "—"
    return f"{num/denom*100:.1f}%"


def _fmt(v, fmt=".4f") -> str:
    try:
        return format(float(v), fmt)
    except Exception:
        return "—"


def generate_fill_adjusted_report(logs_dir: Path, out_md: Path) -> None:
    path = logs_dir / "closed_positions.csv"
    df = _safe_read(path)

    lines: list[str] = [
        "# Fill-Adjusted Returns Report\n",
        f"_Source: `{path}`_\n",
        "This report compares mid-price (fantasy) returns against actual cost-adjusted returns "
        "to show where slippage and execution drag erode edge.\n",
        "---\n",
    ]

    if df.empty:
        lines.append("> ⚠️  `closed_positions.csv` not found.\n")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines), encoding="utf-8")
        print(f"✅  Wrote (empty) fill-adjusted report → {out_md}")
        return

    # Normalise numeric columns
    for col in ["entry_price", "exit_price", "realized_pnl", "net_realized_pnl",
                "exit_realized_slippage_bps", "size_usdc", "confidence"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Only closed trades with both prices
    closed = df[df["status"].astype(str).str.upper() == "CLOSED"].copy() if "status" in df.columns else df.copy()
    closed = closed.dropna(subset=["entry_price", "realized_pnl"])

    if closed.empty:
        lines.append("> ⚠️  No closed positions with complete data found.\n")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines), encoding="utf-8")
        print(f"✅  Wrote (empty) fill-adjusted report → {out_md}")
        return

    # Compute per-trade metrics
    closed["raw_return"] = closed["realized_pnl"]  # already in USDC
    slip_bps = closed["exit_realized_slippage_bps"].fillna(0.0) if "exit_realized_slippage_bps" in closed.columns else pd.Series(0.0, index=closed.index)
    size = closed["size_usdc"].fillna(0.0) if "size_usdc" in closed.columns else pd.Series(0.0, index=closed.index)
    closed["slippage_usdc"] = slip_bps * size / 10000.0
    closed["net_return_usdc"] = closed["net_realized_pnl"] if "net_realized_pnl" in closed.columns else closed["realized_pnl"]
    closed["win"] = closed["net_return_usdc"] > 0

    total = len(closed)
    wins = closed["win"].sum()
    raw_pnl = closed["raw_return"].sum()
    net_pnl = closed["net_return_usdc"].sum()
    slip_total = closed["slippage_usdc"].sum()
    avg_slip_bps = slip_bps[slip_bps > 0].mean()

    lines += [
        "## Overall Summary\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Closed trades | {total} |",
        f"| Win rate | {_pct(wins, total)} |",
        f"| Gross realized PnL (USDC) | {_fmt(raw_pnl)} |",
        f"| Net realized PnL (USDC) | {_fmt(net_pnl)} |",
        f"| Total slippage drag (USDC) | {_fmt(slip_total)} |",
        f"| Average exit slippage (bps) | {_fmt(avg_slip_bps)} |",
        f"| Edge erosion from slippage | {_pct(abs(slip_total), abs(raw_pnl) if raw_pnl != 0 else 1)} |",
        "",
    ]

    # --- By market family ---
    lines.append("## By Market Family\n")
    if "market_family" in closed.columns:
        lines += [
            "| Family | Trades | Win Rate | Gross PnL | Net PnL | Avg Slip (bps) |",
            "|--------|--------|----------|-----------|---------|----------------|",
        ]
        for fam, grp in closed.groupby("market_family", dropna=False):
            g_slip = grp["slippage_usdc"].mean() * 10000 / (grp["size_usdc"].mean() or 1) if "size_usdc" in grp.columns else 0
            bps_val = slip_bps.loc[grp.index]
            avg_bps = bps_val[bps_val > 0].mean()
            lines.append(
                f"| `{fam}` | {len(grp)} | {_pct(grp['win'].sum(), len(grp))} "
                f"| {_fmt(grp['raw_return'].sum())} "
                f"| {_fmt(grp['net_return_usdc'].sum())} "
                f"| {_fmt(avg_bps)} |"
            )
        lines.append("")

    # --- By close reason ---
    lines.append("## By Close Reason\n")
    reason_col = next((c for c in ["close_reason", "exit_reason_family", "intended_exit_reason"] if c in closed.columns), None)
    if reason_col:
        lines += [
            f"| {reason_col} | Trades | Win Rate | Net PnL |",
            f"|{'---'*3}|--------|----------|---------|",
        ]
        for reason, grp in closed.groupby(reason_col, dropna=False):
            lines.append(
                f"| `{reason}` | {len(grp)} | {_pct(grp['win'].sum(), len(grp))} "
                f"| {_fmt(grp['net_return_usdc'].sum())} |"
            )
        lines.append("")

    # --- By confidence bucket ---
    lines.append("## By Model Confidence Bucket\n")
    if "confidence" in closed.columns and closed["confidence"].notna().any():
        conf = closed["confidence"].dropna()
        closed["conf_bucket"] = pd.cut(conf, bins=[0, 0.45, 0.55, 0.65, 0.75, 1.01],
                                       labels=["<0.45", "0.45–0.55", "0.55–0.65", "0.65–0.75", ">0.75"],
                                       include_lowest=True)
        lines += [
            "| Confidence | Trades | Win Rate | Net PnL | Avg Slip (bps) |",
            "|------------|--------|----------|---------|----------------|",
        ]
        for bucket, grp in closed.groupby("conf_bucket", observed=True):
            avg_bps = slip_bps.loc[grp.index]
            avg_bps_v = avg_bps[avg_bps > 0].mean()
            lines.append(
                f"| {bucket} | {len(grp)} | {_pct(grp['win'].sum(), len(grp))} "
                f"| {_fmt(grp['net_return_usdc'].sum())} "
                f"| {_fmt(avg_bps_v)} |"
            )
        lines.append("")

    # --- High-slippage outliers ---
    lines.append("## High-Slippage Outliers (top 10)\n")
    hi_slip = closed[slip_bps > 0].copy()
    hi_slip["slip_bps"] = slip_bps.loc[hi_slip.index]
    hi_slip = hi_slip.nlargest(10, "slip_bps")
    if not hi_slip.empty:
        lines += [
            "| Market | Exit Slip (bps) | Net PnL | Close Reason |",
            "|--------|-----------------|---------|--------------|",
        ]
        title_col = next((c for c in ["market_title", "market", "market_slug"] if c in hi_slip.columns), None)
        for _, row in hi_slip.iterrows():
            title = str(row[title_col])[:60] if title_col else "—"
            cr    = str(row.get(reason_col, "—")) if reason_col else "—"
            lines.append(f"| {title} | {_fmt(row['slip_bps'], '.0f')} | {_fmt(row['net_return_usdc'])} | {cr} |")
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅  Wrote fill-adjusted returns → {out_md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fill-adjusted returns report")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out",  default="logs/research/fill_adjusted_returns.md")
    args = ap.parse_args()
    generate_fill_adjusted_report(Path(args.logs), Path(args.out))
