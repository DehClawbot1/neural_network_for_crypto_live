"""
research/regime_dependence_eval.py

Reads logs/closed_positions.csv + logs/technical_regime_snapshot.csv and
shows how edge, win rate, and net PnL vary across BTC market regimes and
market families. The goal is to prove the model isn't only profitable in
one regime (data snooping red flag).

Run:
    python research/regime_dependence_eval.py
    python research/regime_dependence_eval.py --logs logs --out logs/research/regime_dependence.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _pct(num, denom) -> str:
    try:
        return f"{float(num) / float(denom) * 100:.1f}%" if denom else "—"
    except Exception:
        return "—"


def _fmt(v, fmt=".4f") -> str:
    try:
        return format(float(v), fmt)
    except Exception:
        return "—"


def _pf(wins, losses) -> str:
    """Profit factor = gross_wins / gross_losses."""
    try:
        g = float(wins)
        l = abs(float(losses))
        return f"{g/l:.2f}" if l > 0 else ("∞" if g > 0 else "—")
    except Exception:
        return "—"


def generate_regime_report(logs_dir: Path, out_md: Path) -> None:
    pos_path    = logs_dir / "closed_positions.csv"
    regime_path = logs_dir / "technical_regime_snapshot.csv"

    closed = _safe_read(pos_path)
    regime = _safe_read(regime_path)

    lines: list[str] = [
        "# Regime Dependence Evaluation\n",
        f"_Source: `{pos_path}` + `{regime_path}`_\n",
        "Shows how model edge varies across BTC market conditions.\n",
        "A strategy that only works in one regime is fragile — this report exposes that.\n",
        "---\n",
    ]

    if closed.empty:
        lines.append("> ⚠️  `closed_positions.csv` not found.\n")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines), encoding="utf-8")
        print(f"✅  Wrote (empty) regime report → {out_md}")
        return

    # Numeric normalisation
    for col in ["realized_pnl", "net_realized_pnl", "entry_price", "exit_price", "confidence"]:
        if col in closed.columns:
            closed[col] = pd.to_numeric(closed[col], errors="coerce")

    closed = closed[closed["status"].astype(str).str.upper() == "CLOSED"].copy() if "status" in closed.columns else closed.copy()
    closed = closed.dropna(subset=["realized_pnl"])
    pnl_col = "net_realized_pnl" if "net_realized_pnl" in closed.columns else "realized_pnl"
    closed["win"] = closed[pnl_col] > 0

    def _group_stats(grp: pd.DataFrame) -> dict:
        n     = len(grp)
        wins  = grp["win"].sum()
        gross_w = grp.loc[grp["win"],    pnl_col].sum()
        gross_l = grp.loc[~grp["win"],   pnl_col].sum()
        net   = grp[pnl_col].sum()
        return dict(n=n, win_rate=wins/n if n else 0, gross_w=gross_w,
                    gross_l=gross_l, net=net)

    # --- By market family ---
    lines.append("## By Market Family\n")
    if "market_family" in closed.columns:
        lines += [
            "| Family | Trades | Win Rate | Net PnL | Profit Factor |",
            "|--------|--------|----------|---------|---------------|",
        ]
        for fam, grp in closed.groupby("market_family", dropna=False):
            s = _group_stats(grp)
            lines.append(f"| `{fam}` | {s['n']} | {_pct(s['n']*s['win_rate'], s['n'])} "
                         f"| {_fmt(s['net'])} | {_pf(s['gross_w'], s['gross_l'])} |")
        lines.append("")

    # --- By BTC regime label (from closed_positions if available) ---
    regime_label_col = next(
        (c for c in ["btc_market_regime_label", "active_regime", "technical_regime_bucket", "btc_volatility_regime"]
         if c in closed.columns), None
    )
    lines.append("## By BTC Market Regime\n")
    if regime_label_col:
        lines += [
            f"_(regime from `{regime_label_col}`)_\n",
            "| Regime | Trades | Win Rate | Net PnL | Profit Factor |",
            "|--------|--------|----------|---------|---------------|",
        ]
        for reg, grp in closed.groupby(regime_label_col, dropna=False):
            s = _group_stats(grp)
            lines.append(f"| `{reg}` | {s['n']} | {_pct(s['n']*s['win_rate'], s['n'])} "
                         f"| {_fmt(s['net'])} | {_pf(s['gross_w'], s['gross_l'])} |")
        lines.append("")
    else:
        lines.append("_No regime label column in closed_positions.csv. "
                     "Join with technical_regime_snapshot.csv to populate._\n")

    # --- Time-series regime join (if timestamp available) ---
    if not regime.empty and "timestamp" in closed.columns and "timestamp" in regime.columns:
        try:
            closed["timestamp"] = pd.to_datetime(closed["timestamp"], utc=True, errors="coerce")
            regime["timestamp"] = pd.to_datetime(regime["timestamp"], utc=True, errors="coerce")
            regime = regime.dropna(subset=["timestamp"]).sort_values("timestamp")
            regime_label = next(
                (c for c in ["btc_market_regime_label", "btc_volatility_regime", "market_structure"]
                 if c in regime.columns), None
            )
            if regime_label and closed["timestamp"].notna().any():
                joined = pd.merge_asof(
                    closed.dropna(subset=["timestamp"]).sort_values("timestamp"),
                    regime[["timestamp", regime_label]].rename(columns={regime_label: "_joined_regime"}),
                    on="timestamp", direction="backward", tolerance=pd.Timedelta("12h"),
                )
                lines.append("## Regime Breakdown (time-matched from regime snapshot)\n")
                lines += [
                    f"_(matched `{regime_label}` with ±12h tolerance)_\n",
                    "| Regime | Trades | Win Rate | Net PnL | Profit Factor |",
                    "|--------|--------|----------|---------|---------------|",
                ]
                for reg, grp in joined.groupby("_joined_regime", dropna=True):
                    s = _group_stats(grp)
                    lines.append(f"| `{reg}` | {s['n']} | {_pct(s['n']*s['win_rate'], s['n'])} "
                                 f"| {_fmt(s['net'])} | {_pf(s['gross_w'], s['gross_l'])} |")
                lines.append("")
        except Exception as e:
            lines.append(f"_Regime join failed: {e}_\n")

    # --- By signal label ---
    lines.append("## By Signal Label\n")
    if "signal_label" in closed.columns:
        lines += [
            "| Signal Label | Trades | Win Rate | Net PnL |",
            "|--------------|--------|----------|---------|",
        ]
        for label, grp in closed.groupby("signal_label", dropna=False):
            s = _group_stats(grp)
            lines.append(f"| `{label}` | {s['n']} | {_pct(s['n']*s['win_rate'], s['n'])} "
                         f"| {_fmt(s['net'])} |")
        lines.append("")

    # --- Monotonicity check: does confidence predict win rate? ---
    lines.append("## Monotonicity Check: Model Confidence vs Win Rate\n")
    lines.append("_A well-calibrated model should show higher win rates at higher confidence._\n")
    if "confidence" in closed.columns and closed["confidence"].notna().any():
        bins = [0, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
        labels = ["<0.40", "0.40–0.50", "0.50–0.55", "0.55–0.60", "0.60–0.65", "0.65–0.70", ">0.70"]
        closed["conf_bucket"] = pd.cut(closed["confidence"], bins=bins, labels=labels, include_lowest=True)
        lines += [
            "| Confidence | Trades | Win Rate | Net PnL |",
            "|------------|--------|----------|---------|",
        ]
        for bucket, grp in closed.groupby("conf_bucket", observed=True):
            s = _group_stats(grp)
            lines.append(f"| {bucket} | {s['n']} | {_pct(s['n']*s['win_rate'], s['n'])} "
                         f"| {_fmt(s['net'])} |")
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅  Wrote regime dependence report → {out_md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Regime dependence evaluation report")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out",  default="logs/research/regime_dependence.md")
    args = ap.parse_args()
    generate_regime_report(Path(args.logs), Path(args.out))
