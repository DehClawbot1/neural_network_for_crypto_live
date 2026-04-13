"""
research/feature_importance_report.py

Reads logs/feature_importance.csv (feature, importance) produced by stage1_models.py
and logs/feature_ablation_report.csv (ablation per feature group).
Generates a human-readable markdown audit of what the model actually uses.

Run:
    python research/feature_importance_report.py
    python research/feature_importance_report.py --logs logs --out logs/research/feature_importance.md
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Economic rationale catalogue
# ---------------------------------------------------------------------------
# Every predictive feature should have a one-line economic justification.
# Features NOT in this dict are flagged as "unexplained" in the report.
ECONOMIC_RATIONALE: dict[str, str] = {
    "trader_win_rate":                "Wallet copy-trading alpha signal — past resolution rate",
    "wallet_trade_count_30d":         "Activity proxy — high-volume wallets are more committed",
    "wallet_alpha_30d":               "Raw edge realized by copied wallet over 30d",
    "wallet_avg_forward_return_15m":  "Wallet's 15-min forward return — short-term predictive edge",
    "wallet_signal_precision_tp":     "Wallet take-profit hit rate vs stop-loss",
    "wallet_recent_streak":           "Recent consecutive wins — momentum in wallet skill",
    "normalized_trade_size":          "Position sizing relative to wallet — conviction proxy",
    "whale_pressure":                 "Order-flow imbalance from large wallets — liquidity shock",
    "current_price":                  "Yes-token market price = market implied probability",
    "time_left":                      "Options-style time decay: less time = less uncertainty",
    "liquidity_score":                "Thin liquidity → wider spreads → harder to exit profitably",
    "volume_score":                   "Volume confirms price discovery quality",
    "probability_momentum":           "Recent price drift = crowd opinion shift",
    "volatility_score":               "Higher vol → wider distribution → worse EV when wrong",
    "market_structure_score":         "Orderbook depth / bid-ask shape = residual edge proxy",
    "trend_score":                    "BTC trend confluence with trade direction",
    "btc_atr_pct_15m":                "BTC volatility regime — affects fill quality and slippage",
    "btc_realized_vol_1h":            "Realized BTC vol — sets stop-loss width appropriately",
    "btc_realized_vol_4h":            "Longer vol window — regime shift detection",
    "btc_volatility_regime_score":    "ML vol regime classifier output — direct regime allocation",
    "btc_trend_persistence":          "How long the current BTC trend has extended",
    "btc_rsi_14":                     "RSI overbought/oversold — classic mean-reversion signal",
    "btc_rsi_distance_mid":           "Distance from RSI neutral (50) — extremism measure",
    "btc_rsi_divergence_score":       "Price/RSI divergence — trend exhaustion signal",
    "btc_macd":                       "MACD oscillator — momentum direction indicator",
    "btc_macd_signal":                "MACD signal line — lagged trend confirmation",
    "btc_macd_hist":                  "MACD histogram — acceleration of momentum",
    "btc_macd_hist_slope":            "Histogram slope — second-order momentum",
    "btc_momentum_confluence":        "Multi-indicator momentum agreement score",
    "btc_live_price":                 "Real-time BTC spot price — absolute level context",
    "btc_live_spot_price":            "Spot price from primary source",
    "btc_live_index_price":           "Index price — multi-exchange composite",
    "btc_live_mark_price":            "Mark price — official derivative settlement reference",
    "btc_live_funding_rate":          "Perpetual funding rate — directional positioning pressure",
    "btc_live_source_quality_score":  "Data source reliability — bad data → bad features",
    "btc_live_source_divergence_bps": "Cross-source spread — detects feed latency/errors",
    "btc_live_spot_index_basis_bps":  "Spot vs index gap — arbitrage pressure",
    "btc_live_mark_index_basis_bps":  "Mark vs index basis — liquidation risk proxy",
    "btc_live_mark_spot_basis_bps":   "Mark vs spot — leverage overhang signal",
    "btc_live_return_1m":             "1-min BTC return — ultra-short momentum",
    "btc_live_return_5m":             "5-min BTC return — short momentum",
    "btc_live_return_15m":            "15-min BTC return — medium momentum",
    "btc_live_return_1h":             "1-hour BTC return — macro momentum",
    "btc_live_volatility_proxy":      "Real-time vol estimate from tick data",
    "btc_live_confluence":            "Composite live BTC directional agreement",
    "btc_live_index_ready":           "Whether index feed is live and valid",
    "forecast_p_hit_interval":        "P(temp in resolution interval) from NWP model",
    "forecast_margin_to_lower_c":     "Gap from forecast to lower bound (°C) — safety margin",
    "forecast_margin_to_upper_c":     "Gap from forecast to upper bound (°C)",
    "forecast_uncertainty_c":         "Uncertainty band (°C) from days-out decay model",
    "forecast_drift_c":               "How much forecast changed since last update",
    "forecast_update_sequence":       "How many times forecast has updated — staleness proxy",
    "weather_fair_probability_yes":   "Model estimate of Yes-token resolution probability",
    "weather_forecast_edge":          "Model prob minus market price = edge before costs",
    "weather_forecast_margin_score":  "Normalised margin safety score",
    "weather_forecast_stability_score": "Forecast stability over recent revisions",
}


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def generate_feature_importance_report(logs_dir: Path, out_md: Path) -> None:
    imp_path = logs_dir / "feature_importance.csv"
    abl_path = logs_dir / "feature_ablation_report.csv"

    imp_df = _safe_read(imp_path)
    abl_df = _safe_read(abl_path)

    lines: list[str] = []
    lines += [
        "# Feature Importance & Economic Rationale Report\n",
        f"_Source: `{imp_path}` + `{abl_path}`_\n",
        "---\n",
    ]

    # --- Section 1: Ranked features ---
    if imp_df.empty:
        lines.append("> ⚠️  `feature_importance.csv` not found — run a training cycle first.\n")
    else:
        imp_df["importance"] = pd.to_numeric(imp_df["importance"], errors="coerce").fillna(0.0)
        imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

        nonzero = imp_df[imp_df["importance"] > 0]
        zero    = imp_df[imp_df["importance"] <= 0]

        lines.append(f"## Feature Ranking ({len(imp_df)} total | {len(nonzero)} active | {len(zero)} zero-importance)\n")
        lines += [
            "| Rank | Feature | Importance | Economic Rationale | Status |",
            "|------|---------|------------|-------------------|--------|",
        ]
        for rank, (_, row) in enumerate(imp_df.iterrows(), 1):
            fname = str(row["feature"])
            imp   = float(row["importance"])
            rationale = ECONOMIC_RATIONALE.get(fname, "")
            status = "✅" if rationale else "❌ **UNEXPLAINED**"
            rationale_short = textwrap.shorten(rationale or "No economic rationale on record", width=80)
            lines.append(f"| {rank} | `{fname}` | {imp:.5f} | {rationale_short} | {status} |")

        # Zero-importance summary
        if not zero.empty:
            lines += [
                "",
                f"### Zero-Importance Features ({len(zero)}) — Candidates for Removal",
                "",
                "These features contribute nothing to the trained model. "
                "They should be ablated unless there is a strong economic reason to keep them.",
                "",
            ]
            for _, row in zero.iterrows():
                fname = str(row["feature"])
                rationale = ECONOMIC_RATIONALE.get(fname, "_No rationale — remove._")
                lines.append(f"- `{fname}`: {rationale}")

        # Unexplained active features
        active_unexplained = nonzero[~nonzero["feature"].isin(ECONOMIC_RATIONALE)]
        if not active_unexplained.empty:
            lines += [
                "",
                f"### ❌ Active But Economically Unexplained ({len(active_unexplained)})",
                "",
                "These features have nonzero importance but lack an economic justification. "
                "**Remove them or document why they should predict outcomes.**",
                "",
            ]
            for _, row in active_unexplained.iterrows():
                lines.append(f"- `{row['feature']}` (importance={float(row['importance']):.5f})")

    # --- Section 2: Ablation summary ---
    lines += ["", "---", "## Ablation Study Results\n"]

    if abl_df.empty:
        lines.append("> ⚠️  `feature_ablation_report.csv` not found.\n")
    else:
        # Each row is one ablation scope
        baseline = abl_df[abl_df.get("scope", abl_df.columns[1] if len(abl_df.columns) > 1 else "scope") == "baseline"]
        drops = abl_df[abl_df.get("scope", abl_df.columns[1] if len(abl_df.columns) > 1 else "scope") != "baseline"]

        scope_col  = "scope"       if "scope"       in abl_df.columns else abl_df.columns[1]
        value_col  = "scope_value" if "scope_value" in abl_df.columns else abl_df.columns[2]
        acc_col    = "accuracy"    if "accuracy"    in abl_df.columns else None
        rmse_col   = "return_rmse" if "return_rmse" in abl_df.columns else None
        d_acc_col  = "delta_accuracy_vs_baseline" if "delta_accuracy_vs_baseline" in abl_df.columns else None
        d_rmse_col = "delta_rmse_vs_baseline"     if "delta_rmse_vs_baseline"     in abl_df.columns else None

        if not baseline.empty and acc_col:
            b_acc  = float(baseline.iloc[0][acc_col])
            b_rmse = float(baseline.iloc[0][rmse_col]) if rmse_col else float("nan")
            lines.append(f"**Baseline accuracy:** {b_acc:.3f} | **Baseline RMSE:** {b_rmse:.4f}\n")

        if not drops.empty:
            lines += [
                "| Feature Group Dropped | Accuracy | Δ Accuracy | RMSE | Δ RMSE | Verdict |",
                "|-----------------------|----------|------------|------|--------|---------|",
            ]
            for _, row in drops.sort_values(d_acc_col or acc_col or scope_col).iterrows():
                group   = str(row.get(value_col, "?"))
                acc     = f"{float(row[acc_col]):.3f}"  if acc_col    and pd.notna(row.get(acc_col))    else "—"
                rmse    = f"{float(row[rmse_col]):.4f}" if rmse_col   and pd.notna(row.get(rmse_col))   else "—"
                d_acc   = float(row[d_acc_col])  if d_acc_col  and pd.notna(row.get(d_acc_col))  else float("nan")
                d_rmse  = float(row[d_rmse_col]) if d_rmse_col and pd.notna(row.get(d_rmse_col)) else float("nan")
                d_acc_s  = f"{d_acc:+.3f}"  if pd.notna(d_acc)  else "—"
                d_rmse_s = f"{d_rmse:+.4f}" if pd.notna(d_rmse) else "—"
                # Verdict: positive delta_accuracy = dropping hurt accuracy → feature is USEFUL
                if pd.notna(d_acc):
                    verdict = "🔴 Keep (hurts accuracy)" if d_acc < -0.01 else ("🟢 Drop safe" if d_acc >= 0 else "🟡 Marginal")
                else:
                    verdict = "—"
                lines.append(f"| `{group}` | {acc} | {d_acc_s} | {rmse} | {d_rmse_s} | {verdict} |")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅  Wrote feature importance report → {out_md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Feature importance + economic rationale report")
    ap.add_argument("--logs", default="logs", help="Path to logs directory")
    ap.add_argument("--out",  default="logs/research/feature_importance.md")
    args = ap.parse_args()
    generate_feature_importance_report(Path(args.logs), Path(args.out))
