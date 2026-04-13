"""
research/ablation_study.py

Reads logs/feature_ablation_report.csv (produced by feature_ablation.py)
and generates a ranked impact table showing which feature groups matter most.

Run:
    python research/ablation_study.py
    python research/ablation_study.py --logs logs --out logs/research/ablation_study.md
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


# Map scope_value (group name) to economic category
_GROUP_CATEGORY: dict[str, str] = {
    "onchain_network":      "BTC on-chain",
    "btc_live":             "BTC live feed",
    "btc_regime":           "BTC market regime",
    "wallet":               "Wallet / copy-trade",
    "forecast":             "Weather forecast",
    "weather_market":       "Weather market structure",
    "portfolio":            "Portfolio state",
    "technical":            "Technical indicators",
    "sentiment":            "Sentiment (FGI / Trends)",
    "execution":            "Execution quality",
}


def generate_ablation_report(logs_dir: Path, out_md: Path) -> None:
    abl_path = logs_dir / "feature_ablation_report.csv"
    df = _safe_read(abl_path)

    lines: list[str] = [
        "# Feature Ablation Study\n",
        f"_Source: `{abl_path}`_\n",
        "Each row shows what happens when an entire feature group is removed from training.\n",
        "A negative Δ accuracy means dropping that group HURTS the model — those features are useful.\n",
        "---\n",
    ]

    if df.empty:
        lines.append("> ⚠️  `feature_ablation_report.csv` not found — run `feature_ablation.py` first.\n")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines), encoding="utf-8")
        print(f"✅  Wrote (empty) ablation report → {out_md}")
        return

    # Identify columns
    scope_col   = next((c for c in ["scope", df.columns[1]]     if c in df.columns), df.columns[1])
    value_col   = next((c for c in ["scope_value", df.columns[2]] if c in df.columns), df.columns[2])
    acc_col     = "accuracy"    if "accuracy"    in df.columns else None
    rmse_col    = "return_rmse" if "return_rmse" in df.columns else None
    d_acc_col   = "delta_accuracy_vs_baseline" if "delta_accuracy_vs_baseline" in df.columns else None
    d_rmse_col  = "delta_rmse_vs_baseline"     if "delta_rmse_vs_baseline"     in df.columns else None
    feat_kept   = "usable_feature_count" if "usable_feature_count" in df.columns else None
    fam_feats   = "family_features_used"  if "family_features_used"  in df.columns else None

    baseline = df[df[scope_col].astype(str) == "baseline"]
    drops    = df[df[scope_col].astype(str) != "baseline"].copy()

    # Baseline summary
    if not baseline.empty and acc_col:
        b_row  = baseline.iloc[0]
        b_acc  = float(b_row[acc_col])
        b_rmse = float(b_row[rmse_col]) if rmse_col else float("nan")
        b_rows = int(b_row.get("train_rows", 0)) + int(b_row.get("test_rows", 0))
        b_feat = int(b_row[feat_kept]) if feat_kept else "?"
        b_gen  = str(b_row.get("generated_at", ""))[:10]
        lines += [
            "## Baseline Model\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Accuracy | {b_acc:.3f} |",
            f"| Return RMSE | {b_rmse:.4f} |",
            f"| Total samples | {b_rows} |",
            f"| Features used | {b_feat} |",
            f"| Report date | {b_gen} |",
            "",
        ]

    # Ranked ablation table
    if not drops.empty and d_acc_col:
        drops[d_acc_col]  = pd.to_numeric(drops[d_acc_col],  errors="coerce")
        drops[d_rmse_col] = pd.to_numeric(drops[d_rmse_col], errors="coerce") if d_rmse_col else float("nan")
        drops = drops.sort_values(d_acc_col)  # most harmful drops first

        lines += [
            "## Ablation Results (ranked by accuracy impact)\n",
            "_Negative Δ accuracy = removing this group hurts the model → features are useful._\n",
            "| Category | Group | Δ Accuracy | Δ RMSE | Features Kept | Verdict |",
            "|----------|-------|------------|--------|---------------|---------|",
        ]
        for _, row in drops.iterrows():
            group    = str(row.get(value_col, "?"))
            category = _GROUP_CATEGORY.get(group, group)
            d_acc    = float(row[d_acc_col])  if pd.notna(row.get(d_acc_col))  else float("nan")
            d_rmse   = float(row[d_rmse_col]) if d_rmse_col and pd.notna(row.get(d_rmse_col)) else float("nan")
            kept     = int(row[feat_kept])    if feat_kept  and pd.notna(row.get(feat_kept))  else "?"
            d_acc_s  = f"{d_acc:+.3f}"  if pd.notna(d_acc)  else "—"
            d_rmse_s = f"{d_rmse:+.4f}" if pd.notna(d_rmse) else "—"

            if pd.notna(d_acc):
                if d_acc < -0.02:
                    verdict = "🔴 Keep — significant loss without it"
                elif d_acc < 0:
                    verdict = "🟡 Keep — marginal contribution"
                elif d_acc < 0.01:
                    verdict = "🟢 Safe to drop — no measurable impact"
                else:
                    verdict = "🟢 Drop — model improves without it"
            else:
                verdict = "—"

            lines.append(f"| {category} | `{group}` | {d_acc_s} | {d_rmse_s} | {kept} | {verdict} |")

        # Summary counts
        harmful = drops[drops[d_acc_col] < -0.01]
        safe    = drops[drops[d_acc_col] >= 0]
        lines += [
            "",
            f"**{len(harmful)} group(s) significantly hurt accuracy when removed** (keep these).",
            f"**{len(safe)} group(s) have no or negative contribution** (safe to drop).",
        ]

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅  Wrote ablation study → {out_md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Feature ablation study report")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out",  default="logs/research/ablation_study.md")
    args = ap.parse_args()
    generate_ablation_report(Path(args.logs), Path(args.out))
