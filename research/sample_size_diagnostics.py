"""
research/sample_size_diagnostics.py

Scans current datasets and reports:
  - Trade sample count by market family and date range
  - Shadow sample counts vs the 50-sample promotion threshold
  - Dataset vintage coverage (PiT snapshots)
  - Weather forecast archive stats (PiT traceability)
  - Feature imputation rates from shadow_results.csv

Run:
    python research/sample_size_diagnostics.py
    python research/sample_size_diagnostics.py --logs logs --out logs/research/sample_size.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

_REQUIRED_SHADOW = 50


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _date_range(df: pd.DataFrame, col: str) -> tuple[str, str]:
    if col not in df.columns:
        return "—", "—"
    ts = pd.to_datetime(df[col], errors="coerce", utc=True)
    if ts.notna().any():
        return str(ts.min().date()), str(ts.max().date())
    return "—", "—"


def _pct(n, total) -> str:
    return f"{n/total*100:.1f}%" if total else "—"


def generate_report(logs_dir: Path, out_md: Path) -> None:
    hist_path    = logs_dir / "historical_dataset.csv"
    btc_path     = logs_dir / "btc_price_dataset.csv"
    shadow_path  = logs_dir / "shadow_results.csv"
    registry_path = logs_dir / "model_registry_comparison.csv"
    archive_path = logs_dir / "weather_forecast_archive.csv"
    vintage_dir  = logs_dir / "vintages"
    closed_path  = logs_dir / "closed_positions.csv"

    lines: list[str] = [
        "# Sample Size & Data Quality Diagnostics\n",
        f"_Run against: `{logs_dir.resolve()}`_\n",
        "---\n",
    ]

    # -----------------------------------------------------------------------
    # 1. Historical dataset (Polymarket)
    # -----------------------------------------------------------------------
    lines.append("## 1. Historical Dataset (Polymarket signals)\n")
    hist_df = _safe_read(hist_path)
    if hist_df.empty:
        lines.append(f"> ⚠️  `{hist_path.name}` not found — run `HistoricalDatasetBuilder.write()` first.\n")
    else:
        d_min, d_max = _date_range(hist_df, "timestamp")
        lines += [
            f"**Total rows:** {len(hist_df)} | **Date range:** {d_min} → {d_max}\n",
            "| Family | Rows | Date Min | Date Max |",
            "|--------|------|----------|----------|",
        ]
        if "market_family" in hist_df.columns:
            for fam, grp in hist_df.groupby("market_family", dropna=False):
                d0, d1 = _date_range(grp, "timestamp")
                lines.append(f"| `{fam}` | {len(grp)} | {d0} | {d1} |")
        else:
            lines.append(f"| (all) | {len(hist_df)} | {d_min} | {d_max} |")
        lines.append("")

    # -----------------------------------------------------------------------
    # 2. BTC price dataset
    # -----------------------------------------------------------------------
    lines.append("## 2. BTC Price Dataset (candles)\n")
    # Large file — don't read entirely; just check existence and size
    btc_size_mb = btc_path.stat().st_size / 1e6 if btc_path.exists() else 0
    if btc_size_mb > 0:
        lines.append(f"**File size:** {btc_size_mb:.0f} MB — file exists. "
                     f"Sample the first row for schema verification.\n")
        try:
            btc_head = pd.read_csv(btc_path, nrows=3, engine="python")
            d_min, d_max = _date_range(btc_head, "timestamp")
            lines.append(f"**Columns:** {len(btc_head.columns)} | First rows timestamped from {d_min}\n")
        except Exception as e:
            lines.append(f"_Could not read head: {e}_\n")
    else:
        lines.append(f"> ⚠️  `{btc_path.name}` not found.\n")

    # -----------------------------------------------------------------------
    # 3. Closed positions summary
    # -----------------------------------------------------------------------
    lines.append("## 3. Closed Positions Sample Coverage\n")
    closed_df = _safe_read(closed_path)
    if closed_df.empty:
        lines.append(f"> ⚠️  `{closed_path.name}` not found.\n")
    else:
        closed_only = closed_df[closed_df.get("status", pd.Series()).astype(str).str.upper() == "CLOSED"] if "status" in closed_df.columns else closed_df
        d_min, d_max = _date_range(closed_only, "opened_at" if "opened_at" in closed_only.columns else "timestamp")
        lines += [
            f"**Total closed positions:** {len(closed_only)} | **Date range:** {d_min} → {d_max}\n",
            "| Family | Count | Date Min | Date Max |",
            "|--------|-------|----------|----------|",
        ]
        if "market_family" in closed_only.columns:
            for fam, grp in closed_only.groupby("market_family", dropna=False):
                c0, c1 = _date_range(grp, "opened_at" if "opened_at" in grp.columns else "timestamp")
                lines.append(f"| `{fam}` | {len(grp)} | {c0} | {c1} |")
        lines.append("")

    # -----------------------------------------------------------------------
    # 4. Shadow deployment results
    # -----------------------------------------------------------------------
    lines.append("## 4. Shadow Deployment Coverage\n")
    shadow_df = _safe_read(shadow_path)
    if shadow_df.empty:
        lines.append(f"> ⚠️  `{shadow_path.name}` not found.\n")
    else:
        d_min, d_max = _date_range(shadow_df, "timestamp")
        total = len(shadow_df)
        outcomes = shadow_df["outcome"].value_counts().to_dict() if "outcome" in shadow_df.columns else {}
        imp_count = shadow_df["imputed_feature_count"].fillna(0).astype(int).sum() if "imputed_feature_count" in shadow_df.columns else 0
        imp_rows = (shadow_df["feature_status"].astype(str) == "imputed").sum() if "feature_status" in shadow_df.columns else 0

        lines += [
            f"**Total shadow observations:** {total} | **Date range:** {d_min} → {d_max}\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Observations | {total} |",
            f"| Meets 50-sample threshold | {'✅ Yes' if total >= _REQUIRED_SHADOW else f'❌ No ({total}/{_REQUIRED_SHADOW})'} |",
            f"| Rows with imputed features | {imp_rows} ({_pct(imp_rows, total)}) |",
            f"| Total imputed feature slots | {imp_count} |",
        ]
        for outcome, cnt in outcomes.items():
            lines.append(f"| Outcome `{outcome}` | {cnt} ({_pct(cnt, total)}) |")
        lines.append("")

        # Imputation breakdown
        if "imputed_features" in shadow_df.columns and imp_rows > 0:
            from collections import Counter
            all_imp: list[str] = []
            for val in shadow_df["imputed_features"].dropna():
                all_imp.extend([f.strip() for f in str(val).split("|") if f.strip()])
            counts = Counter(all_imp).most_common(15)
            lines += [
                "### Top Imputed Features (data quality flag)\n",
                "These features are missing from live signals and being filled with priors. "
                "High imputation rates indicate a data pipeline gap.\n",
                "| Feature | Imputed Count |",
                "|---------|---------------|",
            ]
            for feat, cnt in counts:
                lines.append(f"| `{feat}` | {cnt} |")
            lines.append("")

    # -----------------------------------------------------------------------
    # 5. Model registry promotion status
    # -----------------------------------------------------------------------
    lines.append("## 5. Model Registry — Shadow Promotion Gate\n")
    lines.append(f"**Required shadow samples before promotion:** {_REQUIRED_SHADOW}\n")
    registry_df = _safe_read(registry_path)
    if registry_df.empty:
        lines.append(f"_`{registry_path.name}` not found._\n")
    else:
        rows_out: list[dict] = []
        for _, row in registry_df.iterrows():
            try:
                shadow_rpt = json.loads(str(row.get("shadow_report") or "{}"))
                shadow_n   = int(shadow_rpt.get("samples", 0))
            except Exception:
                shadow_n = 0
            rows_out.append({
                "model_group":  row.get("model_group", "?"),
                "family":       row.get("market_family", "?"),
                "status":       row.get("status", "?"),
                "shadow_n":     shadow_n,
                "eligible":     shadow_n >= _REQUIRED_SHADOW,
                "candidate_id": str(row.get("candidate_id", row.get("model_version", "?")))[:20],
            })
        lines += [
            "| Candidate | Family | Status | Shadow Samples | Eligible |",
            "|-----------|--------|--------|----------------|----------|",
        ]
        for r in rows_out:
            elig = "✅" if r["eligible"] else f"❌ ({r['shadow_n']}/{_REQUIRED_SHADOW})"
            lines.append(f"| `{r['candidate_id']}` | `{r['family']}` | {r['status']} "
                         f"| {r['shadow_n']} | {elig} |")
        lines.append("")

    # -----------------------------------------------------------------------
    # 6. Dataset vintages (PiT)
    # -----------------------------------------------------------------------
    lines.append("## 6. Dataset Vintages (Point-in-Time Snapshots)\n")
    if vintage_dir.exists():
        vintages = sorted(vintage_dir.glob("historical_dataset_*.csv"))
        lines.append(f"**{len(vintages)} snapshots** stored in `logs/vintages/`\n")
        if vintages:
            lines += [
                "| Vintage | Size (KB) |",
                "|---------|-----------|",
            ]
            for v in vintages[-15:]:
                lines.append(f"| `{v.name}` | {v.stat().st_size // 1024} |")
    else:
        lines.append("_No vintages directory found. Will be created on next `HistoricalDatasetBuilder.write()` call._\n")
    lines.append("")

    # -----------------------------------------------------------------------
    # 7. Weather forecast archive (PiT traceability)
    # -----------------------------------------------------------------------
    lines.append("## 7. Weather Forecast Archive (PiT Traceability)\n")
    arch_df = _safe_read(archive_path)
    if arch_df.empty:
        lines.append(f"_`{archive_path.name}` not found. Will grow as WeatherForecastService fetches live data._\n")
    else:
        d_min, d_max = _date_range(arch_df, "forecast_issue_time")
        seq_max = pd.to_numeric(arch_df.get("forecast_update_sequence"), errors="coerce").max() if "forecast_update_sequence" in arch_df.columns else 0
        unique_locations = arch_df["cache_key"].nunique() if "cache_key" in arch_df.columns else "?"
        lines += [
            f"**Archive rows:** {len(arch_df)} | **Date range:** {d_min} → {d_max}\n",
            f"**Unique (location, date) keys:** {unique_locations}\n",
            f"**Max update sequence:** {seq_max}\n",
        ]

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅  Wrote sample size diagnostics → {out_md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sample size and data quality diagnostics")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out",  default="logs/research/sample_size.md")
    args = ap.parse_args()
    generate_report(Path(args.logs), Path(args.out))
