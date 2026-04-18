from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _format_money(value) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "nan"


def build_btc_reconciliation_audit(logs_dir: str = "logs") -> dict:
    logs_path = Path(logs_dir)
    attribution_path = logs_path / "btc_trade_attribution.csv"
    out_md = logs_path / "btc_reconciliation_audit.md"
    out_csv = logs_path / "btc_reconciliation_summary.csv"

    df = _safe_read_csv(attribution_path)
    if df.empty:
        out_md.write_text("# BTC Reconciliation Audit\n\nNo btc_trade_attribution.csv data found.\n", encoding="utf-8")
        return {"rows": 0, "report": str(out_md), "summary_csv": str(out_csv)}

    for col in [
        "realized_pnl",
        "confidence_at_entry",
        "price_return",
        "exit_fill_latency_seconds",
        "exit_partial_fill_ratio",
        "exit_realized_slippage_bps",
        "entry_price",
        "exit_price",
        "size_usdc",
        "shares",
    ]:
        if col in df.columns:
            df[col] = _to_num(df[col])

    external = df[df.get("close_reason", pd.Series("", index=df.index)).astype(str) == "external_manual_close"].copy()
    if external.empty:
        out_md.write_text("# BTC Reconciliation Audit\n\nNo external_manual_close rows found.\n", encoding="utf-8")
        return {"rows": len(df), "external_rows": 0, "report": str(out_md), "summary_csv": str(out_csv)}

    pnl = external["realized_pnl"].fillna(0.0)
    summary_rows = []

    for field in [
        "market_family",
        "signal_label",
        "active_model_kind",
        "technical_regime_bucket",
        "operational_close_flag",
        "reconciliation_close_flag",
        "actual_execution_path",
        "intended_exit_reason",
    ]:
        if field not in external.columns:
            continue
        grouped = (
            external.assign(_group=external[field].fillna("NA").astype(str))
            .groupby("_group", dropna=False)["realized_pnl"]
            .agg(["size", "sum", "mean"])
            .reset_index()
            .rename(columns={"_group": "group_value", "size": "n", "sum": "pnl_sum", "mean": "pnl_mean"})
        )
        grouped["group_field"] = field
        summary_rows.append(grouped)

    summary_df = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    if not summary_df.empty:
        summary_df = summary_df[["group_field", "group_value", "n", "pnl_sum", "pnl_mean"]]
        summary_df.to_csv(out_csv, index=False)

    rec_flag_rate = float(external.get("reconciliation_close_flag", pd.Series(False, index=external.index)).fillna(False).astype(bool).mean())
    op_flag_rate = float(external.get("operational_close_flag", pd.Series(False, index=external.index)).fillna(False).astype(bool).mean())
    live_exit_path_rate = float(
        external.get("actual_execution_path", pd.Series("", index=external.index))
        .fillna("")
        .astype(str)
        .eq("live_exit_filled")
        .mean()
    )
    missing_exec_path_rate = float(
        external.get("actual_execution_path", pd.Series("", index=external.index))
        .isna()
        .mean()
    ) if "actual_execution_path" in external.columns else 1.0

    top_loss_markets = (
        external.groupby(external.get("market", pd.Series("NA", index=external.index)).fillna("NA").astype(str))["realized_pnl"]
        .agg(["size", "sum", "mean"])
        .sort_values("sum")
        .head(15)
    )

    oldest = external.sort_values("closed_at").head(5)[["closed_at", "market", "signal_label", "realized_pnl"]]
    newest = external.sort_values("closed_at").tail(5)[["closed_at", "market", "signal_label", "realized_pnl"]]

    lines = [
        "# BTC Reconciliation Audit",
        "",
        "## Summary",
        f"- BTC attribution rows: `{len(df)}`",
        f"- `external_manual_close` rows: `{len(external)}`",
        f"- `external_manual_close` pnl_sum: `{_format_money(pnl.sum())}`",
        f"- `external_manual_close` win_rate: `{(pnl > 0).mean():.4f}`",
        f"- `reconciliation_close_flag` rate: `{rec_flag_rate:.4f}`",
        f"- `operational_close_flag` rate: `{op_flag_rate:.4f}`",
        f"- `actual_execution_path == live_exit_filled` rate: `{live_exit_path_rate:.4f}`",
        f"- missing `actual_execution_path` rate: `{missing_exec_path_rate:.4f}`",
        "",
        "## Interpretation",
        "- These rows are positions that disappeared from the reconciled open-position snapshot and were then force-closed locally.",
        "- High `reconciliation_close_flag` with missing execution-path detail points to weak state continuity, not just bad alpha.",
        "- If these are legitimate exchange-side exits, the runtime is failing to capture the corresponding close path cleanly.",
        "",
        "## Worst Markets",
        top_loss_markets.to_string(),
        "",
        "## Earliest Sample Rows",
        oldest.to_string(index=False),
        "",
        "## Latest Sample Rows",
        newest.to_string(index=False),
        "",
        "## Recommended Fixes",
        "1. Strengthen position reconciliation before invoking `_close_absent_ledger_positions()` so transient empty snapshots do not create local closures.",
        "2. Persist a concrete reconciliation reason when a trade becomes `external_manual_close`.",
        "3. Add a grace period requiring repeated missing-ledger confirmations before forced local close for BTC.",
        "4. Log the source snapshot counts and token/condition identifiers into the attribution trail at reconciliation close time.",
        "",
        "## Summary CSV",
        f"- `{out_csv}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "rows": len(df),
        "external_rows": len(external),
        "external_pnl_sum": float(pnl.sum()),
        "report": str(out_md),
        "summary_csv": str(out_csv),
    }


if __name__ == "__main__":
    result = build_btc_reconciliation_audit()
    print(json.dumps(result, indent=2))
