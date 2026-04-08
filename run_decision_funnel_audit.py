from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


LOGS_DIR = Path("logs")
CANDIDATE_DECISIONS_CSV = LOGS_DIR / "candidate_decisions.csv"
CANDIDATE_CYCLE_STATS_CSV = LOGS_DIR / "candidate_cycle_stats.csv"
DEFAULT_BASELINE_CYCLE_ID = "20260408T065557.059836Z"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _load_details(series: pd.Series) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in series.fillna(""):
        payload = {}
        if isinstance(raw, str) and raw.strip():
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"_raw_details_json": raw}
        out.append(payload)
    return out


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _latest_cycle_id_from_stats(stats_df: pd.DataFrame) -> str | None:
    if stats_df.empty or "cycle_id" not in stats_df.columns:
        return None
    rows = stats_df.copy()
    if "timestamp" in rows.columns:
        rows["timestamp"] = pd.to_datetime(rows["timestamp"], errors="coerce", utc=True)
        rows = rows.sort_values(["timestamp", "cycle_id"], na_position="last")
    return str(rows.iloc[-1]["cycle_id"])


def _summarize_cycle(decisions_df: pd.DataFrame, stats_df: pd.DataFrame, cycle_id: str) -> dict[str, Any]:
    cycle_decisions = decisions_df[decisions_df.get("cycle_id", pd.Series(dtype=str)).astype(str) == str(cycle_id)].copy()
    cycle_stats = stats_df[stats_df.get("cycle_id", pd.Series(dtype=str)).astype(str) == str(cycle_id)].copy()
    details = _load_details(cycle_decisions.get("details_json", pd.Series(dtype=str)))
    cycle_decisions["details"] = details

    reject_counts = Counter(
        str(value).strip()
        for value in cycle_decisions.get("reject_reason", pd.Series(dtype=str)).fillna("")
        if str(value).strip()
    )
    gate_counts = Counter(
        str(value).strip()
        for value in cycle_decisions.get("gate", pd.Series(dtype=str)).fillna("")
        if str(value).strip()
    )
    final_decision_counts = Counter(
        str(value).strip()
        for value in cycle_decisions.get("final_decision", pd.Series(dtype=str)).fillna("")
        if str(value).strip()
    )

    rule_rows = cycle_decisions[
        cycle_decisions.get("reject_reason", pd.Series(dtype=str)).fillna("").astype(str) == "rule_veto"
    ].copy()
    rule_score_fail = 0
    rule_spread_fail = 0
    rule_liquidity_fail = 0
    for _, row in rule_rows.iterrows():
        detail = row.get("details") or {}
        score = _safe_float(detail.get("rule_score"))
        score_threshold = _safe_float(detail.get("rule_score_threshold"))
        spread = _safe_float(detail.get("rule_spread"))
        spread_threshold = _safe_float(detail.get("rule_spread_threshold"))
        liq = _safe_float(detail.get("rule_liquidity_value"))
        liq_threshold = _safe_float(detail.get("rule_liquidity_threshold"))
        if score is not None and score_threshold is not None and score < score_threshold:
            rule_score_fail += 1
        if spread is not None and spread_threshold is not None and spread > spread_threshold:
            rule_spread_fail += 1
        if liq is not None and liq_threshold is not None and liq < liq_threshold:
            rule_liquidity_fail += 1

    wallet_gate_rows = cycle_decisions[
        cycle_decisions.get("reject_reason", pd.Series(dtype=str)).fillna("").astype(str) == "wallet_state_gate_failed"
    ].copy()
    wallet_reason_counts = Counter()
    for _, row in wallet_gate_rows.iterrows():
        detail = row.get("details") or {}
        raw_reason = str(detail.get("wallet_state_gate_reason") or "").strip()
        if raw_reason:
            for part in raw_reason.split("|"):
                token = part.strip()
                if token:
                    wallet_reason_counts[token] += 1
            continue
        if detail.get("wallet_watchlist_approved") is False:
            wallet_reason_counts["wallet_not_approved"] += 1
        if detail.get("wallet_fresh") is False:
            wallet_reason_counts["wallet_state_stale"] += 1
        if detail.get("wallet_conflict_with_stronger") is True:
            wallet_reason_counts["conflict_with_stronger_wallet"] += 1
        wallet_quality_score = _safe_float(detail.get("wallet_quality_score"))
        if wallet_quality_score is not None and wallet_quality_score < 0.55:
            wallet_reason_counts["wallet_quality_below_0_55"] += 1

    governor_level = None
    governor_reason = ""
    if not cycle_decisions.empty:
        for detail in cycle_decisions["details"]:
            if isinstance(detail, dict) and detail:
                if governor_level is None and detail.get("performance_governor_level") is not None:
                    governor_level = detail.get("performance_governor_level")
                if not governor_reason and detail.get("performance_governor_reason"):
                    governor_reason = str(detail.get("performance_governor_reason"))
                if governor_level is not None and governor_reason:
                    break

    cycle_stats_row = cycle_stats.iloc[-1].to_dict() if not cycle_stats.empty else {}
    report = {
        "cycle_id": cycle_id,
        "timestamp": str(cycle_stats_row.get("timestamp") or ""),
        "candidates_seen": int(cycle_stats_row.get("candidates_seen") or len(cycle_decisions)),
        "candidates_tradable": int(cycle_stats_row.get("candidates_tradable") or 0),
        "candidates_rejected": int(cycle_stats_row.get("candidates_rejected") or 0),
        "entries_sent": int(cycle_stats_row.get("entries_sent") or 0),
        "fills_received": int(cycle_stats_row.get("fills_received") or 0),
        "reject_counts": dict(sorted(reject_counts.items())),
        "gate_counts": dict(sorted(gate_counts.items())),
        "final_decision_counts": dict(sorted(final_decision_counts.items())),
        "rule_veto_breakdown": {
            "total": int(len(rule_rows)),
            "score_fail": int(rule_score_fail),
            "spread_fail": int(rule_spread_fail),
            "liquidity_fail": int(rule_liquidity_fail),
        },
        "wallet_gate_breakdown": dict(sorted(wallet_reason_counts.items())),
        "performance_governor_level": governor_level,
        "performance_governor_reason": governor_reason,
    }
    return report


def _diff_counts(before: dict[str, int], after: dict[str, int]) -> list[dict[str, Any]]:
    keys = sorted(set(before) | set(after))
    return [
        {
            "key": key,
            "before": int(before.get(key, 0)),
            "after": int(after.get(key, 0)),
            "delta": int(after.get(key, 0)) - int(before.get(key, 0)),
        }
        for key in keys
    ]


def _write_markdown_report(
    report_path: Path,
    *,
    baseline_report: dict[str, Any],
    fresh_report: dict[str, Any],
    fresh_generated: bool,
) -> None:
    reject_diff = _diff_counts(
        baseline_report.get("reject_counts", {}),
        fresh_report.get("reject_counts", {}),
    )
    gate_diff = _diff_counts(
        baseline_report.get("gate_counts", {}),
        fresh_report.get("gate_counts", {}),
    )
    final_diff = _diff_counts(
        baseline_report.get("final_decision_counts", {}),
        fresh_report.get("final_decision_counts", {}),
    )
    lines = [
        "# Decision Funnel Audit Report",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Fresh cycle generated during this run: {'yes' if fresh_generated else 'no'}",
        f"- Baseline cycle: `{baseline_report.get('cycle_id', '')}`",
        f"- Fresh cycle: `{fresh_report.get('cycle_id', '')}`",
        "",
        "## Baseline",
        "",
        f"- Candidates seen: {baseline_report.get('candidates_seen', 0)}",
        f"- Candidates tradable: {baseline_report.get('candidates_tradable', 0)}",
        f"- Entries sent: {baseline_report.get('entries_sent', 0)}",
        f"- Fills received: {baseline_report.get('fills_received', 0)}",
        "",
        "Reject counts:",
    ]
    lines.extend(
        [f"- `{item}`: {count}" for item, count in baseline_report.get("reject_counts", {}).items()]
        or ["- none"]
    )
    lines.extend(
        [
            "",
            "## Fresh",
            "",
            f"- Candidates seen: {fresh_report.get('candidates_seen', 0)}",
            f"- Candidates tradable: {fresh_report.get('candidates_tradable', 0)}",
            f"- Entries sent: {fresh_report.get('entries_sent', 0)}",
            f"- Fills received: {fresh_report.get('fills_received', 0)}",
            "",
            "Reject counts:",
        ]
    )
    lines.extend(
        [f"- `{item}`: {count}" for item, count in fresh_report.get("reject_counts", {}).items()]
        or ["- none"]
    )
    lines.extend(
        [
            "",
            "## Delta Reject Counts",
            "",
        ]
    )
    lines.extend(
        [f"- `{row['key']}`: {row['before']} -> {row['after']} ({row['delta']:+d})" for row in reject_diff]
        or ["- none"]
    )
    lines.extend(["", "## Delta Gates", ""])
    lines.extend(
        [f"- `{row['key']}`: {row['before']} -> {row['after']} ({row['delta']:+d})" for row in gate_diff]
        or ["- none"]
    )
    lines.extend(["", "## Delta Final Decisions", ""])
    lines.extend(
        [f"- `{row['key']}`: {row['before']} -> {row['after']} ({row['delta']:+d})" for row in final_diff]
        or ["- none"]
    )
    lines.extend(
        [
            "",
            "## Rule Veto Breakdown",
            "",
            f"- Baseline: {json.dumps(baseline_report.get('rule_veto_breakdown', {}), sort_keys=True)}",
            f"- Fresh: {json.dumps(fresh_report.get('rule_veto_breakdown', {}), sort_keys=True)}",
            "",
            "## Wallet Gate Breakdown",
            "",
            f"- Baseline: {json.dumps(baseline_report.get('wallet_gate_breakdown', {}), sort_keys=True)}",
            f"- Fresh: {json.dumps(fresh_report.get('wallet_gate_breakdown', {}), sort_keys=True)}",
            "",
            "## Notes",
            "",
            "- The fresh cycle was run with order submission monkeypatched to no-op so no live orders could be placed.",
            "- Any execution-stage rejects in the fresh cycle mean the candidate reached the submission boundary and would have attempted a live order without the audit safety patch.",
            "- Market conditions can differ between the baseline and the fresh cycle, so the comparison shows funnel-shape changes, not a strict replay.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _run_one_cycle_safely() -> str:
    import supervisor

    class _StopAfterCycle(KeyboardInterrupt):
        pass

    original_append_csv_record = supervisor.append_csv_record
    original_submit_entry = supervisor.OrderManager.submit_entry
    original_submit_market_entry = supervisor.OrderManager.submit_market_entry
    original_wait_for_fill = supervisor.OrderManager.wait_for_fill
    original_cancel_stale_order = supervisor.OrderManager.cancel_stale_order

    def _audit_submit(self, *args, **kwargs):
        return {"reason": "audit_no_submit"}, {"reason": "audit_no_submit"}

    def _audit_wait(self, order_id, timeout_seconds=20, poll_seconds=2):
        return {"filled": False, "response": {"order_id": order_id, "reason": "audit_no_submit"}}

    def _audit_cancel(self, order_id):
        return {"order_id": order_id, "status": "cancelled", "reason": "audit_no_submit"}

    def _append_and_stop(path, record):
        result = original_append_csv_record(path, record)
        if os.path.normcase(os.path.normpath(str(path))) == os.path.normcase(os.path.normpath(supervisor.CANDIDATE_CYCLE_STATS_FILE)):
            raise _StopAfterCycle("decision audit finished after one cycle")
        return result

    supervisor.OrderManager.submit_entry = _audit_submit
    supervisor.OrderManager.submit_market_entry = _audit_submit
    supervisor.OrderManager.wait_for_fill = _audit_wait
    supervisor.OrderManager.cancel_stale_order = _audit_cancel
    supervisor.append_csv_record = _append_and_stop

    os.environ.setdefault("ENABLE_LIVE_RETRAIN", "false")

    try:
        supervisor.main_loop()
    finally:
        supervisor.append_csv_record = original_append_csv_record
        supervisor.OrderManager.submit_entry = original_submit_entry
        supervisor.OrderManager.submit_market_entry = original_submit_market_entry
        supervisor.OrderManager.wait_for_fill = original_wait_for_fill
        supervisor.OrderManager.cancel_stale_order = original_cancel_stale_order

    stats_df = _read_csv(CANDIDATE_CYCLE_STATS_CSV)
    cycle_id = _latest_cycle_id_from_stats(stats_df)
    if not cycle_id:
        raise RuntimeError("Unable to resolve fresh cycle id after audit run.")
    return cycle_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a safe one-cycle decision-funnel audit and compare blocker distributions.")
    parser.add_argument(
        "--baseline-cycle-id",
        default=DEFAULT_BASELINE_CYCLE_ID,
        help="Baseline cycle_id to compare against.",
    )
    parser.add_argument(
        "--report-path",
        default=str(LOGS_DIR / "decision_funnel_audit_report.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--json-path",
        default=str(LOGS_DIR / "decision_funnel_audit_report.json"),
        help="JSON report output path.",
    )
    parser.add_argument(
        "--skip-fresh-run",
        action="store_true",
        help="Only summarize existing cycles without generating a new safe audit cycle.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    decisions_df = _read_csv(CANDIDATE_DECISIONS_CSV)
    stats_df = _read_csv(CANDIDATE_CYCLE_STATS_CSV)

    baseline_cycle_id = str(args.baseline_cycle_id).strip()
    if not baseline_cycle_id:
        baseline_cycle_id = _latest_cycle_id_from_stats(stats_df) or DEFAULT_BASELINE_CYCLE_ID

    fresh_generated = False
    fresh_cycle_id = _latest_cycle_id_from_stats(stats_df)
    if not args.skip_fresh_run:
        fresh_cycle_id = _run_one_cycle_safely()
        fresh_generated = True
        decisions_df = _read_csv(CANDIDATE_DECISIONS_CSV)
        stats_df = _read_csv(CANDIDATE_CYCLE_STATS_CSV)

    if not baseline_cycle_id:
        raise RuntimeError("No baseline cycle id available.")
    if not fresh_cycle_id:
        raise RuntimeError("No fresh cycle id available.")

    baseline_report = _summarize_cycle(decisions_df, stats_df, baseline_cycle_id)
    fresh_report = _summarize_cycle(decisions_df, stats_df, fresh_cycle_id)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_cycle_id": baseline_cycle_id,
        "fresh_cycle_id": fresh_cycle_id,
        "fresh_cycle_generated": fresh_generated,
        "baseline": baseline_report,
        "fresh": fresh_report,
        "reject_diff": _diff_counts(baseline_report.get("reject_counts", {}), fresh_report.get("reject_counts", {})),
        "gate_diff": _diff_counts(baseline_report.get("gate_counts", {}), fresh_report.get("gate_counts", {})),
        "final_decision_diff": _diff_counts(
            baseline_report.get("final_decision_counts", {}),
            fresh_report.get("final_decision_counts", {}),
        ),
    }

    report_path = Path(args.report_path)
    json_path = Path(args.json_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown_report(
        report_path,
        baseline_report=baseline_report,
        fresh_report=fresh_report,
        fresh_generated=fresh_generated,
    )
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
