from __future__ import annotations

from pathlib import Path

import pandas as pd


AUDIT_PATH = Path("logs/clob_selection_audit.csv")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def build_report(audit_path: Path = AUDIT_PATH) -> str:
    if not audit_path.exists():
        return f"No audit file found at {audit_path}."

    df = pd.read_csv(audit_path, engine="python", on_bad_lines="skip")
    if df.empty:
        return f"Audit file {audit_path} is empty."

    for col in (
        "selection_utilization",
        "selection_coverage",
        "downstream_candidates_seen",
        "downstream_candidates_tradable",
        "downstream_entries_sent",
        "downstream_fills_received",
        "selected_tokens",
        "max_clob_tokens",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    linked = df[df.get("linked_cycle_id").notna()] if "linked_cycle_id" in df.columns else pd.DataFrame()
    latest = df.iloc[-1]

    recommendation = "keep_200"
    rationale = "Not enough linked runtime rows yet; keep the current cap until fresh evidence arrives."

    if not linked.empty:
        avg_selected = _safe_float(linked["selected_tokens"].mean(), 0.0)
        avg_max = _safe_float(linked["max_clob_tokens"].mean(), 0.0)
        avg_candidates = _safe_float(linked["downstream_candidates_seen"].mean(), 0.0)
        avg_tradable = _safe_float(linked["downstream_candidates_tradable"].mean(), 0.0)
        avg_entries = _safe_float(linked["downstream_entries_sent"].mean(), 0.0)
        avg_fills = _safe_float(linked["downstream_fills_received"].mean(), 0.0)
        utilization = avg_selected / avg_max if avg_max > 0 else 0.0

        if utilization >= 0.98 and avg_candidates < 40:
            recommendation = "raise_to_300"
            rationale = "The cap is fully utilized but downstream candidate throughput is still thin."
        elif avg_candidates > 150 and avg_tradable <= 1 and avg_entries == 0:
            recommendation = "lower_or_tighten_ranking"
            rationale = "The universe is producing many candidates but almost no tradable setups or entries."
        elif avg_fills > 0 or avg_entries > 0 or avg_tradable > 2:
            recommendation = "keep_200"
            rationale = "The capped universe is already generating usable downstream activity."

    lines = [
        "CLOB Selection Report",
        f"audit_rows: {len(df)}",
        f"linked_rows: {len(linked)}",
        f"latest_selected_tokens: {int(_safe_float(latest.get('selected_tokens', 0), 0))}",
        f"latest_selected_markets: {int(_safe_float(latest.get('selected_markets', 0), 0))}",
        f"latest_btc_markets: {int(_safe_float(latest.get('btc_markets', 0), 0))}",
        f"latest_weather_markets: {int(_safe_float(latest.get('weather_markets', 0), 0))}",
        f"latest_median_liquidity: {_safe_float(latest.get('median_liquidity', 0), 0):.2f}",
        f"latest_median_volume: {_safe_float(latest.get('median_volume', 0), 0):.2f}",
        f"latest_median_spread: {_safe_float(latest.get('median_spread', 0), 0):.4f}",
        f"latest_downstream_candidates_seen: {int(_safe_float(latest.get('downstream_candidates_seen', 0), 0))}",
        f"latest_downstream_candidates_tradable: {int(_safe_float(latest.get('downstream_candidates_tradable', 0), 0))}",
        f"latest_downstream_entries_sent: {int(_safe_float(latest.get('downstream_entries_sent', 0), 0))}",
        f"latest_downstream_fills_received: {int(_safe_float(latest.get('downstream_fills_received', 0), 0))}",
        f"recommendation: {recommendation}",
        f"rationale: {rationale}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(build_report())
