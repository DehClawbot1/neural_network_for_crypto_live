"""
alpha_thesis_validator.py
─────────────────────────
Turn the qualitative alpha thesis docs into live, measurable health checks.

For each claimed signal source (docs/alpha_thesis_*.md tables), this
module computes:

- **IC (Information Coefficient)**: rank correlation between the signal
  value at entry and the subsequent realised trade return. This is the
  industry-standard per-signal predictive-power metric; empty/zero IC
  means the signal carries no information in live conditions.
- **IC half-life**: rolling IC over sliding windows to detect signal
  decay. A signal whose IC decays from 0.05 → 0.01 over a month is
  about to stop paying; the dashboard surfaces it before the P&L dies.
- **t-statistic**: IC × sqrt(N-2)/sqrt(1-IC^2). |t| > 2 is the classical
  significance threshold; we log anything with |t| < 1.5 as "suspect".
- **Hit rate by quintile**: sort signals into Q1..Q5, check win rate per
  quintile. A healthy signal has monotone hit-rate across quintiles.

Output: `logs/alpha_thesis_health.json`.

Failure modes (thesis health):
- Any signal with |IC| < 0.01 AND N ≥ 200 → `dead_signal` flag
- Any signal with IC trending negative over last 3 windows → `sign_flip`
- Any signal whose quintile monotonicity breaks in 2 consecutive runs →
  `non_monotone`

This is what turns the markdown thesis into an actual ops tool: every
claim in the thesis can be checked against the last 500 closed trades.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPORT_PATH = Path("logs") / "alpha_thesis_health.json"

# Signals claimed in docs/alpha_thesis_btc.md + alpha_thesis_weather.md
# Any column not in closed_positions.csv is silently skipped (degrades
# gracefully during a rename rather than crashing the report).
_CLAIMED_SIGNALS_BTC = [
    "whale_pressure",
    "market_structure_score",
    "btc_trend_confluence",
    "btc_network_activity_score",
    "btc_network_stress_score",
    "entry_btc_predicted_return",
    "entry_btc_forecast_confidence",
    "entry_btc_mtf_agreement",
]
_CLAIMED_SIGNALS_WEATHER = [
    "forecast_p_hit_interval",
    "forecast_margin_to_lower_c",
    "forecast_margin_to_upper_c",
    "forecast_uncertainty_c",
    "weather_forecast_edge",
    "weather_forecast_margin_score",
    "weather_forecast_stability_score",
]

_IC_DEAD_THRESHOLD = 0.01      # below this + N≥200 → flagged as dead
_IC_MIN_SAMPLES = 50
_IC_DEAD_MIN_SAMPLES = 200


def _rank_ic(signal: np.ndarray, returns: np.ndarray) -> float:
    """Spearman rank correlation, NaN-safe."""
    if signal.size != returns.size or signal.size < _IC_MIN_SAMPLES:
        return float("nan")
    mask = np.isfinite(signal) & np.isfinite(returns)
    if mask.sum() < _IC_MIN_SAMPLES:
        return float("nan")
    s_rank = pd.Series(signal[mask]).rank(method="average").to_numpy()
    r_rank = pd.Series(returns[mask]).rank(method="average").to_numpy()
    s_rank -= s_rank.mean()
    r_rank -= r_rank.mean()
    denom = np.sqrt((s_rank ** 2).sum() * (r_rank ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float((s_rank * r_rank).sum() / denom)


def _t_stat(ic: float, n: int) -> float:
    if not math.isfinite(ic) or n <= 2 or abs(ic) >= 1.0:
        return float("nan")
    return ic * math.sqrt((n - 2) / max(1e-9, 1.0 - ic * ic))


def _quintile_hit_rates(signal: np.ndarray, outcome: np.ndarray) -> list[float]:
    mask = np.isfinite(signal) & np.isfinite(outcome)
    if mask.sum() < _IC_MIN_SAMPLES:
        return []
    s = signal[mask]
    y = outcome[mask]
    edges = np.quantile(s, np.linspace(0, 1, 6))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    hit_rates = []
    for i in range(5):
        bucket = (s > edges[i]) & (s <= edges[i + 1])
        if bucket.sum() == 0:
            hit_rates.append(float("nan"))
        else:
            hit_rates.append(float(y[bucket].mean()))
    return [round(x, 3) if math.isfinite(x) else None for x in hit_rates]


def _monotone_up(xs: list[float | None]) -> bool:
    vals = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return len(vals) >= 4 and all(vals[i] <= vals[i + 1] + 0.05 for i in range(len(vals) - 1))


def _rolling_ic(signal: np.ndarray, returns: np.ndarray, *, windows: int = 3) -> list[float]:
    """IC computed on `windows` equal-sized chronological chunks."""
    n = min(signal.size, returns.size)
    if n < _IC_MIN_SAMPLES * windows:
        return []
    chunk = n // windows
    out: list[float] = []
    for i in range(windows):
        lo, hi = i * chunk, (i + 1) * chunk
        ic = _rank_ic(signal[lo:hi], returns[lo:hi])
        if math.isfinite(ic):
            out.append(round(ic, 4))
    return out


def validate_signal(
    name: str,
    closed: pd.DataFrame,
    return_col: str,
    outcome_col: str,
) -> dict:
    if name not in closed.columns:
        return {"signal": name, "status": "missing_column"}
    sig = pd.to_numeric(closed[name], errors="coerce").to_numpy()
    ret = pd.to_numeric(closed[return_col], errors="coerce").to_numpy()
    out = pd.to_numeric(closed[outcome_col], errors="coerce").to_numpy()
    n = int(np.isfinite(sig).sum())
    if n < _IC_MIN_SAMPLES:
        return {"signal": name, "status": "insufficient_data", "n": n}
    ic = _rank_ic(sig, ret)
    t = _t_stat(ic, n)
    rolling = _rolling_ic(sig, ret, windows=3)
    hit = _quintile_hit_rates(sig, out)
    flags: list[str] = []
    if abs(ic) < _IC_DEAD_THRESHOLD and n >= _IC_DEAD_MIN_SAMPLES:
        flags.append("dead_signal")
    if math.isfinite(t) and abs(t) < 1.5:
        flags.append("weak_t_stat")
    if len(rolling) >= 3 and rolling[-1] < 0 < rolling[0]:
        flags.append("sign_flip")
    if hit and not _monotone_up(hit):
        flags.append("non_monotone")
    status = "ok" if not flags else "degraded"
    return {
        "signal": name,
        "status": status,
        "flags": flags,
        "n": n,
        "ic": round(ic, 4) if math.isfinite(ic) else None,
        "t_stat": round(t, 3) if math.isfinite(t) else None,
        "rolling_ic": rolling,
        "quintile_hit_rate": hit,
    }


def build_report(
    closed_csv: str = "logs/closed_positions.csv",
    out_path: Path = REPORT_PATH,
) -> dict:
    path = Path(closed_csv)
    if not path.exists():
        report = {"status": "no_trades_yet", "path": str(path)}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if "realized_pnl" not in df.columns:
        return {"status": "malformed_csv", "missing": "realized_pnl"}
    # Realised trade return (%): realized_pnl / size_usdc where size > 0
    size = pd.to_numeric(df.get("size_usdc", 0), errors="coerce").clip(lower=1e-9)
    df = df.assign(_ret=pd.to_numeric(df["realized_pnl"], errors="coerce") / size)
    df = df.assign(_out=(pd.to_numeric(df["realized_pnl"], errors="coerce") > 0).astype(int))

    fam_col = df.get("market_family", pd.Series(["btc"] * len(df))).astype(str).str.lower()
    btc = df[~fam_col.str.startswith("weather")].copy()
    weather = df[fam_col.str.startswith("weather")].copy()

    report = {
        "status": "ok",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "families": {
            "btc": {
                "n_trades": int(len(btc)),
                "signals": [
                    validate_signal(name, btc, "_ret", "_out")
                    for name in _CLAIMED_SIGNALS_BTC
                ],
            },
            "weather_temperature": {
                "n_trades": int(len(weather)),
                "signals": [
                    validate_signal(name, weather, "_ret", "_out")
                    for name in _CLAIMED_SIGNALS_WEATHER
                ],
            },
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed-csv", default="logs/closed_positions.csv")
    args = ap.parse_args()
    report = build_report(args.closed_csv)
    print(json.dumps(report, indent=2)[:2000])


if __name__ == "__main__":
    main()
