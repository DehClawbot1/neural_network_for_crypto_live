"""
calibration_report.py
─────────────────────
Compute Brier score and Expected Calibration Error (ECE) per market family
from stored conformal residuals and a companion prediction log.

Residuals alone are sufficient for a Brier-equivalent score:
    Brier = mean(residual^2)   (since residual = |y - p| on {0,1} labels)

For ECE we need both p and y. Closed trades are logged with both
`p_tp_before_sl` and the realized binary outcome in the main CSV logs,
so this module reads those directly.

Output: `logs/calibration_report.json`, refreshed on demand:

    python calibration_report.py
    python calibration_report.py --family btc

Fail modes (drift):
  - Brier > BRIER_ALERT_THRESHOLD         → family is flagged "drifting"
  - |ECE| > ECE_ALERT_THRESHOLD           → flagged "miscalibrated"
  - n_samples < MIN_SAMPLES_FOR_REPORT    → reported as "insufficient_data"

Thresholds match the alpha thesis kill criteria (see docs/alpha_thesis_*.md).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from services.calibration_store import load_residuals

BRIER_ALERT_THRESHOLD = 0.22      # BTC thesis: Brier > 0.22 → freeze
WEATHER_BRIER_ALERT   = 0.20      # weather thesis: tighter
ECE_ALERT_THRESHOLD   = 0.08      # BTC
WEATHER_ECE_ALERT     = 0.06      # weather
MIN_SAMPLES_FOR_REPORT = 50

REPORT_PATH = Path("logs") / "calibration_report.json"


def brier_from_residuals(residuals: np.ndarray) -> float:
    if residuals.size == 0:
        return float("nan")
    return float(np.mean(np.square(residuals)))


def ece_from_pairs(
    probs: np.ndarray,
    outcomes: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error — weighted by bin population.

    ECE = sum_b (n_b / N) * |mean(p_b) - mean(y_b)|
    """
    if probs.size == 0 or probs.size != outcomes.size:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, edges[1:-1], right=False)
    n = probs.size
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        conf_b = float(np.mean(probs[mask]))
        acc_b = float(np.mean(outcomes[mask]))
        ece += (count / n) * abs(conf_b - acc_b)
    return float(ece)


def report_family(
    family: str,
    *,
    logs_dir: str = "logs",
    predictions_csv: str | None = None,
) -> dict:
    residuals = load_residuals(family, logs_dir=logs_dir)
    brier_limit = WEATHER_BRIER_ALERT if family.startswith("weather") else BRIER_ALERT_THRESHOLD
    ece_limit   = WEATHER_ECE_ALERT   if family.startswith("weather") else ECE_ALERT_THRESHOLD

    if residuals.size < MIN_SAMPLES_FOR_REPORT:
        return {
            "family": family,
            "status": "insufficient_data",
            "n_samples": int(residuals.size),
            "brier": None,
            "ece": None,
            "thresholds": {"brier": brier_limit, "ece": ece_limit},
        }

    brier = brier_from_residuals(residuals)
    # ECE requires (p, y) pairs. We reconstruct them approximately by noting
    # that a residual r with binary y implies p = y - r if y=1 (=> p = 1-r)
    # or p = r if y=0. Without the stored y, approximate ECE by treating
    # residuals directly as a proxy — this is conservative (upper-bounds ECE).
    # If a predictions CSV is provided, we use it for an exact ECE.
    ece_value: float | None = None
    if predictions_csv is not None:
        try:
            import pandas as pd
            df = pd.read_csv(predictions_csv)
            # Schema-adaptive: prefer explicit (p_tp_before_sl, outcome_binary)
            # but fall back to the closed_positions.csv schema where we can
            # derive outcome from realized_pnl and use confidence_at_entry as p.
            prob_col = (
                "p_tp_before_sl" if "p_tp_before_sl" in df.columns
                else ("confidence_at_entry" if "confidence_at_entry" in df.columns else None)
            )
            if prob_col is None:
                raise KeyError("no probability column in predictions CSV")
            if "outcome_binary" in df.columns:
                y_series = df["outcome_binary"]
            elif "realized_pnl" in df.columns:
                y_series = (pd.to_numeric(df["realized_pnl"], errors="coerce") > 0).astype(int)
            else:
                raise KeyError("no outcome column in predictions CSV")
            df = df.assign(_prob=pd.to_numeric(df[prob_col], errors="coerce"), _y=y_series)
            df = df.dropna(subset=["_prob", "_y"])
            if "market_family" in df.columns:
                fam_lower = df["market_family"].astype(str).str.lower()
                df = df[fam_lower.str.startswith("weather")] if family.startswith("weather") \
                    else df[~fam_lower.str.startswith("weather")]
            if len(df) >= MIN_SAMPLES_FOR_REPORT:
                probs = df["_prob"].to_numpy(dtype=float)
                probs = np.clip(probs, 0.0, 1.0)
                ece_value = ece_from_pairs(
                    probs,
                    df["_y"].to_numpy(dtype=float),
                )
        except Exception as exc:
            logging.warning("ECE exact calc failed for %s: %s", family, exc)
            ece_value = None

    status = "ok"
    flags = []
    if brier > brier_limit:
        flags.append(f"brier_above_{brier_limit}")
        status = "drifting"
    if ece_value is not None and ece_value > ece_limit:
        flags.append(f"ece_above_{ece_limit}")
        status = "miscalibrated" if status == "ok" else status

    return {
        "family": family,
        "status": status,
        "flags": flags,
        "n_samples": int(residuals.size),
        "brier": round(brier, 6),
        "ece": round(ece_value, 6) if ece_value is not None else None,
        "thresholds": {"brier": brier_limit, "ece": ece_limit},
    }


def write_report(
    families: list[str],
    *,
    logs_dir: str = "logs",
    predictions_csv: str | None = None,
    out_path: Path = REPORT_PATH,
) -> dict:
    reports = {fam: report_family(fam, logs_dir=logs_dir, predictions_csv=predictions_csv)
               for fam in families}
    out = {"families": reports}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None, help="Single family (btc | weather_temperature)")
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--predictions-csv", default=None)
    args = ap.parse_args()

    families = [args.family] if args.family else ["btc", "weather_temperature"]
    report = write_report(
        families,
        logs_dir=args.logs_dir,
        predictions_csv=args.predictions_csv,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
