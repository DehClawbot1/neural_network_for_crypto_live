"""Data-derived confidence cap calibrator.

Replaces the hand-picked magic-number caps (0.59 / 0.44 / 0.39 / 0.42) in
``SignalEngine.score_row`` with caps fitted on the closed-trade history.

Methodology
-----------
For each cap regime we identify the subset of historical trades that would
have matched the regime's predicate, bin their model confidence into deciles,
and pick the maximum confidence threshold at which the **realised win rate
still exceeds a floor** (default 0.50). That threshold becomes the cap: above
it the historical edge collapses, so production scores should not be allowed
to graduate past it.

The artefact is persisted as pickle-free JSON (schema_version=1) at
``logs/confidence_caps.json``. If the file is missing or has insufficient
samples the legacy defaults are returned, so the system is always safe.

Regimes
-------
* ``profitability_weak``  -> ``expected_return <= 0 or edge_score <= 0 or p_tp < 0.52``
* ``deep_negative``       -> ``expected_return < 0 and p_tp < 0.48``
* ``ta_conflict``         -> TA bias opposes target direction
* ``fractal_pending``     -> TA supports direction but fractal trigger not yet confirmed
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1

DEFAULT_CAPS: Dict[str, float] = {
    "profitability_weak": 0.59,
    "deep_negative": 0.44,
    "ta_conflict": 0.39,
    "fractal_pending": 0.42,
}


@dataclass
class CapArtefact:
    schema_version: int = SCHEMA_VERSION
    caps: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CAPS))
    sample_counts: Dict[str, int] = field(default_factory=dict)
    win_rate_floor: float = 0.50
    fitted_on: str = ""
    n_trades: int = 0


def _regime_mask(df: pd.DataFrame, regime: str) -> pd.Series:
    er = pd.to_numeric(df.get("expected_return", 0.0), errors="coerce").fillna(0.0)
    edge = pd.to_numeric(df.get("edge_score", 0.0), errors="coerce").fillna(0.0)
    p_tp = pd.to_numeric(df.get("p_tp_before_sl", 0.0), errors="coerce").fillna(0.0)
    bias = df.get("btc_trend_bias", pd.Series(["NEUTRAL"] * len(df))).astype(str).str.upper()
    direction = df.get("target_direction", pd.Series(["LONG"] * len(df))).astype(str).str.upper()
    long_fb = df.get("long_fractal_breakout", pd.Series([False] * len(df))).astype(bool)
    short_fb = df.get("short_fractal_breakout", pd.Series([False] * len(df))).astype(bool)

    if regime == "profitability_weak":
        return (er <= 0) | (edge <= 0) | (p_tp < 0.52)
    if regime == "deep_negative":
        return (er < 0) & (p_tp < 0.48)
    if regime == "ta_conflict":
        return ((bias == "LONG") & (direction == "SHORT")) | ((bias == "SHORT") & (direction == "LONG"))
    if regime == "fractal_pending":
        ta_support = bias.isin(["LONG", "SHORT"]) & (bias == direction)
        ready = ((direction == "LONG") & long_fb) | ((direction == "SHORT") & short_fb)
        return ta_support & (~ready)
    return pd.Series([False] * len(df))


def _fit_cap_for_regime(conf: np.ndarray, wins: np.ndarray, floor: float, default: float) -> float:
    """Find the largest confidence threshold at which realised win rate >= floor.

    Uses decile binning and cumulative-from-below evaluation so noise in any
    single bin cannot collapse the cap.
    """
    if conf.size < 20:
        return float(default)
    order = np.argsort(conf)
    c = conf[order]
    w = wins[order].astype(float)
    # Evaluate win rate of trades with confidence <= threshold at each decile
    deciles = np.quantile(c, np.linspace(0.1, 1.0, 10))
    best_cap = default
    for thr in deciles:
        mask = c <= thr
        if mask.sum() < 10:
            continue
        wr = float(w[mask].mean())
        if wr >= floor:
            best_cap = float(thr)
    # Clamp to sensible range so a degenerate fit cannot nuke the gate
    return float(np.clip(best_cap, 0.25, 0.90))


def fit_caps_from_closed_trades(
    csv_path: str | Path,
    out_path: str | Path = "logs/confidence_caps.json",
    win_rate_floor: float = 0.50,
    min_samples_per_regime: int = 30,
) -> CapArtefact:
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    if not csv_path.exists():
        art = CapArtefact()
        _persist(art, out_path)
        return art
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        art = CapArtefact()
        _persist(art, out_path)
        return art
    if df.empty or "realized_pnl" not in df.columns:
        art = CapArtefact()
        _persist(art, out_path)
        return art

    wins_full = (pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0) > 0).to_numpy()
    conf_col = pd.to_numeric(df.get("confidence_at_entry", np.nan), errors="coerce")
    if conf_col.isna().all():
        art = CapArtefact()
        _persist(art, out_path)
        return art
    conf_full = conf_col.fillna(0.0).to_numpy()

    caps: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for regime, default in DEFAULT_CAPS.items():
        mask = _regime_mask(df, regime).to_numpy()
        n = int(mask.sum())
        counts[regime] = n
        if n < min_samples_per_regime:
            caps[regime] = default
            continue
        caps[regime] = _fit_cap_for_regime(conf_full[mask], wins_full[mask], win_rate_floor, default)

    art = CapArtefact(
        caps=caps,
        sample_counts=counts,
        win_rate_floor=float(win_rate_floor),
        fitted_on=pd.Timestamp.utcnow().isoformat(),
        n_trades=int(len(df)),
    )
    _persist(art, out_path)
    return art


def _persist(art: CapArtefact, out_path: Path) -> None:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(art), indent=2))
    except Exception:
        pass


def load_caps(path: str | Path = "logs/confidence_caps.json") -> Dict[str, float]:
    p = Path(path)
    if not p.exists():
        return dict(DEFAULT_CAPS)
    try:
        data = json.loads(p.read_text())
        if int(data.get("schema_version", 0)) != SCHEMA_VERSION:
            return dict(DEFAULT_CAPS)
        caps = data.get("caps") or {}
        out = dict(DEFAULT_CAPS)
        for k, v in caps.items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
        return out
    except Exception:
        return dict(DEFAULT_CAPS)
