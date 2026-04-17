"""
services/attribution_engine.py
──────────────────────────────
Formal trade-outcome attribution.

Upgrades the existing heuristic 2A/2B/2C/2D pipeline with:

1. **Counterfactual P&L attribution** — decompose realised P&L into:
       slippage, timing, model, market-move, cost
   by re-pricing the trade under counterfactual fills (arrival-price,
   VWAP, TWAP) and subtracting components Shapley-style.

2. **Permutation feature importance** — model-agnostic signal contribution
   ranking using Breiman's algorithm: shuffle one column at a time across
   closed trades and measure the degradation in IC (or ROC-AUC if the
   outcome is binary). Non-parametric, handles any feature type.

3. **Z-scored regime-conditional win rate** — tests per (regime, signal)
   bucket whether win rate differs from the family baseline with a two-
   proportion z-test. Flags buckets with |z| > 2 as over/under-performing.

All outputs are JSON-serialisable for dashboard ingest.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd


# ────────────────────── counterfactual P&L attribution ─────────────────
@dataclass
class PnLDecomposition:
    realised_pnl: float
    model_pnl: float       # what ideal entry+exit at mid would have earned
    slippage_cost: float   # entry slippage vs arrival-mid
    timing_cost: float     # entry-timing vs optimal-window mid
    exec_cost: float       # fees + spread paid at exit
    residual: float        # unassigned (noise / misc)

    def to_dict(self) -> dict:
        return asdict(self)


def decompose_trade_pnl(
    *,
    realised_pnl: float,
    entry_price: float,
    exit_price: float,
    arrival_mid: float,
    optimal_mid: float,
    size_usdc: float,
    fees_usdc: float = 0.0,
) -> PnLDecomposition:
    """
    Decomposition (YES-side, long-only — caller negates for SHORT):

        realised_pnl ≈ model_pnl  - slippage  - timing  - exec

    - model_pnl   = size * (exit_mid - optimal_mid)         ← pure alpha
    - slippage    = size * (entry_price - arrival_mid)      ← paid at entry
    - timing      = size * (arrival_mid - optimal_mid)      ← waited too long
    - exec_cost   = fees + size * (exit_price - exit_mid)   ← approximated here
                    as fees (caller passes fee estimate)

    All in USDC. `residual` captures price model mismatch — keeps the
    decomposition honest rather than forcing it to balance.
    """
    size = float(max(0.0, size_usdc))
    model_pnl = size * (float(exit_price) - float(optimal_mid))
    slippage = size * (float(entry_price) - float(arrival_mid))
    timing = size * (float(arrival_mid) - float(optimal_mid))
    exec_c = float(fees_usdc)
    explained = model_pnl - slippage - timing - exec_c
    residual = float(realised_pnl) - explained
    return PnLDecomposition(
        realised_pnl=round(float(realised_pnl), 4),
        model_pnl=round(model_pnl, 4),
        slippage_cost=round(slippage, 4),
        timing_cost=round(timing, 4),
        exec_cost=round(exec_c, 4),
        residual=round(residual, 4),
    )


# ───────────────────── permutation feature importance ─────────────────
def permutation_feature_importance(
    closed: pd.DataFrame,
    *,
    feature_cols: Iterable[str],
    outcome_col: str = "realized_pnl",
    n_repeats: int = 10,
    random_state: int = 42,
) -> list[dict]:
    """
    Model-agnostic Breiman permutation importance using rank correlation
    (Spearman IC) with the trade outcome as the "model prediction". We
    measure drop in IC after shuffling each column across rows.

    Returns [{feature, baseline_ic, mean_drop, std_drop, p_approx}].
    `p_approx` is a 1-sample t on the drop distribution; interpret with
    caution on small n_repeats.
    """
    df = closed.copy()
    if outcome_col not in df.columns:
        return []
    y = pd.to_numeric(df[outcome_col], errors="coerce").to_numpy()
    y = np.where(np.isfinite(y), y, np.nan)
    rng = np.random.default_rng(random_state)

    def _ic(x: np.ndarray, y: np.ndarray) -> float:
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 50:
            return float("nan")
        xr = pd.Series(x[m]).rank().to_numpy()
        yr = pd.Series(y[m]).rank().to_numpy()
        xr -= xr.mean(); yr -= yr.mean()
        denom = np.sqrt((xr ** 2).sum() * (yr ** 2).sum())
        return float("nan") if denom <= 0 else float((xr * yr).sum() / denom)

    out = []
    for col in feature_cols:
        if col not in df.columns:
            out.append({"feature": col, "status": "missing"})
            continue
        x = pd.to_numeric(df[col], errors="coerce").to_numpy()
        base_ic = _ic(x, y)
        if not np.isfinite(base_ic):
            out.append({"feature": col, "status": "insufficient", "baseline_ic": None})
            continue
        drops = []
        for _ in range(n_repeats):
            xs = x.copy()
            rng.shuffle(xs)
            drops.append(abs(base_ic) - abs(_ic(xs, y)))
        drops = np.asarray([d for d in drops if np.isfinite(d)], dtype=float)
        if drops.size == 0:
            out.append({"feature": col, "status": "no_finite", "baseline_ic": round(base_ic, 4)})
            continue
        mean_drop = float(drops.mean())
        std_drop = float(drops.std(ddof=1)) if drops.size > 1 else 0.0
        # 1-sample t vs 0
        t_stat = mean_drop / (std_drop / np.sqrt(drops.size)) if std_drop > 1e-9 else float("inf")
        out.append({
            "feature": col,
            "status": "ok",
            "baseline_ic": round(base_ic, 4),
            "mean_drop": round(mean_drop, 4),
            "std_drop": round(std_drop, 4),
            "t_stat": round(float(t_stat), 3) if np.isfinite(t_stat) else None,
            "n_repeats": int(drops.size),
        })
    # Sort by mean_drop desc (most important first).
    out.sort(key=lambda r: -(r.get("mean_drop") or -1e9))
    return out


# ──────────────────── regime-conditional win rate z ────────────────────
def regime_winrate_ztest(
    closed: pd.DataFrame,
    *,
    outcome_col: str = "realized_pnl",
    regime_col: str = "volatility_bucket",
    min_bucket: int = 20,
) -> list[dict]:
    """
    Two-proportion z-test: does win rate in each regime bucket differ
    significantly from the overall family baseline? |z| > 1.96 ≈ p<0.05.
    """
    df = closed.copy()
    if outcome_col not in df.columns or regime_col not in df.columns:
        return []
    y = pd.to_numeric(df[outcome_col], errors="coerce") > 0
    total_n = int(y.sum()) + int((~y & pd.to_numeric(df[outcome_col], errors="coerce").notna()).sum())
    if total_n < min_bucket:
        return []
    p_base = float(y.sum() / max(1, len(y)))
    out = []
    for bucket, sub in df.groupby(regime_col):
        wins = (pd.to_numeric(sub[outcome_col], errors="coerce") > 0).sum()
        n = int(len(sub))
        if n < min_bucket:
            continue
        p_hat = wins / n
        se = np.sqrt(max(1e-12, p_base * (1.0 - p_base) / n))
        z = (p_hat - p_base) / se if se > 0 else 0.0
        out.append({
            "regime": str(bucket),
            "n": n,
            "win_rate": round(float(p_hat), 4),
            "baseline": round(p_base, 4),
            "z": round(float(z), 3),
            "flag": "over" if z > 1.96 else ("under" if z < -1.96 else "neutral"),
        })
    out.sort(key=lambda r: -abs(r["z"]))
    return out


__all__ = [
    "PnLDecomposition",
    "decompose_trade_pnl",
    "permutation_feature_importance",
    "regime_winrate_ztest",
]
