"""
services/family_stats_loader.py
────────────────────────────────
Derive FamilyStats for the PortfolioAllocator from closed_positions.csv.

Uses the last `window` closed trades per family. Conservative defaults:
window=200 (enough samples for Kelly shrinkage to stabilise but recent
enough to reflect current regime).

Returns {family: FamilyStats}; families with zero trades get a neutral
seed (priors that produce a small-but-nonzero pool so a new family can
start trading).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from services.portfolio_allocator import FamilyStats

_FAMILIES = ("btc", "weather_temperature")
_DEFAULT_WINDOW = 200


def _empty_stats(family: str) -> FamilyStats:
    # Neutral seed: 50% win rate, symmetric expected win/loss, tiny recent
    # history. Produces a small Kelly and a conservative vol scalar.
    return FamilyStats(
        family=family,
        n_trades=0,
        win_rate=0.5,
        avg_win=1.0,
        avg_loss=1.0,
        returns_series=np.array([], dtype=float),
        current_drawdown=0.0,
        peak_equity=0.0,
    )


def _compute_drawdown(returns: np.ndarray) -> tuple[float, float]:
    if returns.size == 0:
        return 0.0, 0.0
    equity = np.cumsum(returns)
    peak = np.maximum.accumulate(equity)
    dd_series = peak - equity
    # Express drawdown as a fraction of peak equity where peak > 0; fall
    # back to absolute when peak is small/zero.
    peak_val = float(np.max(peak)) if peak.size else 0.0
    if peak_val > 1e-6:
        dd_frac = float(dd_series[-1] / peak_val)
    else:
        dd_frac = float(max(0.0, -equity[-1]))  # cumulative loss from start
    return max(0.0, dd_frac), peak_val


def load_family_stats(
    closed_csv: str = "logs/closed_positions.csv",
    window: int = _DEFAULT_WINDOW,
) -> dict[str, FamilyStats]:
    path = Path(closed_csv)
    if not path.exists():
        return {fam: _empty_stats(fam) for fam in _FAMILIES}

    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return {fam: _empty_stats(fam) for fam in _FAMILIES}
    if df.empty or "realized_pnl" not in df.columns:
        return {fam: _empty_stats(fam) for fam in _FAMILIES}

    if "closed_at" in df.columns:
        df["_closed_at"] = pd.to_datetime(df["closed_at"], errors="coerce", utc=True)
        df = df.sort_values("_closed_at")

    fam_col = df.get("market_family", pd.Series(["btc"] * len(df))).astype(str).str.lower()
    size = pd.to_numeric(df.get("size_usdc", 0), errors="coerce").clip(lower=1e-6)
    df = df.assign(
        _pnl=pd.to_numeric(df["realized_pnl"], errors="coerce"),
        _ret=pd.to_numeric(df["realized_pnl"], errors="coerce") / size,
        _fam=np.where(fam_col.str.startswith("weather"), "weather_temperature", "btc"),
    )
    df = df.dropna(subset=["_pnl", "_ret"])

    out: dict[str, FamilyStats] = {}
    for fam in _FAMILIES:
        sub = df[df["_fam"] == fam].tail(window)
        n = len(sub)
        if n == 0:
            out[fam] = _empty_stats(fam)
            continue
        pnls = sub["_pnl"].to_numpy(dtype=float)
        rets = sub["_ret"].to_numpy(dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        win_rate = float(len(wins) / n) if n else 0.5
        avg_win = float(wins.mean()) if wins.size else 1.0
        avg_loss = float(-losses.mean()) if losses.size else 1.0
        dd, peak = _compute_drawdown(rets)
        out[fam] = FamilyStats(
            family=fam,
            n_trades=n,
            win_rate=win_rate,
            avg_win=max(avg_win, 1e-6),
            avg_loss=max(avg_loss, 1e-6),
            returns_series=rets,
            current_drawdown=dd,
            peak_equity=peak,
        )
    return out


__all__ = ["load_family_stats"]
