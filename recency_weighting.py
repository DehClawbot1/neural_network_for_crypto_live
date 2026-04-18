"""Recency weighting with exponential decay, minimum sample thresholds,
and Bayesian shrinkage toward a family-level prior.

PURPOSE
-------
Replace raw EWMA / tail(N) statistics that cause the system to overreact to
short runs of bad trades with a principled Bayesian estimator that:

  1. Weights recent trades more heavily via exponential decay
     (configurable half-life in trades, not arbitrary EWMA span)
  2. Refuses to move far from the prior until a minimum effective sample
     size is reached — 5 bad trades cannot thrash the governor
  3. Shrinks toward a family-level prior with strength proportional to how
     few samples we have — converges to truth as data accumulates

MATH
----
Exponential decay weights:
    w_i = exp(-λ * (N-1-i))   for i = 0 … N-1  (0 is oldest, N-1 is newest)
    λ   = ln(2) / half_life_trades
    Effective sample size n_eff = Σ w_i

Bayesian shrinkage (Normal-Normal conjugate for means):
    posterior_mean = (κ₀ · μ₀  +  n_eff · sample_mean)
                     ─────────────────────────────────
                           κ₀  +  n_eff
    κ₀  = prior_strength  (virtual trades in the prior — default 20)
    μ₀  = prior_mean      (family baseline — e.g. 0.50 for win rate)

Credibility (how much to trust data vs prior):
    credibility = n_eff / (n_eff + κ₀)   ∈ [0, 1]
    credibility → 0 : only the prior matters
    credibility → 1 : data fully dominates

Minimum sample gate:
    If n_eff < min_eff_samples, return prior_mean with credibility = 0.
    This is the hard guard against the "5 bad trades" overreaction.

Beta posterior for win rate (optional, more correct for binary outcomes):
    posterior_alpha = prior_strength/2 + decayed_wins
    posterior_beta  = prior_strength/2 + decayed_losses
    posterior_mean  = posterior_alpha / (posterior_alpha + posterior_beta)

PUBLIC API
----------
    # Core estimator
    est = DecayedEstimator(half_life=25, prior_mean=0.50, prior_strength=20, min_eff_samples=15)
    est.update(pnl_array)
    win_rate, credibility = est.win_rate()
    mean_pnl, credibility = est.mean()

    # Family-level manager (all metrics for one family)
    family = FamilyRecencyEstimator("btc")
    family.ingest(closed_df)
    wr, cred = family.win_rate()
    dd, cred = family.max_drawdown()

    # Scoped manager (per signal_label / regime / etc.)
    store = ScopedRecencyStore("btc")
    store.ingest(feedback_report_df)
    wr, cred = store.win_rate(scope="signal_label", value="momentum_long")

INTEGRATION
-----------
performance_governor.FamilyPerformanceGovernor.evaluate():
    Uses FamilyRecencyEstimator instead of raw EWMA.

trade_feedback_learner._refresh_summary() and _compute_feedback_factors():
    Uses ScopedRecencyStore instead of tail(N) mean.

CONFIGURATION (env vars, all optional)
---------------------------------------
    RECENCY_HALF_LIFE_TRADES       default 25
    RECENCY_PRIOR_STRENGTH         default 20
    RECENCY_MIN_EFF_SAMPLES        default 15
    RECENCY_BTC_PRIOR_WIN_RATE     default 0.50   (family prior for win rate)
    RECENCY_BTC_PRIOR_MEAN_PNL     default 0.0
    RECENCY_WEATHER_PRIOR_WIN_RATE default 0.50
    RECENCY_WEATHER_PRIOR_MEAN_PNL default 0.0
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name, "").strip()
        return float(raw) if raw else default
    except ValueError:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name, "").strip()
        return int(raw) if raw else default
    except ValueError:
        return default

# Defaults — deliberately conservative
_DEFAULT_HALF_LIFE   = 25   # recent 25 trades get >50% of total weight
_DEFAULT_PRIOR_STR   = 20   # 20 "virtual trades" of prior before data moves the needle
_DEFAULT_MIN_EFF     = 15   # must see 15 effective samples before trusting the estimate


# ---------------------------------------------------------------------------
# Core: DecayedEstimator
# ---------------------------------------------------------------------------

class DecayedEstimator:
    """
    Single-metric estimator with exponential decay and Bayesian shrinkage.

    Usage:
        est = DecayedEstimator(half_life=25, prior_mean=0.5, prior_strength=20, min_eff_samples=15)
        est.update(np.array([1, 0, 0, 1, 1, ...]))   # binary: 1=win, 0=loss
        wr, cred = est.win_rate()   # Bayesian posterior win rate + credibility
        mu, cred = est.mean()       # Bayesian posterior mean + credibility
    """

    def __init__(
        self,
        half_life: float   = _DEFAULT_HALF_LIFE,
        prior_mean: float  = 0.5,
        prior_strength: float = _DEFAULT_PRIOR_STR,
        min_eff_samples: float = _DEFAULT_MIN_EFF,
    ) -> None:
        if half_life <= 0:
            raise ValueError("half_life must be > 0")
        self.half_life      = float(half_life)
        self.prior_mean     = float(prior_mean)
        self.prior_strength = float(prior_strength)
        self.min_eff_samples = float(min_eff_samples)
        self._lam           = math.log(2) / self.half_life   # decay rate

        # Stored from last update
        self._values: np.ndarray = np.array([], dtype=float)
        self._weights: np.ndarray = np.array([], dtype=float)

    # -- Update --------------------------------------------------------------

    def update(self, values: np.ndarray | list | pd.Series) -> "DecayedEstimator":
        """
        Replace stored values with the new series (already sorted oldest→newest).
        Computes exponential decay weights so the newest observation has weight 1.0
        and weights decay toward zero as we go back in time.
        """
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            self._values  = np.array([], dtype=float)
            self._weights = np.array([], dtype=float)
            return self
        n = len(arr)
        # w[i] = exp(-λ * (n-1-i))  →  newest element has weight exp(0)=1.0
        indices   = np.arange(n, dtype=float)
        self._weights = np.exp(-self._lam * (n - 1 - indices))
        self._values  = arr
        return self

    # -- Effective sample size -----------------------------------------------

    @property
    def n_eff(self) -> float:
        """Sum of decay weights ≈ effective number of equally-weighted samples."""
        return float(self._weights.sum()) if len(self._weights) > 0 else 0.0

    @property
    def credibility(self) -> float:
        """
        Fraction in [0, 1] indicating how much the data (vs prior) should be
        trusted.  0 = no data, use prior.  1 = fully trust the data.

        Respects min_eff_samples: credibility is 0 until the minimum is met.
        """
        n = self.n_eff
        if n < self.min_eff_samples:
            return 0.0
        return float(n / (n + self.prior_strength))

    def is_reliable(self) -> bool:
        return self.n_eff >= self.min_eff_samples

    # -- Posterior statistics ------------------------------------------------

    def mean(self) -> tuple[float, float]:
        """
        Bayesian posterior mean of the values (Normal-Normal conjugate).
        Returns (posterior_mean, credibility).
        """
        cred = self.credibility
        if cred == 0.0 or len(self._values) == 0:
            return self.prior_mean, 0.0
        w_sum   = float(self._weights.sum())
        w_mean  = float((self._weights * self._values).sum() / w_sum) if w_sum > 0 else self.prior_mean
        post    = (self.prior_strength * self.prior_mean + self.n_eff * w_mean) / (self.prior_strength + self.n_eff)
        return float(post), float(cred)

    def win_rate(self) -> tuple[float, float]:
        """
        Bayesian posterior win rate using a Beta conjugate prior.

        Equivalent to adding (prior_strength/2) virtual wins and
        (prior_strength/2) virtual losses, then observing the decayed counts.
        Returns (posterior_win_rate, credibility).
        """
        cred = self.credibility
        if cred == 0.0 or len(self._values) == 0:
            return self.prior_mean, 0.0
        # Decayed win/loss counts
        wins   = float((self._weights * (self._values > 0).astype(float)).sum())
        losses = float((self._weights * (self._values <= 0).astype(float)).sum())
        # Beta prior: α₀ = β₀ = prior_strength/2
        alpha = self.prior_strength / 2.0 + wins
        beta  = self.prior_strength / 2.0 + losses
        post_wr = alpha / (alpha + beta)
        return float(post_wr), float(cred)

    def std(self) -> float:
        """Weighted standard deviation of the values (not the posterior std)."""
        if len(self._values) < 2:
            return float("nan")
        w_sum  = float(self._weights.sum())
        w_mean = float((self._weights * self._values).sum() / w_sum)
        variance = float((self._weights * (self._values - w_mean) ** 2).sum() / w_sum)
        return float(math.sqrt(max(0.0, variance)))

    def max_drawdown(self) -> tuple[float, float]:
        """
        Maximum rolling drawdown on the cumulative equity curve of values,
        returned as a fraction of peak (e.g. 0.15 = 15% drawdown).
        Credibility is the same as for mean().
        """
        cred = self.credibility
        if len(self._values) < 2:
            return 0.0, cred
        equity = 1.0 + np.cumsum(self._values)
        peak   = np.maximum.accumulate(equity)
        with np.errstate(invalid="ignore", divide="ignore"):
            dd = np.where(peak != 0, (equity - peak) / np.abs(peak), 0.0)
        return float(abs(dd.min())), float(cred)

    def summary(self) -> dict[str, Any]:
        mu, cred = self.mean()
        wr, _    = self.win_rate()
        dd, _    = self.max_drawdown()
        return {
            "n_eff":        round(self.n_eff, 2),
            "n_raw":        int(len(self._values)),
            "credibility":  round(cred, 4),
            "is_reliable":  self.is_reliable(),
            "posterior_mean": round(mu, 6),
            "posterior_win_rate": round(wr, 4),
            "max_drawdown": round(dd, 4),
            "prior_mean":   self.prior_mean,
            "prior_strength": self.prior_strength,
            "half_life":    self.half_life,
            "min_eff_samples": self.min_eff_samples,
        }


# ---------------------------------------------------------------------------
# Family-level manager
# ---------------------------------------------------------------------------

# Family priors — conservative baselines, not optimistic ones
_FAMILY_PRIORS: dict[str, dict[str, float]] = {
    "btc": {
        "win_rate":  _env_float("RECENCY_BTC_PRIOR_WIN_RATE",  0.50),
        "mean_pnl":  _env_float("RECENCY_BTC_PRIOR_MEAN_PNL",  0.0),
        "edge_decay": 0.0,
    },
    "weather": {
        "win_rate":  _env_float("RECENCY_WEATHER_PRIOR_WIN_RATE", 0.50),
        "mean_pnl":  _env_float("RECENCY_WEATHER_PRIOR_MEAN_PNL", 0.0),
        "edge_decay": 0.0,
    },
}


def _family_prior(family: str, metric: str) -> float:
    key = "btc" if "btc" in family.lower() else "weather"
    return _FAMILY_PRIORS.get(key, _FAMILY_PRIORS["btc"]).get(metric, 0.5)


class FamilyRecencyEstimator:
    """
    Maintains decayed + shrinkage-adjusted estimates for one market family.

    Designed to replace the raw EWMA statistics in FamilyPerformanceGovernor.
    All estimates are anchored to the family prior until enough data accumulates.

    Usage:
        est = FamilyRecencyEstimator("btc")
        est.ingest(closed_positions_df)
        wr, cred = est.win_rate()    # (float, float)
        dd, cred = est.max_drawdown()
        mu, cred = est.avg_pnl()
        ed, cred = est.edge_decay()

        # Governor-ready values (already credibility-weighted):
        gov_wr = est.governor_win_rate()   # credibility-weighted blend of posterior+prior
    """

    def __init__(
        self,
        family: str,
        half_life:     float = _DEFAULT_HALF_LIFE,
        prior_strength: float = _DEFAULT_PRIOR_STR,
        min_eff_samples: float = _DEFAULT_MIN_EFF,
    ) -> None:
        self.family = family.lower()
        hl  = _env_float("RECENCY_HALF_LIFE_TRADES", half_life)
        ps  = _env_float("RECENCY_PRIOR_STRENGTH",   prior_strength)
        mes = _env_float("RECENCY_MIN_EFF_SAMPLES",  min_eff_samples)

        pwr   = _family_prior(self.family, "win_rate")
        ppnl  = _family_prior(self.family, "mean_pnl")
        pedge = _family_prior(self.family, "edge_decay")

        self._wr_est   = DecayedEstimator(hl, prior_mean=pwr,   prior_strength=ps, min_eff_samples=mes)
        self._pnl_est  = DecayedEstimator(hl, prior_mean=ppnl,  prior_strength=ps, min_eff_samples=mes)
        self._edge_est = DecayedEstimator(hl, prior_mean=pedge, prior_strength=ps, min_eff_samples=mes)
        self._dd_est   = DecayedEstimator(hl, prior_mean=0.0,   prior_strength=ps, min_eff_samples=mes)

    # -- Ingestion -----------------------------------------------------------

    def ingest(
        self,
        df: pd.DataFrame,
        pnl_col: str | None = None,
        ev_col: str = "expected_return",
        slip_col: str = "exit_realized_slippage_bps",
        max_rows: int = 200,
    ) -> "FamilyRecencyEstimator":
        """
        Update all estimators from a closed positions / canonical DataFrame.
        Automatically selects the PnL column and sorts by time.
        """
        if df is None or df.empty:
            return self

        work = df.copy()

        # Family filter
        if "market_family" in work.columns:
            mf = work["market_family"].astype(str).str.lower()
            if "weather" in self.family:
                work = work[mf.str.startswith("weather")]
            else:
                work = work[~mf.str.startswith("weather")]

        if work.empty:
            return self

        # Sort by time
        for ts in ("closed_at", "timestamp", "fill_timestamp"):
            if ts in work.columns:
                work[ts] = pd.to_datetime(work[ts], errors="coerce", utc=True)
                work = work.sort_values(ts)
                break

        work = work.tail(max_rows)

        # PnL series
        if pnl_col and pnl_col in work.columns:
            _pnl_col = pnl_col
        else:
            _pnl_col = next(
                (c for c in ("price_return", "net_realized_pnl", "realized_pnl") if c in work.columns),
                None,
            )
        if _pnl_col is None:
            log.warning("[FamilyRecencyEstimator/%s] no PnL column found", self.family)
            return self

        pnl = pd.to_numeric(work[_pnl_col], errors="coerce").to_numpy()

        # Update win-rate estimator with binary wins (1/0)
        self._wr_est.update((pnl > 0).astype(float))
        self._pnl_est.update(pnl)

        # Drawdown on the raw PnL stream
        self._dd_est.update(pnl)   # max_drawdown() will compute the rolling curve

        # Edge decay: realized_cost - expected_value
        if ev_col in work.columns and slip_col in work.columns:
            ev   = pd.to_numeric(work[ev_col],   errors="coerce").fillna(0.0).to_numpy()
            slip = pd.to_numeric(work[slip_col], errors="coerce").fillna(0.0).to_numpy()
            cost = slip * 0.0001 * 2.0   # bps → fraction, round-trip
            decay = cost - ev
            self._edge_est.update(decay)

        return self

    # -- Metric accessors ----------------------------------------------------

    def win_rate(self) -> tuple[float, float]:
        """Returns (posterior_win_rate, credibility)."""
        return self._wr_est.win_rate()

    def avg_pnl(self) -> tuple[float, float]:
        """Returns (posterior_mean_pnl, credibility)."""
        return self._pnl_est.mean()

    def max_drawdown(self) -> tuple[float, float]:
        """Returns (max_drawdown_fraction, credibility)."""
        return self._dd_est.max_drawdown()

    def edge_decay(self) -> tuple[float, float]:
        """Returns (posterior_mean_edge_decay, credibility)."""
        return self._edge_est.mean()

    def is_reliable(self) -> bool:
        """True when the win-rate estimator has enough effective samples."""
        return self._wr_est.is_reliable()

    def n_eff(self) -> float:
        return self._wr_est.n_eff

    # -- Governor-ready values -----------------------------------------------

    def governor_win_rate(self) -> float:
        """
        Single float suitable for governor threshold comparison.
        If below min_eff_samples: returns prior (no action taken).
        Otherwise: returns Bayesian posterior win rate.
        """
        wr, _ = self.win_rate()
        return wr

    def governor_drawdown(self) -> float:
        """Single float: max drawdown on cumulative PnL equity curve."""
        dd, _ = self.max_drawdown()
        return dd

    def governor_edge_decay(self) -> float:
        """Single float: mean edge decay (cost - EV)."""
        ed, _ = self.edge_decay()
        return ed

    def summary(self) -> dict[str, Any]:
        wr, wr_cred = self.win_rate()
        mu, mu_cred = self.avg_pnl()
        dd, dd_cred = self.max_drawdown()
        ed, ed_cred = self.edge_decay()
        return {
            "family":          self.family,
            "n_eff":           round(self.n_eff(), 2),
            "is_reliable":     self.is_reliable(),
            "win_rate":        round(wr, 4),
            "win_rate_cred":   round(wr_cred, 4),
            "avg_pnl":         round(mu, 6),
            "avg_pnl_cred":    round(mu_cred, 4),
            "max_drawdown":    round(dd, 4),
            "drawdown_cred":   round(dd_cred, 4),
            "edge_decay":      round(ed, 6),
            "edge_decay_cred": round(ed_cred, 4),
        }


# ---------------------------------------------------------------------------
# Scoped manager (per signal_label, regime, etc.)
# ---------------------------------------------------------------------------

class ScopedRecencyStore:
    """
    Maintains a separate FamilyRecencyEstimator per (scope, value) pair.
    Used by trade_feedback_learner to replace the per-scope tail(N) approach.

    min_eff_samples is enforced per scope: if a scope key has fewer effective
    samples, its estimate falls back to the family-level estimate (not the
    pure prior) — this provides a sensible intermediate anchor.

    Usage:
        store = ScopedRecencyStore("btc")
        store.ingest(feedback_report_df)
        wr, cred = store.win_rate("signal_label", "momentum_long")
        wr, cred = store.win_rate("technical_regime_bucket", "chaotic")
    """

    def __init__(
        self,
        family: str,
        half_life: float = _DEFAULT_HALF_LIFE,
        prior_strength: float = _DEFAULT_PRIOR_STR,
        min_eff_samples: float = _DEFAULT_MIN_EFF,
    ) -> None:
        self.family = family.lower()
        self._hl  = _env_float("RECENCY_HALF_LIFE_TRADES", half_life)
        self._ps  = _env_float("RECENCY_PRIOR_STRENGTH",   prior_strength)
        self._mes = _env_float("RECENCY_MIN_EFF_SAMPLES",  min_eff_samples)

        # Family-level estimator serves as "scoped prior" for low-sample scopes
        self._family_est = FamilyRecencyEstimator(family, self._hl, self._ps, self._mes)
        # {(scope, value): DecayedEstimator}
        self._scopes: dict[tuple[str, str], DecayedEstimator] = {}

    def ingest(
        self,
        df: pd.DataFrame,
        pnl_col: str | None = None,
        scopes: list[str] | None = None,
    ) -> "ScopedRecencyStore":
        """
        Update the family estimator and all per-scope estimators.
        df should be the feedback_report CSV, sorted oldest→newest.
        """
        if df is None or df.empty:
            return self

        self._family_est.ingest(df, pnl_col=pnl_col)

        _scopes = scopes or [
            "signal_label", "market_family", "horizon_bucket",
            "liquidity_bucket", "volatility_bucket",
            "technical_regime_bucket", "exit_reason_family",
        ]

        _pnl_col = pnl_col
        if _pnl_col is None:
            _pnl_col = next(
                (c for c in ("realized_pnl", "price_return", "net_realized_pnl") if c in df.columns),
                None,
            )
        if _pnl_col is None:
            return self

        # Sort once
        work = df.copy()
        for ts in ("closed_at", "timestamp"):
            if ts in work.columns:
                work[ts] = pd.to_datetime(work[ts], errors="coerce", utc=True)
                work = work.sort_values(ts)
                break

        for scope in _scopes:
            if scope not in work.columns:
                continue
            for val, grp in work.groupby(scope, dropna=True):
                key = (scope, str(val))
                pnl = pd.to_numeric(grp[_pnl_col], errors="coerce").dropna().to_numpy()
                if len(pnl) == 0:
                    continue
                family_prior_wr, _ = self._family_est.win_rate()
                family_prior_pnl, _ = self._family_est.avg_pnl()
                est = DecayedEstimator(
                    half_life=self._hl,
                    prior_mean=family_prior_pnl,   # scope prior = family estimate
                    prior_strength=self._ps,
                    min_eff_samples=self._mes,
                )
                # Override prior_mean for win_rate estimator
                est_wr = DecayedEstimator(
                    half_life=self._hl,
                    prior_mean=family_prior_wr,
                    prior_strength=self._ps,
                    min_eff_samples=self._mes,
                )
                est.update(pnl)
                est_wr.update((pnl > 0).astype(float))
                self._scopes[key] = est
                self._scopes[(scope + "_wr", str(val))] = est_wr

        return self

    def win_rate(self, scope: str, value: str) -> tuple[float, float]:
        """
        Posterior win rate for (scope, value).
        Falls back to family posterior if insufficient scoped data.
        """
        key = (scope + "_wr", str(value))
        if key in self._scopes:
            est = self._scopes[key]
            if est.is_reliable():
                return est.win_rate()
            # Blend: (1-cred)*family_wr + cred*scoped_wr
            scoped_wr, cred = est.win_rate()
            family_wr, _    = self._family_est.win_rate()
            blended = (1 - cred) * family_wr + cred * scoped_wr
            return float(blended), float(cred)
        return self._family_est.win_rate()

    def avg_pnl(self, scope: str, value: str) -> tuple[float, float]:
        """Posterior mean PnL for (scope, value)."""
        key = (scope, str(value))
        if key in self._scopes:
            est = self._scopes[key]
            if est.is_reliable():
                return est.mean()
            scoped_mu, cred = est.mean()
            family_mu, _    = self._family_est.avg_pnl()
            blended = (1 - cred) * family_mu + cred * scoped_mu
            return float(blended), float(cred)
        return self._family_est.avg_pnl()

    def is_reliable(self, scope: str, value: str) -> bool:
        key = (scope, str(value))
        return key in self._scopes and self._scopes[key].is_reliable()

    def all_summaries(self) -> list[dict[str, Any]]:
        rows = [{"scope": "family", "scope_value": self.family, **self._family_est.summary()}]
        seen: set[tuple[str, str]] = set()
        for (sc, val), est in self._scopes.items():
            if sc.endswith("_wr"):
                continue
            if (sc, val) in seen:
                continue
            seen.add((sc, val))
            mu, cred = est.mean()
            rows.append({
                "scope": sc, "scope_value": val,
                "n_eff": round(est.n_eff, 2),
                "is_reliable": est.is_reliable(),
                "posterior_mean_pnl": round(mu, 6),
                "credibility": round(cred, 4),
            })
        return rows


# ---------------------------------------------------------------------------
# Convenience factory — reads env vars once
# ---------------------------------------------------------------------------

def make_family_estimator(family: str) -> FamilyRecencyEstimator:
    return FamilyRecencyEstimator(
        family,
        half_life      = _env_float("RECENCY_HALF_LIFE_TRADES", _DEFAULT_HALF_LIFE),
        prior_strength = _env_float("RECENCY_PRIOR_STRENGTH",   _DEFAULT_PRIOR_STR),
        min_eff_samples= _env_float("RECENCY_MIN_EFF_SAMPLES",  _DEFAULT_MIN_EFF),
    )


def make_scoped_store(family: str) -> ScopedRecencyStore:
    return ScopedRecencyStore(
        family,
        half_life      = _env_float("RECENCY_HALF_LIFE_TRADES", _DEFAULT_HALF_LIFE),
        prior_strength = _env_float("RECENCY_PRIOR_STRENGTH",   _DEFAULT_PRIOR_STR),
        min_eff_samples= _env_float("RECENCY_MIN_EFF_SAMPLES",  _DEFAULT_MIN_EFF),
    )


# ---------------------------------------------------------------------------
# CLI — diagnostics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Recency weighting diagnostics")
    parser.add_argument("--family",   default="btc")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--simulate-n-bad", type=int, default=0,
                        help="Simulate N consecutive losing trades and show response")
    args = parser.parse_args()

    # Load real data
    est = make_family_estimator(args.family)
    for path in [
        Path(args.logs_dir) / args.family / "canonical_dataset.csv",
        Path(args.logs_dir) / "closed_positions.csv",
    ]:
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            est.ingest(df)
            print(f"Loaded from {path}  ({len(df)} rows)")
            break

    print("\n=== Family Recency Summary ===")
    print(json.dumps(est.summary(), indent=2))

    if args.simulate_n_bad > 0:
        print(f"\n=== Simulating {args.simulate_n_bad} consecutive -1% losses ===")
        sim = make_family_estimator(args.family)
        pnl_base = np.array([-0.01] * args.simulate_n_bad)
        sim.update = lambda x: None  # stub — ingest directly
        sim._wr_est.update((pnl_base > 0).astype(float))
        sim._pnl_est.update(pnl_base)
        sim._dd_est.update(pnl_base)

        wr, cred = sim.win_rate()
        print(f"  n_eff={sim.n_eff():.1f}  credibility={cred:.3f}")
        print(f"  posterior_win_rate={wr:.4f}  (prior=0.50)")
        print(f"  governor_win_rate={sim.governor_win_rate():.4f}")
        if not sim.is_reliable():
            print(f"  -> IS NOT RELIABLE (n_eff < {sim._wr_est.min_eff_samples}) — governor unchanged")
        else:
            print(f"  -> Reliable — governor would respond")
