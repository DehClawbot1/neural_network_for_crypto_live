"""
services/monte_carlo_backtest.py
────────────────────────────────
Monte Carlo backtesting with stationary / circular block bootstrap.

What this buys over point-estimate walk-forward
-----------------------------------------------
A single walk-forward realisation gives ONE Sharpe, ONE maxDD, ONE CAGR.
That's a sample of size 1 from a noisy distribution. Live performance can
look arbitrarily worse than the backtest just from sample noise.

Monte Carlo bootstrap resamples the realised trade-return series B times
(B ≥ 2000), recomputes every metric on each synthetic path, and returns
confidence intervals. The block bootstrap preserves autocorrelation
(trade-to-trade streakiness, vol clustering) that iid bootstrap destroys.

References
----------
- Politis & Romano (1994), "The Stationary Bootstrap"
- López de Prado (2018), Ch. 11 — "The Dangers of Backtesting"
- Harvey & Liu (2014), "Backtesting" — PSR / DSR multiple-testing
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


# ─────────────────────────── bootstrap samplers ─────────────────────────
def circular_block_bootstrap(
    returns: np.ndarray,
    block_size: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Circular block bootstrap — wraps at the end so every position has equal
    resampling probability. Returns shape (n_samples, len(returns)).
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n == 0 or block_size <= 0:
        return np.empty((n_samples, 0), dtype=float)
    n_blocks = int(np.ceil(n / block_size))
    # Pre-concatenate a wrapped copy so slicing past the end works.
    wrapped = np.concatenate([r, r[: block_size - 1]]) if block_size > 1 else r
    out = np.empty((n_samples, n), dtype=float)
    for i in range(n_samples):
        starts = rng.integers(0, n, size=n_blocks)
        chunks = [wrapped[s:s + block_size] for s in starts]
        path = np.concatenate(chunks)[:n]
        out[i] = path
    return out


def stationary_bootstrap(
    returns: np.ndarray,
    mean_block: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Politis-Romano stationary bootstrap: geometric block lengths with mean
    `mean_block`. Strictly stationary resampled series.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n == 0:
        return np.empty((n_samples, 0), dtype=float)
    p = 1.0 / max(1.0, float(mean_block))
    out = np.empty((n_samples, n), dtype=float)
    for i in range(n_samples):
        path = np.empty(n, dtype=float)
        j = int(rng.integers(0, n))
        for t in range(n):
            path[t] = r[j]
            if rng.random() < p:
                j = int(rng.integers(0, n))
            else:
                j = (j + 1) % n
        out[i] = path
    return out


# ───────────────────────────── metrics ─────────────────────────────────
def _sharpe(r: np.ndarray, periods_per_year: float) -> float:
    if r.size < 2:
        return 0.0
    s = r.std(ddof=1)
    if s <= 1e-12:
        return 0.0
    return float(r.mean() / s * np.sqrt(periods_per_year))

def _max_drawdown(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    peak_max = float(np.max(peak)) if np.max(peak) > 1e-9 else 1.0
    return float(np.max(dd) / peak_max)

def _calmar(r: np.ndarray, periods_per_year: float) -> float:
    dd = _max_drawdown(r)
    if dd <= 1e-12:
        return 0.0
    cagr = r.mean() * periods_per_year
    return float(cagr / dd)


# ───────────────────── probabilistic / deflated Sharpe ─────────────────
def probabilistic_sharpe_ratio(sharpe: float, n: int, skew: float = 0.0,
                               kurt_excess: float = 0.0, threshold: float = 0.0) -> float:
    """
    Bailey & López de Prado (2012). Probability that the true Sharpe > threshold
    given the observed Sharpe, adjusting for non-normal higher moments.
    """
    if n < 5 or not np.isfinite(sharpe):
        return float("nan")
    denom_sq = 1.0 - skew * sharpe + (kurt_excess / 4.0) * sharpe * sharpe
    if denom_sq <= 0:
        return float("nan")
    z = (sharpe - threshold) * np.sqrt(n - 1) / np.sqrt(denom_sq)
    return float(_norm_cdf(z))


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))


def _erf(x: float) -> float:
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


# ──────────────────────────── main driver ──────────────────────────────
@dataclass
class MCResult:
    n_trades: int
    n_simulations: int
    block_size: int
    method: str                 # "circular" or "stationary"
    point_sharpe: float
    point_calmar: float
    point_max_dd: float
    sharpe_ci: tuple[float, float]
    calmar_ci: tuple[float, float]
    max_dd_ci: tuple[float, float]
    p_sharpe_positive: float   # fraction of bootstrapped Sharpes > 0
    probabilistic_sharpe: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sharpe_ci"] = list(d["sharpe_ci"])
        d["calmar_ci"] = list(d["calmar_ci"])
        d["max_dd_ci"] = list(d["max_dd_ci"])
        return d


def monte_carlo_backtest(
    returns: np.ndarray,
    *,
    n_simulations: int = 2000,
    block_size: int = 10,
    method: str = "stationary",
    periods_per_year: float = 252.0,
    ci: tuple[float, float] = (0.05, 0.95),
    random_state: int = 42,
) -> MCResult:
    """
    Bootstrap the returns series `n_simulations` times and report
    confidence intervals on Sharpe, Calmar and max-drawdown.

    Parameters
    ----------
    returns : per-trade or per-period realised returns (as fractions).
    method  : "circular" (fixed block) or "stationary" (geometric blocks).
    block_size : fixed length for circular; mean length for stationary.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 10:
        return MCResult(r.size, 0, block_size, method, 0, 0, 0,
                        (0, 0), (0, 0), (0, 0), 0.0, float("nan"))

    rng = np.random.default_rng(random_state)
    if method == "circular":
        paths = circular_block_bootstrap(r, block_size, n_simulations, rng)
    else:
        paths = stationary_bootstrap(r, float(block_size), n_simulations, rng)

    sharpes = np.asarray([_sharpe(p, periods_per_year) for p in paths])
    calmars = np.asarray([_calmar(p, periods_per_year) for p in paths])
    dds = np.asarray([_max_drawdown(p) for p in paths])

    lo, hi = ci
    # Higher moments for PSR
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd > 1e-12:
        skew = float(((r - mu) ** 3).mean() / sd ** 3)
        kurt = float(((r - mu) ** 4).mean() / sd ** 4 - 3.0)
    else:
        skew = kurt = 0.0
    point_sharpe_val = _sharpe(r, periods_per_year)
    psr = probabilistic_sharpe_ratio(point_sharpe_val, r.size, skew, kurt, 0.0)

    return MCResult(
        n_trades=int(r.size),
        n_simulations=int(n_simulations),
        block_size=int(block_size),
        method=method,
        point_sharpe=round(point_sharpe_val, 4),
        point_calmar=round(_calmar(r, periods_per_year), 4),
        point_max_dd=round(_max_drawdown(r), 4),
        sharpe_ci=(round(float(np.quantile(sharpes, lo)), 4),
                   round(float(np.quantile(sharpes, hi)), 4)),
        calmar_ci=(round(float(np.quantile(calmars, lo)), 4),
                   round(float(np.quantile(calmars, hi)), 4)),
        max_dd_ci=(round(float(np.quantile(dds, lo)), 4),
                   round(float(np.quantile(dds, hi)), 4)),
        p_sharpe_positive=round(float((sharpes > 0).mean()), 4),
        probabilistic_sharpe=round(psr, 4) if np.isfinite(psr) else float("nan"),
    )


# ─────────────────────── Deflated Sharpe (DSR) ─────────────────────────
def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n: int,
    skew: float = 0.0,
    kurt_excess: float = 0.0,
) -> float:
    """
    Bailey & López de Prado (2014). Probability that the BEST-of-`n_trials`
    Sharpe is a true-positive — penalty for strategy-selection bias.
    """
    if n < 5 or n_trials < 1:
        return float("nan")
    # Expected max of N iid standard normals (Euler-Mascheroni approximation)
    euler = 0.5772156649
    emax = (1.0 - euler) * _inv_norm(1.0 - 1.0 / n_trials) + \
           euler * _inv_norm(1.0 - 1.0 / (n_trials * np.e))
    denom_sq = 1.0 - skew * sharpe + (kurt_excess / 4.0) * sharpe * sharpe
    if denom_sq <= 0:
        return float("nan")
    z = (sharpe - emax) * np.sqrt(n - 1) / np.sqrt(denom_sq)
    return float(_norm_cdf(z))


def _inv_norm(p: float) -> float:
    # Beasley-Springer-Moro approximation for inverse normal CDF.
    p = min(max(p, 1e-10), 1.0 - 1e-10)
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p <= phigh:
        q = p - 0.5; r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    q = np.sqrt(-2 * np.log(1 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


__all__ = [
    "MCResult",
    "monte_carlo_backtest",
    "circular_block_bootstrap",
    "stationary_bootstrap",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
]
