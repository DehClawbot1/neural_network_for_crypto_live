"""Blocker profit audit — replay reports by blocker reason.

For every gate/blocker that rejected a candidate, answers:
  "If this gate had NOT blocked, what was the 15m return,
   TP-before-SL rate, and realized replay PnL?"

Also runs feature-family ablation reports and regime-sliced
performance comparisons.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ── blocker replay ──────────────────────────────────────────────────

def _filter_by_family(df: pd.DataFrame, market_family: str | None) -> pd.DataFrame:
    """Filter a DataFrame to rows matching market_family, if family is specified."""
    if not market_family or "market_family" not in df.columns:
        return df
    fam = market_family.lower()
    mask = df["market_family"].astype(str).str.lower().str.startswith(fam)
    return df[mask].copy()


def _is_btc_family(market_family: str | None) -> bool:
    if not market_family:
        return True  # default assumption: BTC unless told otherwise
    return not market_family.lower().startswith("weather")


def blocker_replay_report(logs_dir: str = "logs", market_family: str | None = None) -> pd.DataFrame:
    """For each reject_reason, compute hypothetical forward metrics.

    Joins candidate_decisions (rejected) with contract_targets (outcomes)
    on token_id to see what would have happened.

    Pass market_family="btc" or market_family="weather_temperature" to restrict
    the report to one family.  Mixed-family reports produce invalid metrics because
    BTC path columns (tp_before_sl_60m, mfe_60m, mae_60m) are meaningless for
    weather probability-space contracts.
    """
    logs_path = Path(logs_dir)
    decisions = _safe_read(logs_path / "candidate_decisions.csv")
    targets = _safe_read(logs_path / "contract_targets.csv")

    if decisions.empty or targets.empty:
        logger.warning("Insufficient data for blocker replay.")
        return pd.DataFrame()

    # Apply family filter before any metric computation
    decisions = _filter_by_family(decisions, market_family)
    targets = _filter_by_family(targets, market_family)

    if decisions.empty or targets.empty:
        logger.warning("No data after family filter (%s) for blocker replay.", market_family)
        return pd.DataFrame()

    rejected = decisions[
        decisions.get("final_decision", pd.Series("", dtype=str)).astype(str).isin(
            ["REJECTED", "SKIPPED"]
        )
    ].copy()
    if rejected.empty:
        return pd.DataFrame()

    # join with contract targets on token_id
    if "token_id" not in rejected.columns or "token_id" not in targets.columns:
        return pd.DataFrame()

    # BTC-only path metrics — only include when we know we're in BTC family
    btc_path_cols = ["tp_before_sl_60m", "mfe_60m", "mae_60m"] if _is_btc_family(market_family) else []
    target_col_candidates = ["token_id", "forward_return_15m"] + btc_path_cols
    target_cols = [c for c in target_col_candidates if c in targets.columns]
    target_dedup = targets[target_cols].drop_duplicates(subset=["token_id"], keep="last")

    merged = rejected.merge(target_dedup, on="token_id", how="inner")
    if merged.empty:
        return pd.DataFrame()

    reject_col = "reject_reason" if "reject_reason" in merged.columns else "gate"
    if reject_col not in merged.columns:
        return pd.DataFrame()

    rows = []
    for reason, group in merged.groupby(reject_col, dropna=False):
        reason_str = str(reason or "unknown").strip()
        if not reason_str:
            continue

        def _col_mean(col_name, _g=group):
            if col_name not in _g.columns:
                return None
            s = pd.to_numeric(_g[col_name], errors="coerce")
            return float(s.mean()) if s.notna().any() else None

        def _col_median(col_name, _g=group):
            if col_name not in _g.columns:
                return None
            s = pd.to_numeric(_g[col_name], errors="coerce")
            return float(s.median()) if s.notna().any() else None

        fwd_mean = _col_mean("forward_return_15m")
        row = {
            "reject_reason": reason_str,
            "n_blocked": len(group),
            "mean_forward_return_15m": fwd_mean,
            "median_forward_return_15m": _col_median("forward_return_15m"),
            "replay_ev": (fwd_mean * len(group)) if fwd_mean is not None else None,
        }
        # BTC-only path metrics — only compute when we know the family is BTC
        if _is_btc_family(market_family):
            row["tp_before_sl_rate"] = _col_mean("tp_before_sl_60m")
            row["mean_mfe_60m"] = _col_mean("mfe_60m")
            row["mean_mae_60m"] = _col_mean("mae_60m")
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("replay_ev", ascending=False, na_position="last")
    return result


# ── feature-family ablation ─────────────────────────────────────────

def feature_family_ablation(logs_dir: str = "logs", market_family: str | None = None) -> pd.DataFrame:
    """Report model performance when trained on each feature family alone.

    Reads pre-computed ``contract_targets.csv`` and trains a quick
    logistic model per family to measure isolated predictive power.

    Uses the primary classifier target for the specified market family:
    - BTC: tp_before_sl_60m  (price-space TP/SL gate hit label)
    - Weather: target_up / weather_contract_resolved_yes  (resolution label)
    """
    from model_feature_catalog import TRAINING_FEATURE_FAMILIES
    from model_feature_safety import drop_all_nan_features

    logs_path = Path(logs_dir)
    df = _safe_read(logs_path / "contract_targets.csv")
    if df.empty:
        return pd.DataFrame()

    df = _filter_by_family(df, market_family)
    if df.empty:
        return pd.DataFrame()

    # Select the correct classifier target for this family
    if _is_btc_family(market_family):
        target_col = "tp_before_sl_60m"
    else:
        # Weather: prefer resolved label, fall back to directional target
        target_col = next(
            (c for c in ["weather_contract_resolved_yes", "target_up"] if c in df.columns),
            None,
        )
    if target_col is None or target_col not in df.columns:
        logger.warning("No valid classifier target for family=%s in feature_family_ablation", market_family)
        return pd.DataFrame()

    y = df[target_col].fillna(0).astype(int)
    if y.nunique() < 2 or len(df) < 10:
        return pd.DataFrame()

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning("sklearn not available for feature-family ablation.")
        return pd.DataFrame()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")

    n_splits = min(3, len(df) - 1)
    if n_splits < 2:
        return pd.DataFrame()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for family_name, features in TRAINING_FEATURE_FAMILIES.items():
        available = [c for c in features if c in df.columns]
        available, _ = drop_all_nan_features(df, available, context=f"ablation_{family_name}")
        if len(available) < 2:
            continue

        accs = []
        for train_idx, test_idx in tscv.split(df):
            train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
            y_tr = train_df[target_col].fillna(0).astype(int)
            if y_tr.nunique() < 2:
                continue
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ])
            pipe.fit(train_df[available], y_tr)
            preds = pipe.predict(test_df[available])
            y_te = test_df[target_col].fillna(0).astype(int)
            accs.append(accuracy_score(y_te, preds))

        if accs:
            rows.append({
                "family": family_name,
                "n_features": len(available),
                "accuracy": sum(accs) / len(accs),
            })

    return pd.DataFrame(rows)


# ── regime performance ──────────────────────────────────────────────

def regime_performance_report(logs_dir: str = "logs", market_family: str | None = None) -> pd.DataFrame:
    """Performance by regime (calm/trend/volatile/chaotic) from closed trades."""
    logs_path = Path(logs_dir)
    closed = _safe_read(logs_path / "closed_positions.csv")
    if closed.empty:
        return pd.DataFrame()
    closed = _filter_by_family(closed, market_family)

    regime_col = None
    for candidate in ["technical_regime_bucket", "btc_market_regime_label", "btc_volatility_regime"]:
        if candidate in closed.columns:
            regime_col = candidate
            break
    if regime_col is None:
        return pd.DataFrame()

    pnl_col = "net_realized_pnl" if "net_realized_pnl" in closed.columns else "realized_pnl"
    if pnl_col not in closed.columns:
        return pd.DataFrame()

    rows = []
    for regime, group in closed.groupby(regime_col, dropna=False):
        regime_str = str(regime or "unknown").strip()
        pnl = pd.to_numeric(group[pnl_col], errors="coerce")
        wins = (pnl > 0).sum()
        total = pnl.notna().sum()
        rows.append({
            "regime": regime_str,
            "n_trades": total,
            "win_rate": wins / total if total > 0 else None,
            "mean_pnl": pnl.mean() if pnl.notna().any() else None,
            "total_pnl": pnl.sum() if pnl.notna().any() else None,
            "profit_factor": (
                pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())
                if (pnl > 0).any() and (pnl < 0).any()
                else None
            ),
        })

    return pd.DataFrame(rows)


# ── combined audit ──────────────────────────────────────────────────

def run_full_audit(logs_dir: str = "logs", market_family: str | None = None) -> Dict[str, pd.DataFrame]:
    """Run all audit reports and write them to CSV.

    Always pass market_family to avoid mixing BTC and weather metrics.
    BTC path labels (tp_before_sl_60m, mfe_60m, mae_60m) are meaningless
    for weather probability-space contracts and vice versa.
    """
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

    reports: Dict[str, pd.DataFrame] = {}

    blocker = blocker_replay_report(logs_dir, market_family=market_family)
    if not blocker.empty:
        blocker.to_csv(logs_path / "blocker_replay_report.csv", index=False)
        reports["blocker_replay"] = blocker
        logger.info("Blocker replay: %d reasons analysed", len(blocker))

    ablation = feature_family_ablation(logs_dir, market_family=market_family)
    if not ablation.empty:
        ablation.to_csv(logs_path / "feature_family_ablation.csv", index=False)
        reports["feature_family_ablation"] = ablation
        logger.info("Feature ablation: %d families tested", len(ablation))

    regime = regime_performance_report(logs_dir, market_family=market_family)
    if not regime.empty:
        regime.to_csv(logs_path / "regime_performance_report.csv", index=False)
        reports["regime_performance"] = regime
        logger.info("Regime performance: %d regimes analysed", len(regime))

    return reports
