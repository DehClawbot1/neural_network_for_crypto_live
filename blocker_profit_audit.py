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

def blocker_replay_report(logs_dir: str = "logs") -> pd.DataFrame:
    """For each reject_reason, compute hypothetical forward metrics.

    Joins candidate_decisions (rejected) with contract_targets (outcomes)
    on token_id to see what would have happened.
    """
    logs_path = Path(logs_dir)
    decisions = _safe_read(logs_path / "candidate_decisions.csv")
    targets = _safe_read(logs_path / "contract_targets.csv")

    if decisions.empty or targets.empty:
        logger.warning("Insufficient data for blocker replay.")
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

    target_cols = [c for c in ["token_id", "forward_return_15m", "tp_before_sl_60m", "mfe_60m", "mae_60m"] if c in targets.columns]
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

        def _col_mean(col_name):
            if col_name not in group.columns:
                return None
            s = pd.to_numeric(group[col_name], errors="coerce")
            return float(s.mean()) if s.notna().any() else None

        def _col_median(col_name):
            if col_name not in group.columns:
                return None
            s = pd.to_numeric(group[col_name], errors="coerce")
            return float(s.median()) if s.notna().any() else None

        fwd_mean = _col_mean("forward_return_15m")
        rows.append({
            "reject_reason": reason_str,
            "n_blocked": len(group),
            "mean_forward_return_15m": fwd_mean,
            "median_forward_return_15m": _col_median("forward_return_15m"),
            "tp_before_sl_rate": _col_mean("tp_before_sl_60m"),
            "mean_mfe_60m": _col_mean("mfe_60m"),
            "mean_mae_60m": _col_mean("mae_60m"),
            "replay_ev": (fwd_mean * len(group)) if fwd_mean is not None else None,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("replay_ev", ascending=False, na_position="last")
    return result


# ── feature-family ablation ─────────────────────────────────────────

def feature_family_ablation(logs_dir: str = "logs") -> pd.DataFrame:
    """Report model performance when trained on each feature family alone.

    Reads pre-computed ``contract_targets.csv`` and trains a quick
    logistic model per family to measure isolated predictive power.
    """
    from model_feature_catalog import TRAINING_FEATURE_FAMILIES
    from model_feature_safety import drop_all_nan_features

    logs_path = Path(logs_dir)
    df = _safe_read(logs_path / "contract_targets.csv")
    if df.empty or "tp_before_sl_60m" not in df.columns:
        return pd.DataFrame()

    y = df["tp_before_sl_60m"].fillna(0).astype(int)
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
            y_tr = train_df["tp_before_sl_60m"].fillna(0).astype(int)
            if y_tr.nunique() < 2:
                continue
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ])
            pipe.fit(train_df[available], y_tr)
            preds = pipe.predict(test_df[available])
            y_te = test_df["tp_before_sl_60m"].fillna(0).astype(int)
            accs.append(accuracy_score(y_te, preds))

        if accs:
            rows.append({
                "family": family_name,
                "n_features": len(available),
                "accuracy": sum(accs) / len(accs),
            })

    return pd.DataFrame(rows)


# ── regime performance ──────────────────────────────────────────────

def regime_performance_report(logs_dir: str = "logs") -> pd.DataFrame:
    """Performance by regime (calm/trend/volatile/chaotic) from closed trades."""
    logs_path = Path(logs_dir)
    closed = _safe_read(logs_path / "closed_positions.csv")
    if closed.empty:
        return pd.DataFrame()

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

def run_full_audit(logs_dir: str = "logs") -> Dict[str, pd.DataFrame]:
    """Run all audit reports and write them to CSV."""
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

    reports: Dict[str, pd.DataFrame] = {}

    blocker = blocker_replay_report(logs_dir)
    if not blocker.empty:
        blocker.to_csv(logs_path / "blocker_replay_report.csv", index=False)
        reports["blocker_replay"] = blocker
        logger.info("Blocker replay: %d reasons analysed", len(blocker))

    ablation = feature_family_ablation(logs_dir)
    if not ablation.empty:
        ablation.to_csv(logs_path / "feature_family_ablation.csv", index=False)
        reports["feature_family_ablation"] = ablation
        logger.info("Feature ablation: %d families tested", len(ablation))

    regime = regime_performance_report(logs_dir)
    if not regime.empty:
        regime.to_csv(logs_path / "regime_performance_report.csv", index=False)
        reports["regime_performance"] = regime
        logger.info("Regime performance: %d regimes analysed", len(regime))

    return reports
