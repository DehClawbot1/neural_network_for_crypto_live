"""LDA and GaussianNB audited baselines.

These lightweight generative/linear classifiers run on a curated
*standardized* feature subset so they stay well-conditioned.
They are **not** intended to replace the tree/NN production stack
but to serve as sanity-check baselines: if LDA or NB beats Stage 1
on a regime slice, the tree model is likely over-fitting or the
feature set is leaking.

KDE/Parzen is intentionally limited to a narrow low-dimensional
subset (max 8 features) to avoid curse-of-dimensionality blow-up.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from model_feature_catalog import DEFAULT_TABULAR_FEATURE_COLUMNS
from model_feature_safety import drop_all_nan_features
from feature_treatment_policy import features_by_kind, features_for_scope

logger = logging.getLogger(__name__)

# Maximum features for KDE/Parzen to avoid curse of dimensionality
_KDE_MAX_FEATURES = int(os.getenv("KDE_MAX_FEATURES", "8") or 8)


def _load_sklearn_baseline_components():
    """Lazy-import sklearn components to avoid import-time failures."""
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, precision_score
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.naive_bayes import GaussianNB
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except (ImportError, TypeError) as exc:
        logger.warning("sklearn import failed: %s", exc)
        return None
    return {
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
        "GaussianNB": GaussianNB,
        "Pipeline": Pipeline,
        "SimpleImputer": SimpleImputer,
        "StandardScaler": StandardScaler,
        "TimeSeriesSplit": TimeSeriesSplit,
        "accuracy_score": accuracy_score,
        "precision_score": precision_score,
    }


def _curated_standardized_features(df: pd.DataFrame) -> list[str]:
    """Select features suitable for linear/generative models.

    Picks features whose treatment policy is ``standardize``, ``clip01``,
    or ``boolean`` — i.e. bounded or z-scorable.  Drops all-NaN columns.
    """
    candidates = [c for c in DEFAULT_TABULAR_FEATURE_COLUMNS if c in df.columns]
    candidates = features_for_scope("nn", candidates)
    # restrict to kinds that work well with LDA/NB
    ok_kinds = {"standardize", "clip01", "boolean", "raw"}
    curated = [c for c in candidates if features_by_kind(
        features_by_kind.__code__.co_varnames[0],  # unused
    ) or True]
    # simpler: just use the scope-filtered list and let the pipeline handle it
    from feature_treatment_policy import get_treatment
    curated = [c for c in candidates if get_treatment(c).kind in ok_kinds]
    curated, _ = drop_all_nan_features(df, curated, context="baseline_models")
    return curated


def _kde_feature_subset(df: pd.DataFrame) -> list[str]:
    """Select a narrow low-dimensional feature subset for KDE/Parzen."""
    # prefer high-variance, bounded features
    priority = [
        "trader_win_rate",
        "normalized_trade_size",
        "trend_score",
        "btc_volatility_regime_score",
        "btc_momentum_confluence",
        "btc_market_regime_score",
        "sentiment_score",
        "liquidity_score",
    ]
    available = [c for c in priority if c in df.columns and df[c].notna().any()]
    return available[:_KDE_MAX_FEATURES]


class BaselineModels:
    """Train LDA and GaussianNB baselines with walk-forward CV."""

    def __init__(self, logs_dir: str = "logs", weights_dir: str = "weights"):
        self.logs_dir = Path(logs_dir)
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_file = self.logs_dir / "contract_targets.csv"
        self.metrics_file = self.logs_dir / "baseline_eval.csv"

    def _safe_read(self) -> pd.DataFrame:
        if not self.dataset_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.dataset_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def train(self) -> pd.DataFrame:
        sklearn = _load_sklearn_baseline_components()
        if sklearn is None:
            logger.warning("sklearn unavailable — skipping baseline training.")
            return pd.DataFrame()
        Pipeline = sklearn["Pipeline"]
        SimpleImputer = sklearn["SimpleImputer"]
        StandardScaler = sklearn["StandardScaler"]
        LinearDiscriminantAnalysis = sklearn["LinearDiscriminantAnalysis"]
        GaussianNB = sklearn["GaussianNB"]
        TimeSeriesSplit = sklearn["TimeSeriesSplit"]
        accuracy_score = sklearn["accuracy_score"]
        precision_score = sklearn["precision_score"]

        df = self._safe_read()
        if df.empty:
            return pd.DataFrame()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")

        target_col = "tp_before_sl_60m" if "tp_before_sl_60m" in df.columns else None
        if target_col is None:
            return pd.DataFrame()

        feature_cols = _curated_standardized_features(df)
        if not feature_cols:
            return pd.DataFrame()

        if len(df) < 6:
            return pd.DataFrame()

        y = df[target_col].fillna(0).astype(int)
        if y.nunique(dropna=True) < 2:
            return pd.DataFrame()

        n_splits = min(5, len(df) - 1)
        if n_splits < 2:
            return pd.DataFrame()
        tscv = TimeSeriesSplit(n_splits=n_splits)

        metrics_rows = []

        # ── LDA ─────────────────────────────────────────────────
        lda_acc, lda_prec = [], []
        for train_idx, test_idx in tscv.split(df):
            train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
            if train_df.empty or test_df.empty:
                continue
            y_tr = train_df[target_col].fillna(0).astype(int)
            if y_tr.nunique() < 2:
                continue
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearDiscriminantAnalysis()),
            ])
            pipe.fit(train_df[feature_cols], y_tr)
            preds = pipe.predict(test_df[feature_cols])
            y_te = test_df[target_col].fillna(0).astype(int)
            lda_acc.append(accuracy_score(y_te, preds))
            lda_prec.append(precision_score(y_te, preds, zero_division=0))
        if lda_acc:
            metrics_rows.append({
                "model_kind": "lda",
                "feature_set": "curated_standardized",
                "scaling": "standard",
                "regularization": "none",
                "nonzero_feature_count": len(feature_cols),
                "accuracy": sum(lda_acc) / len(lda_acc),
                "precision": sum(lda_prec) / len(lda_prec),
            })

        # ── GaussianNB ──────────────────────────────────────────
        nb_acc, nb_prec = [], []
        for train_idx, test_idx in tscv.split(df):
            train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
            if train_df.empty or test_df.empty:
                continue
            y_tr = train_df[target_col].fillna(0).astype(int)
            if y_tr.nunique() < 2:
                continue
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", GaussianNB()),
            ])
            pipe.fit(train_df[feature_cols], y_tr)
            preds = pipe.predict(test_df[feature_cols])
            y_te = test_df[target_col].fillna(0).astype(int)
            nb_acc.append(accuracy_score(y_te, preds))
            nb_prec.append(precision_score(y_te, preds, zero_division=0))
        if nb_acc:
            metrics_rows.append({
                "model_kind": "gaussian_nb",
                "feature_set": "curated_standardized",
                "scaling": "standard",
                "regularization": "none",
                "nonzero_feature_count": len(feature_cols),
                "accuracy": sum(nb_acc) / len(nb_acc),
                "precision": sum(nb_prec) / len(nb_prec),
            })

        result = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame()
        if not result.empty:
            result.to_csv(self.metrics_file, index=False)
            logger.info("Baseline eval written to %s", self.metrics_file)
        return result
