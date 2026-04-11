"""LDA, GaussianNB, and KDE/Parzen audited baselines."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from feature_treatment_policy import features_for_scope, get_treatment
from model_feature_catalog import DEFAULT_TABULAR_FEATURE_COLUMNS
from model_feature_safety import drop_all_nan_features


logger = logging.getLogger(__name__)

_KDE_MAX_FEATURES = int(os.getenv("KDE_MAX_FEATURES", "8") or 8)


def _load_sklearn_baseline_components():
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, precision_score
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KernelDensity
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except (ImportError, TypeError) as exc:
        logger.warning("sklearn import failed: %s", exc)
        return None
    return {
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
        "GaussianNB": GaussianNB,
        "KernelDensity": KernelDensity,
        "Pipeline": Pipeline,
        "SimpleImputer": SimpleImputer,
        "StandardScaler": StandardScaler,
        "TimeSeriesSplit": TimeSeriesSplit,
        "accuracy_score": accuracy_score,
        "precision_score": precision_score,
    }


def _curated_standardized_features(df: pd.DataFrame) -> list[str]:
    candidates = [c for c in DEFAULT_TABULAR_FEATURE_COLUMNS if c in df.columns]
    candidates = features_for_scope("nn", candidates)
    ok_kinds = {"standardize", "clip01", "boolean", "raw"}
    curated = [c for c in candidates if get_treatment(c).kind in ok_kinds]
    curated, _ = drop_all_nan_features(df, curated, context="baseline_models")
    return curated


def _kde_feature_subset(df: pd.DataFrame) -> list[str]:
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


def _resolve_regime_column(df: pd.DataFrame) -> str | None:
    for candidate in ("btc_market_regime_label", "technical_regime_bucket", "btc_volatility_regime"):
        if candidate in df.columns:
            return candidate
    return None


def _normalize_regime_value(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _precision(precision_score, y_true, preds) -> float | None:
    try:
        return float(precision_score(y_true, preds, zero_division=0))
    except Exception:
        return None


def _replay_metrics(df: pd.DataFrame, preds: pd.Series) -> tuple[float | None, float | None]:
    if "forward_return_15m" not in df.columns:
        return None, None
    selected = pd.to_numeric(df.loc[preds == 1, "forward_return_15m"], errors="coerce").dropna()
    if selected.empty:
        return None, None
    gross_profit = float(selected[selected > 0].sum())
    gross_loss = abs(float(selected[selected < 0].sum()))
    profit_factor = gross_profit if gross_loss <= 0 else gross_profit / gross_loss
    return float(selected.mean()), profit_factor


class BaselineModels:
    """Train audited tabular baselines with walk-forward CV and regime slices."""

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

    def _append_metrics(
        self,
        *,
        metrics_rows: list[dict],
        model_kind: str,
        feature_set: str,
        scaling: str,
        feature_count: int,
        prediction_frame: pd.DataFrame,
        regime_col: str | None,
        accuracy_score,
        precision_score,
        n_train_rows: int,
    ) -> None:
        scopes = [("all", prediction_frame)]
        if regime_col and regime_col in prediction_frame.columns:
            for regime_value, regime_df in prediction_frame.groupby(regime_col, dropna=False):
                if len(regime_df.index) < 10:
                    continue
                regime_name = _normalize_regime_value(regime_value)
                if not regime_name:
                    continue
                scopes.append((regime_name, regime_df))
        for regime_name, scope_df in scopes:
            y_true = scope_df["target"].fillna(0).astype(int)
            preds = scope_df["pred"].fillna(0).astype(int)
            replay_ev, profit_factor = _replay_metrics(scope_df, preds)
            metrics_rows.append(
                {
                    "model_kind": model_kind,
                    "artifact_group": model_kind,
                    "feature_set": feature_set,
                    "scaling": scaling,
                    "regularization": "none",
                    "market_family": "all",
                    "regime_slice": regime_name,
                    "nonzero_feature_count": feature_count,
                    "n_train_rows": int(n_train_rows),
                    "n_test_rows": int(len(scope_df.index)),
                    "accuracy": float(accuracy_score(y_true, preds)),
                    "precision": _precision(precision_score, y_true, preds),
                    "profit_factor": profit_factor,
                    "replay_ev": replay_ev,
                }
            )

    def train(self) -> pd.DataFrame:
        sklearn = _load_sklearn_baseline_components()
        if sklearn is None:
            logger.warning("sklearn unavailable; skipping baseline training.")
            return pd.DataFrame()
        Pipeline = sklearn["Pipeline"]
        SimpleImputer = sklearn["SimpleImputer"]
        StandardScaler = sklearn["StandardScaler"]
        LinearDiscriminantAnalysis = sklearn["LinearDiscriminantAnalysis"]
        GaussianNB = sklearn["GaussianNB"]
        KernelDensity = sklearn["KernelDensity"]
        TimeSeriesSplit = sklearn["TimeSeriesSplit"]
        accuracy_score = sklearn["accuracy_score"]
        precision_score = sklearn["precision_score"]

        df = self._safe_read()
        if df.empty or "tp_before_sl_60m" not in df.columns:
            return pd.DataFrame()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp", kind="stable")

        y = df["tp_before_sl_60m"].fillna(0).astype(int)
        if y.nunique() < 2 or len(df.index) < 10:
            return pd.DataFrame()

        curated_features = _curated_standardized_features(df)
        if not curated_features:
            return pd.DataFrame()
        kde_features = _kde_feature_subset(df)
        regime_col = _resolve_regime_column(df)

        n_splits = min(5, len(df.index) - 1)
        if n_splits < 2:
            return pd.DataFrame()
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics_rows: list[dict] = []
        lda_predictions = []
        nb_predictions = []
        kde_predictions = []
        last_train_rows = 0

        for train_idx, test_idx in tscv.split(df):
            train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
            if train_df.empty or test_df.empty:
                continue
            y_train = train_df["tp_before_sl_60m"].fillna(0).astype(int)
            if y_train.nunique() < 2:
                continue
            last_train_rows = len(train_df.index)

            lda_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LinearDiscriminantAnalysis()),
                ]
            )
            lda_pipe.fit(train_df[curated_features], y_train)
            lda_fold = test_df.copy()
            lda_fold["pred"] = lda_pipe.predict(test_df[curated_features])
            lda_fold["target"] = test_df["tp_before_sl_60m"].fillna(0).astype(int)
            lda_predictions.append(lda_fold)

            nb_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", GaussianNB()),
                ]
            )
            nb_pipe.fit(train_df[curated_features], y_train)
            nb_fold = test_df.copy()
            nb_fold["pred"] = nb_pipe.predict(test_df[curated_features])
            nb_fold["target"] = test_df["tp_before_sl_60m"].fillna(0).astype(int)
            nb_predictions.append(nb_fold)

            if kde_features:
                imputer = SimpleImputer(strategy="median")
                scaler = StandardScaler()
                X_train = scaler.fit_transform(imputer.fit_transform(train_df[kde_features]))
                X_test = scaler.transform(imputer.transform(test_df[kde_features]))
                priors = y_train.value_counts(normalize=True).to_dict()
                models = {}
                for cls in sorted(y_train.unique()):
                    mask = y_train == cls
                    if int(mask.sum()) < 2:
                        continue
                    kde = KernelDensity(kernel="gaussian", bandwidth=0.75)
                    kde.fit(X_train[mask.values])
                    models[int(cls)] = kde
                if len(models) >= 2:
                    score_0 = models.get(0).score_samples(X_test) + np.log(max(float(priors.get(0, 0.5)), 1e-9))
                    score_1 = models.get(1).score_samples(X_test) + np.log(max(float(priors.get(1, 0.5)), 1e-9))
                    kde_fold = test_df.copy()
                    kde_fold["pred"] = np.where(score_1 >= score_0, 1, 0)
                    kde_fold["target"] = test_df["tp_before_sl_60m"].fillna(0).astype(int)
                    kde_predictions.append(kde_fold)

        if lda_predictions:
            self._append_metrics(
                metrics_rows=metrics_rows,
                model_kind="lda",
                feature_set="curated_standardized",
                scaling="standard",
                feature_count=len(curated_features),
                prediction_frame=pd.concat(lda_predictions, ignore_index=False),
                regime_col=regime_col,
                accuracy_score=accuracy_score,
                precision_score=precision_score,
                n_train_rows=last_train_rows,
            )
        if nb_predictions:
            self._append_metrics(
                metrics_rows=metrics_rows,
                model_kind="gaussian_nb",
                feature_set="curated_standardized",
                scaling="standard",
                feature_count=len(curated_features),
                prediction_frame=pd.concat(nb_predictions, ignore_index=False),
                regime_col=regime_col,
                accuracy_score=accuracy_score,
                precision_score=precision_score,
                n_train_rows=last_train_rows,
            )
        if kde_predictions:
            self._append_metrics(
                metrics_rows=metrics_rows,
                model_kind="kde_parzen",
                feature_set="kde_curated_subset",
                scaling="standard",
                feature_count=len(kde_features),
                prediction_frame=pd.concat(kde_predictions, ignore_index=False),
                regime_col=regime_col,
                accuracy_score=accuracy_score,
                precision_score=precision_score,
                n_train_rows=last_train_rows,
            )

        result = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame()
        if not result.empty:
            result.to_csv(self.metrics_file, index=False)
            logger.info("Baseline eval written to %s", self.metrics_file)
        return result
