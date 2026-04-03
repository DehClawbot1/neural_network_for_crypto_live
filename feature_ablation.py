from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline

from model_feature_catalog import DEFAULT_TABULAR_FEATURE_COLUMNS, TRAINING_FEATURE_FAMILIES
from model_feature_safety import drop_all_nan_features


class FeatureAblationReporter:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.dataset_file = self.logs_dir / "contract_targets.csv"
        self.output_file = self.logs_dir / "feature_ablation_report.csv"

    def _safe_read(self, path: Path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def _build_classifier(self):
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
            ]
        )

    def _build_regressor(self):
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("reg", RandomForestRegressor(n_estimators=200, random_state=42)),
            ]
        )

    def _evaluate_feature_set(self, train_df: pd.DataFrame, test_df: pd.DataFrame, candidates: list[str]) -> dict | None:
        usable, dropped_all_nan = drop_all_nan_features(train_df, candidates, context="feature_ablation")
        if not usable:
            return None

        result = {
            "usable_feature_count": int(len(usable)),
            "usable_features_json": str(usable),
            "dropped_all_nan_json": str(dropped_all_nan),
        }

        if "target_up" in train_df.columns and "target_up" in test_df.columns:
            clf = self._build_classifier()
            clf.fit(train_df[usable], train_df["target_up"].fillna(0).astype(int))
            preds = clf.predict(test_df[usable])
            result["accuracy"] = float(accuracy_score(test_df["target_up"].fillna(0).astype(int), preds))

        if "forward_return_15m" in train_df.columns and "forward_return_15m" in test_df.columns:
            reg = self._build_regressor()
            reg.fit(train_df[usable], pd.to_numeric(train_df["forward_return_15m"], errors="coerce").fillna(0.0))
            pred = reg.predict(test_df[usable])
            actual = pd.to_numeric(test_df["forward_return_15m"], errors="coerce").fillna(0.0)
            result["return_rmse"] = float(mean_squared_error(actual, pred) ** 0.5)

        return result

    def write(self):
        df = self._safe_read(self.dataset_file)
        if df.empty:
            return pd.DataFrame()

        if "timestamp" in df.columns:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.sort_values("timestamp")

        candidates = [col for col in DEFAULT_TABULAR_FEATURE_COLUMNS if col in df.columns]
        if not candidates or len(df) < 50:
            return pd.DataFrame()

        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        if train_df.empty or test_df.empty:
            return pd.DataFrame()

        baseline = self._evaluate_feature_set(train_df, test_df, candidates)
        if baseline is None:
            return pd.DataFrame()

        generated_at = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "generated_at": generated_at,
                "scope": "baseline",
                "scope_value": "all_features",
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "usable_feature_count": baseline.get("usable_feature_count"),
                "accuracy": baseline.get("accuracy"),
                "return_rmse": baseline.get("return_rmse"),
                "delta_accuracy_vs_baseline": 0.0,
                "delta_rmse_vs_baseline": 0.0,
                "family_features_present": int(len(candidates)),
                "family_features_used": int(baseline.get("usable_feature_count", 0)),
                "usable_features_json": baseline.get("usable_features_json"),
                "dropped_all_nan_json": baseline.get("dropped_all_nan_json"),
            }
        ]

        baseline_accuracy = baseline.get("accuracy")
        baseline_rmse = baseline.get("return_rmse")

        for family_name, family_features in TRAINING_FEATURE_FAMILIES.items():
            present = [feature for feature in family_features if feature in candidates]
            ablated_candidates = [feature for feature in candidates if feature not in set(present)]
            metrics = self._evaluate_feature_set(train_df, test_df, ablated_candidates)
            if metrics is None:
                continue
            accuracy = metrics.get("accuracy")
            rmse = metrics.get("return_rmse")
            rows.append(
                {
                    "generated_at": generated_at,
                    "scope": "feature_family_drop",
                    "scope_value": family_name,
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "usable_feature_count": metrics.get("usable_feature_count"),
                    "accuracy": accuracy,
                    "return_rmse": rmse,
                    "delta_accuracy_vs_baseline": (
                        float(accuracy - baseline_accuracy)
                        if accuracy is not None and baseline_accuracy is not None
                        else None
                    ),
                    "delta_rmse_vs_baseline": (
                        float(rmse - baseline_rmse)
                        if rmse is not None and baseline_rmse is not None
                        else None
                    ),
                    "family_features_present": int(len(present)),
                    "family_features_used": int(len([feature for feature in present if feature in ablated_candidates])),
                    "usable_features_json": metrics.get("usable_features_json"),
                    "dropped_all_nan_json": metrics.get("dropped_all_nan_json"),
                }
            )

        out = pd.DataFrame(rows)
        out.to_csv(self.output_file, index=False)
        return out
