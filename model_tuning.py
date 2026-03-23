from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import optuna
except Exception:
    optuna = None


class Stage2TemporalTuner:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.dataset_file = self.logs_dir / "sequence_dataset.csv"

    def _safe_read(self):
        if not self.dataset_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.dataset_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def tune_classifier(self, n_trials=20):
        if optuna is None:
            raise ImportError("Optuna is required for model_tuning.py. Install optuna to run hyperparameter tuning.")

        df = self._safe_read()
        if df.empty or "tp_before_sl_60m" not in df.columns:
            return None
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")
        feature_cols = [c for c in df.columns if "_lag_" in c or c in ["recent_wallet_activity_5", "recent_yes_ratio_5"]]
        if not feature_cols:
            return None

        X = df[feature_cols]
        y = df["tp_before_sl_60m"].fillna(0).astype(int)
        tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(df) // 50)))

        def objective(trial):
            hidden1 = trial.suggest_int("hidden1", 32, 256, step=32)
            hidden2 = trial.suggest_int("hidden2", 16, 128, step=16)
            alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
            lr = trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True)
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(
                    hidden_layer_sizes=(hidden1, hidden2),
                    random_state=42,
                    max_iter=300,
                    learning_rate_init=lr,
                    alpha=alpha,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=15,
                )),
            ])
            scores = cross_val_score(pipe, X, y, cv=tscv, scoring="accuracy")
            return float(scores.mean())

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_trial.params

