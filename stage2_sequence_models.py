from pathlib import Path
import re

import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # optional dependency scaffold
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


class SequenceGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1, task="classification"):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )
        self.task = task

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)


class WeightedBCELoss(_BaseModule):
    def forward(self, logits, targets, sample_weights):
        base = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (base * sample_weights).mean()


class WeightedMSELoss(nn.Module):
    def forward(self, preds, targets, sample_weights):
        base = (preds - targets) ** 2
        return (base * sample_weights).mean()


class Stage2SequenceModels:
    """
    Optional PyTorch sequence-model scaffold for Stage 2.
    Uses lagged columns but reshapes them into [samples, time_steps, features_per_step]
    so temporal order is explicit instead of flattened into an MLP input.
    """

    def __init__(self, logs_dir="logs", weights_dir="weights"):
        self.logs_dir = Path(logs_dir)
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_file = self.logs_dir / "sequence_dataset.csv"
        self.classifier_file = self.weights_dir / "stage2_sequence_classifier.pt"
        self.regressor_file = self.weights_dir / "stage2_sequence_regressor.pt"

    def _safe_read(self):
        if not self.dataset_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.dataset_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def _lag_columns(self, df):
        return [c for c in df.columns if "_lag_" in c]

    def _reshape_sequence(self, df, feature_cols):
        lag_pattern = re.compile(r"(.+)_lag_(\d+)$")
        grouped = {}
        for col in feature_cols:
            match = lag_pattern.match(col)
            if not match:
                continue
            base, lag = match.group(1), int(match.group(2))
            grouped.setdefault(lag, []).append((base, col))
        ordered_lags = sorted(grouped.keys())
        if not ordered_lags:
            return None
        step_arrays = []
        for lag in ordered_lags:
            cols = [col for _, col in sorted(grouped[lag])]
            step_arrays.append(df[cols].fillna(0.0).astype(float).values)
        import numpy as np
        return np.stack(step_arrays, axis=1)

    def _sample_weights(self, df):
        spread = pd.to_numeric(df.get("spread", 0.0), errors="coerce").fillna(0.0)
        vol = pd.to_numeric(df.get("volatility_score", df.get("btc_realized_vol_15m", 0.0)), errors="coerce").fillna(0.0)
        weights = 1.0 + spread.abs() + vol.abs()
        return weights.astype(float).values

    def train(self, epochs=5, batch_size=64, learning_rate=1e-3):
        if torch is None:
            raise ImportError("PyTorch is required for Stage2SequenceModels. Install torch to use this sequence-model path.")

        df = self._safe_read()
        if df.empty:
            return pd.DataFrame()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")

        feature_cols = self._lag_columns(df)
        if not feature_cols:
            return pd.DataFrame()

        X = self._reshape_sequence(df, feature_cols)
        if X is None:
            return pd.DataFrame()

        sample_weights = self._sample_weights(df)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        w_train = sample_weights[:split_idx]
        results = {}

        if "tp_before_sl_60m" in df.columns:
            y = df["tp_before_sl_60m"].fillna(0).astype(int).values
            y_train, y_test = y[:split_idx], y[split_idx:]
            clf = SequenceGRU(input_size=X.shape[2], output_size=1, task="classification")
            self._fit_model(clf, X_train, y_train, w_train, epochs, batch_size, learning_rate, task="classification")
            torch.save(clf.state_dict(), self.classifier_file)
            results["sequence_classifier_trained"] = True

        if "forward_return_15m" in df.columns:
            y = df["forward_return_15m"].fillna(0.0).astype(float).values
            y_train, y_test = y[:split_idx], y[split_idx:]
            reg = SequenceGRU(input_size=X.shape[2], output_size=1, task="regression")
            self._fit_model(reg, X_train, y_train, w_train, epochs, batch_size, learning_rate, task="regression")
            torch.save(reg.state_dict(), self.regressor_file)
            results["sequence_regressor_trained"] = True

        return pd.DataFrame([results]) if results else pd.DataFrame()

    def _fit_model(self, model, X, y, sample_weights, epochs, batch_size, learning_rate, task="classification"):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        w_tensor = torch.tensor(sample_weights, dtype=torch.float32).view(-1, 1)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor, w_tensor), batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = WeightedBCELoss() if task == "classification" else WeightedMSELoss()
        model.train()
        for _ in range(epochs):
            for xb, yb, wb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb, wb)
                loss.backward()
                optimizer.step()

