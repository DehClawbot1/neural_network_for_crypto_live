from pathlib import Path
import re

import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    _BaseModule = object
else:
    _BaseModule = nn.Module


class TimeSeriesTransformer(_BaseModule):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, output_dim=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        x = self.input_proj(x)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


class Stage2TransformerModels:
    """
    Optional transformer scaffold for temporal Stage 2 modeling.
    Converts lagged columns into explicit [samples, time_steps, features] tensors.
    """

    def __init__(self, logs_dir="logs", weights_dir="weights"):
        self.logs_dir = Path(logs_dir)
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_file = self.logs_dir / "sequence_dataset.csv"
        self.model_file = self.weights_dir / "stage2_transformer.pt"

    def _safe_read(self):
        if not self.dataset_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.dataset_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

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
        import numpy as np
        step_arrays = []
        for lag in ordered_lags:
            cols = [col for _, col in sorted(grouped[lag])]
            step_arrays.append(df[cols].fillna(0.0).astype(float).values)
        return np.stack(step_arrays, axis=1)

    def train_classifier(self, epochs=5, batch_size=64, learning_rate=1e-3):
        if torch is None:
            raise ImportError("PyTorch is required for stage2_transformer_models.py. Install torch to use the transformer path.")

        df = self._safe_read()
        if df.empty or "tp_before_sl_60m" not in df.columns:
            return None
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")
        feature_cols = [c for c in df.columns if "_lag_" in c]
        X = self._reshape_sequence(df, feature_cols)
        if X is None:
            return None
        y = df["tp_before_sl_60m"].fillna(0).astype(float).values

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=False)

        model = TimeSeriesTransformer(input_dim=X.shape[2])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), self.model_file)
        return self.model_file

