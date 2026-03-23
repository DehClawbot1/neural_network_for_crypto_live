from pathlib import Path

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


class FeatureAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class AutoencoderFeatureBuilder:
    def __init__(self, logs_dir="logs", weights_dir="weights"):
        self.logs_dir = Path(logs_dir)
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.input_file = self.logs_dir / "historical_dataset.csv"
        self.output_file = self.logs_dir / "autoencoder_latent_features.csv"
        self.model_file = self.weights_dir / "feature_autoencoder.pt"

    def _safe_read(self):
        if not self.input_file.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.input_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def build(self, latent_dim=8, epochs=10, batch_size=128, learning_rate=1e-3):
        if torch is None:
            raise ImportError("PyTorch is required for autoencoder_features.py. Install torch to use the autoencoder path.")

        df = self._safe_read()
        if df.empty:
            return pd.DataFrame()

        feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ["tp_before_sl_60m", "forward_return_15m"]]
        if not feature_cols:
            return pd.DataFrame()

        X = df[feature_cols].fillna(0.0).astype(float)
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True)

        model = FeatureAutoencoder(input_dim=X.shape[1], latent_dim=latent_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(epochs):
            for (xb,) in loader:
                optimizer.zero_grad()
                recon, _ = model(xb)
                loss = criterion(recon, xb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            _, latent = model(X_tensor)
        latent_df = pd.DataFrame(latent.numpy(), columns=[f"latent_{i}" for i in range(latent_dim)])
        if "timestamp" in df.columns:
            latent_df.insert(0, "timestamp", df["timestamp"].values)
        latent_df.to_csv(self.output_file, index=False)
        torch.save(model.state_dict(), self.model_file)
        return latent_df

