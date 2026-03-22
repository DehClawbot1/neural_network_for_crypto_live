from pathlib import Path
import ast

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class LiveReplayBuffer:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.logs_dir / "live_experience.csv"

    def load(self):
        if not self.path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def size(self):
        df = self.load()
        return 0 if df.empty else len(df)

    def recent(self, limit=5000):
        df = self.load()
        return df.tail(limit).reset_index(drop=True) if not df.empty else df


class LiveReplayDatasetEnv(gym.Env):
    """
    Replay environment backed by logged live experience.
    Safe nearline fine-tuning scaffold: it replays observations/rewards from
    `live_experience.csv` without turning the live trading loop itself into a
    blocking training process.
    """

    def __init__(self, experience_df):
        super().__init__()
        self.df = experience_df.reset_index(drop=True)
        self.idx = 0
        self.feature_dim = self._infer_feature_dim()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self.feature_dim,), dtype=np.float32)

    def _infer_feature_dim(self):
        if self.df.empty or "obs_before" not in self.df.columns:
            return 12
        try:
            arr = ast.literal_eval(str(self.df.iloc[0]["obs_before"]))
            return len(arr)
        except Exception:
            return 12

    def _parse_obs(self, value):
        try:
            arr = ast.literal_eval(str(value))
            arr = np.asarray(arr, dtype=np.float32)
            if arr.shape[0] == self.feature_dim:
                return arr
        except Exception:
            pass
        return np.zeros(self.feature_dim, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        if self.df.empty:
            return np.zeros(self.feature_dim, dtype=np.float32), {}
        row = self.df.iloc[self.idx]
        return self._parse_obs(row.get("obs_before")), {}

    def step(self, action):
        if self.df.empty:
            obs = np.zeros(self.feature_dim, dtype=np.float32)
            return obs, 0.0, True, False, {}
        row = self.df.iloc[self.idx]
        reward = float(row.get("reward", 0.0) or 0.0)
        obs_after = self._parse_obs(row.get("obs_after"))
        self.idx += 1
        terminated = self.idx >= len(self.df)
        truncated = False
        info = {
            "logged_action": row.get("action"),
            "token_id": row.get("token_id"),
            "order_status": row.get("order_status"),
        }
        return obs_after, reward, terminated, truncated, info
