from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from polytrade_env import PolyTradeEnv


def prepare_entry_observation(feature_row):
    return np.array(
        [
            float(feature_row.get("trader_win_rate", 0.5)),
            float(feature_row.get("normalized_trade_size", 0.5)),
            float(feature_row.get("current_price", 0.5)),
            float(feature_row.get("time_left", 0.5)),
            float(feature_row.get("liquidity_score", 0.5)),
            float(feature_row.get("volume_score", 0.5)),
            float(feature_row.get("probability_momentum", 0.5)),
            float(feature_row.get("volatility_score", 0.5)),
            float(feature_row.get("whale_pressure", 0.5)),
            float(feature_row.get("market_structure_score", 0.5)),
        ],
        dtype=np.float32,
    )


class EntryRLInference:
    def __init__(self, weights_dir="weights"):
        self.weights_dir = Path(weights_dir)
        self.model_path = self.weights_dir / "ppo_entry_policy"
        self.vecnorm_path = self.weights_dir / "ppo_entry_vecnormalize.pkl"
        self.model = None
        self.env = None

    def load(self):
        if not self.model_path.with_suffix(".zip").exists():
            return None
        env = DummyVecEnv([lambda: PolyTradeEnv()])
        if self.vecnorm_path.exists():
            env = VecNormalize.load(str(self.vecnorm_path), env)
            env.training = False
            env.norm_reward = False
        self.env = env
        self.model = PPO.load(str(self.model_path), env=env)
        return self.model

    def predict(self, feature_row):
        if self.model is None and self.load() is None:
            return None
        obs = prepare_entry_observation(feature_row)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action.item() if hasattr(action, "item") else action[0] if hasattr(action, "__len__") else action)
