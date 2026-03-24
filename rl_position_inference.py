from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from polytrade_env import PolyTradeEnv


def prepare_position_observation(position_row):
    return np.array(
        [
            float(position_row.get("confidence", 0.5)),
            float(position_row.get("shares", 0.0)),
            float(position_row.get("current_price", 0.5)),
            float(position_row.get("entry_price", 0.5)),
            float(position_row.get("market_value", 0.0)),
            float(position_row.get("unrealized_pnl", 0.0)),
            0.5,
            0.5,
            0.5,
            0.5,
        ],
        dtype=np.float32,
    )


class PositionRLInference:
    def __init__(self, weights_dir="weights"):
        self.weights_dir = Path(weights_dir)
        self.model_path = self.weights_dir / "ppo_position_policy"
        self.vecnorm_path = self.weights_dir / "ppo_position_vecnormalize.pkl"
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

    def predict(self, position_row):
        if self.model is None and self.load() is None:
            return None
        obs = prepare_position_observation(position_row)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action.item() if hasattr(action, "item") else action[0] if hasattr(action, "__len__") else action)
