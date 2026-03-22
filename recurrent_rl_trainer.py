import os
import importlib.util

from gymnasium.wrappers import FlattenObservation

from polytrade_env import PolyTradeEnv

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None


def train_recurrent_model(timesteps=10000):
    """
    Optional recurrent PPO trainer using sb3-contrib.
    This is the correct path when the policy needs memory rather than a flat MLP snapshot.
    """
    if RecurrentPPO is None:
        raise ImportError("sb3-contrib is required for recurrent_rl_trainer.py. Install sb3-contrib to use RecurrentPPO.")

    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = FlattenObservation(PolyTradeEnv())
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        tensorboard_log="./logs/",
    )
    model.learn(total_timesteps=timesteps)
    save_path = "weights/recurrent_ppo_polytrader"
    model.save(save_path)
    print(f"[+] Recurrent PPO model saved to {save_path}.zip")
    return model, env


if __name__ == "__main__":
    train_recurrent_model(timesteps=5000)
