import os
import importlib.util
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from polytrade_env import PolyTradeEnv

# Ensure directories exist for weights and TensorBoard logs
os.makedirs("weights", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def train_model(timesteps=10000):
    """
    Initializes the environment, builds the PPO model, and trains it.
    """
    print(f"[+] Starting RL training phase for {timesteps} timesteps...")

    tensorboard_available = importlib.util.find_spec("tensorboard") is not None
    progress_bar_available = (
        importlib.util.find_spec("tqdm") is not None and importlib.util.find_spec("rich") is not None
    )

    if not tensorboard_available:
        print("[!] tensorboard is not installed. Continuing without TensorBoard logging.")
    if not progress_bar_available:
        print("[!] tqdm/rich not fully installed. Continuing without progress bar.")

    # Vectorize the environment (SB3 requirement for efficient training)
    env = make_vec_env(lambda: PolyTradeEnv(), n_envs=1)

    # Initialize PPO agent
    # MlpPolicy is standard for flat array observations
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        tensorboard_log="./logs/" if tensorboard_available else None,
    )

    # Train the agent
    model.learn(total_timesteps=timesteps, progress_bar=progress_bar_available)

    # Save the model weights locally
    save_path = "weights/ppo_polytrader"
    model.save(save_path)
    print(f"[+] Model saved successfully to {save_path}.zip")

    return model, env


def test_inference(model, env, episodes=5):
    """
    Runs the trained model to demonstrate decision-making.
    """
    print(f"\n[+] Running inference tests for {episodes} episodes...")
    for episode in range(episodes):
        obs = env.reset()

        # Predict the best action based on the observation
        # deterministic=True ensures it picks the highest probability action
        action, _states = model.predict(obs, deterministic=True)

        # Step the environment
        obs, reward, done, info = env.step(action)

        # make_vec_env batches outputs, so we index [0] to get the single environment result
        print(f"Episode {episode + 1} | Action Taken: {action[0]} | Simulated Reward: {reward[0]:.2f}")


if __name__ == "__main__":
    # 1. Execute the training pipeline (5,000 steps is enough for a quick test)
    trained_model, eval_env = train_model(timesteps=5000)

    # 2. Run a quick sanity check to ensure the model outputs valid actions
    test_inference(trained_model, eval_env)
