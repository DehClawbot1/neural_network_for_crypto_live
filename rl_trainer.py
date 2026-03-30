import os
import time
import importlib.util
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from polytrade_env import PolyTradeEnv
from live_replay_buffer import LiveReplayBuffer, LiveReplayDatasetEnv

# Ensure directories exist for weights and TensorBoard logs
os.makedirs("weights", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def train_model(timesteps=10000):
    """
    Initializes the environment, wraps it with normalization, builds the PPO model, and trains it.
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

    def make_env():
        return PolyTradeEnv()

    venv = DummyVecEnv([make_env])
    env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        tensorboard_log="./logs/" if tensorboard_available else None,
    )

    model.learn(total_timesteps=timesteps, progress_bar=progress_bar_available)

    save_path = "weights/ppo_polytrader"
    model.save(save_path)
    env.save("weights/ppo_polytrader_vecnormalize.pkl")
    print(f"[+] Model saved successfully to {save_path}.zip")

    return model, env


def fine_tune_from_live_buffer(min_rows=100, batch_rows=1000, timesteps=256, sleep_seconds=60):
    """
    Safe nearline live fine-tuning scaffold.
    It does not block the trading loop: it periodically loads recent live
    experience, builds a replay env, and fine-tunes the latest PPO weights
    in short batches.
    """
    if os.getenv("ENABLE_OFFLINE_REPLAY_PPO", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        print("[!] Offline replay PPO fine-tuning is disabled by default; logged transitions are not a valid control env.")
        return None

    buffer = LiveReplayBuffer()
    model_path = "weights/ppo_polytrader"
    zipped_path = model_path + ".zip"
    if not os.path.exists(zipped_path):
        print(f"[!] No PPO weights found at {zipped_path}. Skipping live fine-tune.")
        return None

    while True:
        df = buffer.recent(limit=batch_rows)
        if df.empty or len(df) < min_rows:
            print(f"[!] Live replay buffer too small for fine-tuning: {len(df) if not df.empty else 0}/{min_rows}")
            time.sleep(sleep_seconds)
            continue

        replay_env = LiveReplayDatasetEnv(df)
        expected_dim = int(LiveReplayDatasetEnv(df).observation_space.shape[0]) # BUG FIX 2: Use dummy offline env to prevent API crash
        if int(replay_env.observation_space.shape[0]) != expected_dim:
            print(
                f"[!] Skipping live fine-tune: replay observation dim {replay_env.observation_space.shape[0]} "
                f"does not match PPO training env dim {expected_dim}."
            )
            time.sleep(sleep_seconds)
            continue

        env = make_vec_env(lambda: LiveReplayDatasetEnv(df), n_envs=1)
        try:
            model = PPO.load(model_path, env=env)
        print(f"[+] Fine-tuning PPO from live replay buffer ({len(df)} rows, {timesteps} timesteps)...")
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        model.save(model_path)
        print(f"[+] Updated PPO weights saved to {model_path}.zip")
        except Exception as e:
            print(f"[!] Fine-tuning interrupted (possible file lock race condition): {e}") # BUG FIX 1
        time.sleep(sleep_seconds)


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

