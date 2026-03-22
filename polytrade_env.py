from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from pnl_engine import PNLEngine


class PolyTradeEnv(gym.Env):
    """
    Replay-based Gymnasium environment for paper-trading research.
    One episode = one trade lifecycle replayed over real token price history.
    """

    FEATURE_DIM = 9

    def __init__(self, logs_dir="logs", max_hold_steps=60, fee_rate=0.0, slippage_penalty=0.002, risk_penalty=0.001):
        super().__init__()
        self.logs_dir = Path(logs_dir)
        self.contract_targets_file = self.logs_dir / "contract_targets.csv"
        self.history_file = self.logs_dir / "clob_price_history.csv"
        self.max_hold_steps = max_hold_steps
        self.fee_rate = fee_rate
        self.slippage_penalty = slippage_penalty
        self.risk_penalty = risk_penalty

        # 0 = stay flat, 1 = enter small, 2 = enter large, 3 = hold, 4 = reduce, 5 = exit
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self.FEATURE_DIM,), dtype=np.float32)

        self.dataset = self._load_dataset()
        self.episode_prices = []
        self.episode_row = None
        self.step_idx = 0
        self.position_open = False
        self.position_side = "YES"
        self.capital_usdc = 0.0
        self.shares = 0.0
        self.entry_price = 0.0
        self.prev_unrealized = 0.0
        self.state = np.zeros(self.FEATURE_DIM, dtype=np.float32)

    def _safe_read(self, path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    def _load_dataset(self):
        targets = self._safe_read(self.contract_targets_file)
        history = self._safe_read(self.history_file)
        if targets.empty or history.empty or "token_id" not in targets.columns:
            return []

        targets["timestamp"] = pd.to_datetime(targets["timestamp"], utc=True, errors="coerce")
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
        history = history.sort_values(["token_id", "timestamp"]).reset_index(drop=True)

        dataset = []
        for _, row in targets.iterrows():
            token_id = row.get("token_id")
            ts = row.get("timestamp")
            if pd.isna(ts) or pd.isna(token_id):
                continue
            token_history = history[(history["token_id"].astype(str) == str(token_id)) & (history["timestamp"] >= ts)].head(self.max_hold_steps + 1)
            if len(token_history) < 2:
                continue
            dataset.append({"signal": row.to_dict(), "prices": token_history["price"].astype(float).tolist()})
        return dataset

    def _current_price(self):
        if not self.episode_prices:
            return 0.5
        return float(self.episode_prices[min(self.step_idx, len(self.episode_prices) - 1)])

    def _unrealized_pnl(self, price):
        if not self.position_open:
            return 0.0
        return PNLEngine.mark_to_market_pnl(self.capital_usdc, self.entry_price, price)

    def _build_state(self):
        row = self.episode_row or {}
        price = self._current_price()
        unrealized = self._unrealized_pnl(price)
        time_in_trade = self.step_idx / max(1, self.max_hold_steps)
        state = np.array(
            [
                float(price),
                float(self.entry_price if self.position_open else price),
                1.0 if self.position_side == "YES" else 0.0,
                float(self.shares),
                float(row.get("confidence", 0.0) or 0.0),
                float(row.get("edge_score", 0.0) or 0.0),
                float(row.get("volatility_score", 0.0) or 0.0),
                float(time_in_trade),
                float(unrealized),
            ],
            dtype=np.float32,
        )
        self.state = state
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self.dataset:
            self.episode_row = {}
            self.episode_prices = [0.5, 0.5]
        else:
            idx = int(self.np_random.integers(0, len(self.dataset)))
            episode = self.dataset[idx]
            self.episode_row = episode["signal"]
            self.episode_prices = episode["prices"]

        self.step_idx = 0
        self.position_open = False
        self.position_side = str(self.episode_row.get("side", self.episode_row.get("outcome_side", "YES"))).upper()
        self.capital_usdc = 0.0
        self.shares = 0.0
        self.entry_price = 0.0
        self.prev_unrealized = 0.0
        return self._build_state(), {}

    def step(self, action):
        current_price = self._current_price()
        reward = 0.0
        realized_component = 0.0

        if action in [1, 2] and not self.position_open:
            self.capital_usdc = 10.0 if action == 1 else 50.0
            self.entry_price = current_price
            self.position_open = True
            self.shares = PNLEngine.shares_from_capital(self.capital_usdc, self.entry_price)
            reward -= self.slippage_penalty
        elif action == 4 and self.position_open:
            self.capital_usdc *= 0.5
            self.shares *= 0.5
            reward -= self.fee_rate
        elif action == 5 and self.position_open:
            realized_component = self._unrealized_pnl(current_price)
            reward += realized_component - self.fee_rate
            self.position_open = False
            self.capital_usdc = 0.0
            self.shares = 0.0

        next_idx = min(self.step_idx + 1, len(self.episode_prices) - 1)
        next_price = float(self.episode_prices[next_idx])
        new_unrealized = self._unrealized_pnl(next_price)
        unrealized_delta = new_unrealized - self.prev_unrealized
        reward += unrealized_delta - self.risk_penalty * abs(new_unrealized)

        self.prev_unrealized = new_unrealized
        self.step_idx = next_idx
        terminated = self.step_idx >= len(self.episode_prices) - 1
        truncated = False

        if terminated and self.position_open:
            final_realized = self._unrealized_pnl(next_price)
            reward += final_realized - self.fee_rate
            self.position_open = False
            self.capital_usdc = 0.0
            self.shares = 0.0

        info = {
            "action_taken": int(action),
            "current_price": current_price,
            "next_price": next_price,
            "realized_component": realized_component,
            "unrealized_delta": unrealized_delta,
        }
        return self._build_state(), float(reward), terminated, truncated, info


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = PolyTradeEnv()
    check_env(env, warn=True)
    print("\n[+] PolyTradeEnv initialized and passed Gymnasium compliance checks.")
