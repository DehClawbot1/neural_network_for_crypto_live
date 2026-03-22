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
    Reward is based on portfolio value transitions, not ad-hoc realized/unrealized bonuses.
    """

    FEATURE_DIM = 10

    def __init__(self, logs_dir="logs", max_hold_steps=60, fee_rate=0.001, slippage_penalty=0.002, risk_penalty=0.001):
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
        self.inventory_fraction = 0.0
        self.cash_usdc = 0.0
        self.committed_capital = 0.0
        self.shares = 0.0
        self.entry_price = 0.0
        self.position_age = 0
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

    def _position_value(self, price):
        return self.shares * float(price)

    def _portfolio_value(self, price):
        return float(self.cash_usdc) + self._position_value(price)

    def _build_state(self):
        row = self.episode_row or {}
        price = self._current_price()
        position_value = self._position_value(price)
        unrealized = position_value - (self.shares * self.entry_price if self.position_open else 0.0)
        spread = float(row.get("spread", 0.0) or 0.0)
        realized_vol = float(row.get("btc_realized_vol_15m", row.get("volatility_score", 0.0)) or 0.0)
        time_to_close = float(row.get("time_to_close_minutes", 0.0) or 0.0)
        liquidity = float(row.get("liquidity_score", 0.0) or 0.0)
        drawdown = min(0.0, unrealized)
        state = np.array(
            [
                float(price),
                float(self.entry_price if self.position_open else price),
                float(self.inventory_fraction),
                float(self.shares),
                float(row.get("confidence", 0.0) or 0.0),
                float(row.get("edge_score", 0.0) or 0.0),
                float(realized_vol),
                float(self.position_age / max(1, self.max_hold_steps)),
                float(drawdown),
                float(spread + liquidity + (time_to_close / max(1.0, time_to_close + 1.0))),
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
        self.position_side = str(self.episode_row.get("outcome_side", self.episode_row.get("side", "YES"))).upper()
        self.inventory_fraction = 0.0
        self.cash_usdc = 0.0
        self.committed_capital = 0.0
        self.shares = 0.0
        self.entry_price = 0.0
        self.position_age = 0
        return self._build_state(), {}

    def step(self, action):
        current_price = self._current_price()
        valid_actions = [0, 1, 2] if not self.position_open else [3, 4, 5]
        fee_cost = 0.0
        slippage_cost = 0.0

        portfolio_before = self._portfolio_value(current_price)

        if action not in valid_actions:
            action = valid_actions[0]

        if action in [1, 2] and not self.position_open:
            self.committed_capital = 10.0 if action == 1 else 50.0
            self.cash_usdc = 0.0
            self.entry_price = current_price
            self.position_open = True
            self.inventory_fraction = 1.0
            self.shares = PNLEngine.shares_from_capital(self.committed_capital, self.entry_price)
            slippage_cost = self.slippage_penalty
            fee_cost = self.committed_capital * self.fee_rate
        elif action == 4 and self.position_open:
            exited_fraction = 0.5
            exited_shares = self.shares * exited_fraction
            exit_value = exited_shares * current_price
            self.cash_usdc += exit_value
            self.shares -= exited_shares
            self.inventory_fraction = 0.5
            fee_cost = exit_value * self.fee_rate
        elif action == 5 and self.position_open:
            exit_value = self.shares * current_price
            self.cash_usdc += exit_value
            fee_cost = exit_value * self.fee_rate
            self.shares = 0.0
            self.inventory_fraction = 0.0
            self.position_open = False

        next_idx = min(self.step_idx + 1, len(self.episode_prices) - 1)
        self.step_idx = next_idx
        self.position_age += 1 if self.position_open else 0
        next_price = float(self.episode_prices[next_idx])
        portfolio_after = self._portfolio_value(next_price)

        inventory_penalty = self.risk_penalty * abs(self.shares * next_price)
        reward = (portfolio_after - portfolio_before) - fee_cost - slippage_cost - inventory_penalty

        terminated = self.step_idx >= len(self.episode_prices) - 1
        truncated = False

        info = {
            "action_taken": int(action),
            "valid_actions": valid_actions,
            "current_price": current_price,
            "next_price": next_price,
            "portfolio_before": portfolio_before,
            "portfolio_after": portfolio_after,
            "fee_cost": fee_cost,
            "slippage_cost": slippage_cost,
            "inventory_penalty": inventory_penalty,
        }
        return self._build_state(), float(reward), terminated, truncated, info


class LivePolyTradeEnv(gym.Env):
    """
    Live-state environment scaffold for `live-test`.
    This does not perform online training by itself; it exposes a live observation
    built from execution, balance, and quote context so the runtime can evolve
    toward shadow-mode / live experience collection safely.
    """

    FEATURE_DIM = 12

    def __init__(self, execution_client, market_price_service, token_id=None, outcome_side="YES", max_hold_steps=120):
        super().__init__()
        self.execution_client = execution_client
        self.market_price_service = market_price_service
        self.current_token_id = str(token_id) if token_id else None
        self.outcome_side = str(outcome_side).upper()
        self.max_hold_steps = max_hold_steps

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self.FEATURE_DIM,), dtype=np.float32)

        self.position_open = False
        self.entry_price = 0.0
        self.shares = 0.0
        self.position_age = 0
        self.last_quote = {}
        self.state = np.zeros(self.FEATURE_DIM, dtype=np.float32)

    def set_market_context(self, token_id, outcome_side="YES", entry_price=None, shares=None, position_open=False, position_age=0):
        self.current_token_id = str(token_id) if token_id else None
        self.outcome_side = str(outcome_side).upper()
        self.entry_price = float(entry_price or 0.0)
        self.shares = float(shares or 0.0)
        self.position_open = bool(position_open)
        self.position_age = int(position_age or 0)

    def _safe_balance(self):
        try:
            payload = self.execution_client.get_balance_allowance()
            if isinstance(payload, dict):
                for key in ["balance", "available", "available_balance", "amount"]:
                    if key in payload and payload[key] is not None:
                        return float(payload[key])
            return 0.0
        except Exception:
            return 0.0

    def _safe_quote(self):
        if not self.current_token_id:
            return {}
        try:
            return self.market_price_service.get_quote(self.current_token_id) or {}
        except Exception:
            return {}

    def _build_state(self):
        quote = self._safe_quote()
        self.last_quote = quote
        best_bid = float(quote.get("best_bid") or 0.0)
        best_ask = float(quote.get("best_ask") or 0.0)
        midpoint = float(quote.get("midpoint") or quote.get("last_trade_price") or 0.0)
        spread = float(quote.get("spread") or max(0.0, best_ask - best_bid))
        live_balance = float(self._safe_balance())
        position_value = float(self.shares * midpoint)
        unrealized = float(position_value - (self.shares * self.entry_price if self.position_open else 0.0))
        side_flag = 1.0 if self.outcome_side == "YES" else -1.0

        state = np.array(
            [
                midpoint,
                best_bid,
                best_ask,
                spread,
                live_balance,
                float(self.entry_price if self.position_open else midpoint),
                float(self.shares),
                float(position_value),
                float(unrealized),
                float(self.position_age / max(1, self.max_hold_steps)),
                float(side_flag),
                float(1.0 if self.position_open else 0.0),
            ],
            dtype=np.float32,
        )
        self.state = state
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position_age = 0 if not self.position_open else self.position_age
        return self._build_state(), {"token_id": self.current_token_id, "outcome_side": self.outcome_side}

    def step(self, action):
        action = int(action)
        if self.position_open:
            self.position_age += 1
        obs = self._build_state()
        reward = 0.0
        terminated = False
        truncated = self.position_age >= self.max_hold_steps
        info = {
            "action_requested": action,
            "token_id": self.current_token_id,
            "outcome_side": self.outcome_side,
            "quote": self.last_quote,
            "position_open": self.position_open,
            "shares": self.shares,
            "entry_price": self.entry_price,
        }
        return obs, float(reward), terminated, truncated, info


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = PolyTradeEnv()
    check_env(env, warn=True)
    print("\n[+] PolyTradeEnv initialized and passed Gymnasium compliance checks.")
