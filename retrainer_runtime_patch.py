from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd

from training_windowed_models import WindowedStage1Models, WindowedStage2TemporalModels

_BASE_DIR = Path(__file__).resolve().parent


def _load_legacy_module(module_name: str, filename: str):
    path = _BASE_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy_retrainer = _load_legacy_module("legacy_retrainer", "retrainer.py")
_legacy_rl_trainer = _load_legacy_module("legacy_rl_trainer", "rl_trainer.py")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "true" if default else "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resume_ppo_train_model(timesteps=5000):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from polytrade_env import PolyTradeEnv
    import importlib.util as _importlib_util
    import os as _os

    _os.makedirs("weights", exist_ok=True)
    _os.makedirs("logs", exist_ok=True)
    print(f"[+] Starting incremental RL training phase for {timesteps} timesteps...")

    tensorboard_available = _importlib_util.find_spec("tensorboard") is not None
    progress_bar_available = _importlib_util.find_spec("tqdm") is not None and _importlib_util.find_spec("rich") is not None

    def make_env():
        return PolyTradeEnv()

    venv = DummyVecEnv([make_env])
    env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    save_path = "weights/ppo_polytrader"
    zipped_path = save_path + ".zip"
    if _os.path.exists(zipped_path):
        print(f"[+] Resuming PPO from existing weights at {zipped_path}")
        model = PPO.load(save_path, env=env)
        model.learn(total_timesteps=timesteps, progress_bar=progress_bar_available, reset_num_timesteps=False)
    else:
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

    model.save(save_path)
    env.save("weights/ppo_polytrader_vecnormalize.pkl")
    print(f"[+] Model saved successfully to {save_path}.zip")
    return model, env


class IncrementalRetrainer(_legacy_retrainer.Retrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ── FIX: Changed from 25/50 to 5/10 so model retrains after every 5 closed trades
        self.min_new_closed_rows = _env_int("RETRAIN_MIN_NEW_CLOSED_ROWS", 5)
        self.min_new_replay_rows = _env_int("RETRAIN_MIN_NEW_REPLAY_ROWS", 10)
        self.cooldown_minutes = _env_int("RETRAIN_COOLDOWN_MINUTES", 60)
        self.rl_timesteps = _env_int("RETRAIN_RL_TIMESTEPS", 1000)
        self.enable_startup_retrain = _env_bool("ENABLE_STARTUP_RETRAIN", False)

    def _read_status_state(self):
        if not self.status_csv.exists():
            return {}
        try:
            df = pd.read_csv(self.status_csv)
            if df.empty:
                return {}
            return df.iloc[-1].to_dict()
        except Exception:
            return {}

    def _status_payload(
        self,
        *,
        closed_rows: int,
        replay_rows: int,
        last_trained_closed_rows: int,
        last_trained_replay_rows: int,
        last_retrained_at,
        action: str,
    ):
        new_closed_rows = max(0, int(closed_rows) - int(last_trained_closed_rows))
        new_replay_rows = max(0, int(replay_rows) - int(last_trained_replay_rows))
        progress_closed = round(new_closed_rows / max(1, self.min_new_closed_rows), 4)
        progress_replay = round(new_replay_rows / max(1, self.min_new_replay_rows), 4)
        return {
            "closed_trade_rows": int(closed_rows),
            "replay_rows": int(replay_rows),
            "last_trained_closed_rows": int(last_trained_closed_rows),
            "last_trained_replay_rows": int(last_trained_replay_rows),
            "new_closed_rows": int(new_closed_rows),
            "new_replay_rows": int(new_replay_rows),
            "min_new_closed_rows": int(self.min_new_closed_rows),
            "min_new_replay_rows": int(self.min_new_replay_rows),
            "closed_trade_threshold": int(self.min_new_closed_rows),
            "replay_threshold": int(self.min_new_replay_rows),
            "cooldown_minutes": int(self.cooldown_minutes),
            "progress_ratio": max(progress_closed, progress_replay),
            "last_action": action,
            "last_retrained_at": last_retrained_at,
            "status_schema": "incremental_runtime_patch_v2",
        }

    def _write_status(self, closed_rows: int, replay_rows: int, action: str):
        state = self._read_status_state()
        last_trained_closed_rows = int(state.get("last_trained_closed_rows", 0) or 0)
        last_trained_replay_rows = int(state.get("last_trained_replay_rows", 0) or 0)
        last_retrained_at = state.get("last_retrained_at")
        self.status_file.write_text(action + "\n", encoding="utf-8")
        pd.DataFrame([
            self._status_payload(
                closed_rows=closed_rows,
                replay_rows=replay_rows,
                last_trained_closed_rows=last_trained_closed_rows,
                last_trained_replay_rows=last_trained_replay_rows,
                last_retrained_at=last_retrained_at,
                action=action,
            )
        ]).to_csv(self.status_csv, index=False)

    def maybe_retrain(self, force=False, reason="scheduled_cycle_check"):
        closed_df = self._safe_read(self.closed_file)
        replay_df = self._safe_read(self.replay_file)
        backtest_df = self._safe_read(self.backtest_summary_file)

        closed_rows = len(closed_df)
        replay_rows = len(replay_df)
        status = self._read_status_state()
        last_trained_closed_rows = int(status.get("last_trained_closed_rows", 0) or 0)
        last_trained_replay_rows = int(status.get("last_trained_replay_rows", 0) or 0)
        last_retrained_at = pd.to_datetime(status.get("last_retrained_at"), errors="coerce", utc=True)

        new_closed_rows = max(0, closed_rows - last_trained_closed_rows)
        new_replay_rows = max(0, replay_rows - last_trained_replay_rows)
        pnl_degraded = False
        if not backtest_df.empty and "average_pnl" in backtest_df.columns:
            pnl_degraded = self._safe_float(backtest_df.iloc[-1].get("average_pnl"), 0.0) < 0

        force_every_closed_trade = _env_bool("RETRAIN_ON_EVERY_CLOSED_TRADE", True)
        if force and not force_every_closed_trade:
            force = False

        cooldown_active = False
        if pd.notna(last_retrained_at):
            minutes_since = (pd.Timestamp.utcnow() - last_retrained_at).total_seconds() / 60.0
            cooldown_active = minutes_since < self.cooldown_minutes

        should_retrain = (
            force
            or (
                (
                    new_closed_rows >= self.min_new_closed_rows
                    or new_replay_rows >= self.min_new_replay_rows
                    or pnl_degraded
                ) and not cooldown_active
            )
        )

        if not should_retrain:
            reason = (
                f"Retrain skipped: new_closed={new_closed_rows}/{self.min_new_closed_rows}, "
                f"new_replay={new_replay_rows}/{self.min_new_replay_rows}, "
                f"cooldown_active={cooldown_active}, pnl_degraded={pnl_degraded}, trigger={reason}"
            )
            self.status_file.write_text(reason + "\n", encoding="utf-8")
            pd.DataFrame([
                self._status_payload(
                    closed_rows=closed_rows,
                    replay_rows=replay_rows,
                    last_trained_closed_rows=last_trained_closed_rows,
                    last_trained_replay_rows=last_trained_replay_rows,
                    last_retrained_at=status.get("last_retrained_at"),
                    action=reason,
                )
            ]).to_csv(self.status_csv, index=False)
            return False

        WindowedStage1Models(logs_dir=self.logs_dir, weights_dir=self.weights_dir).train()
        WindowedStage2TemporalModels(logs_dir=self.logs_dir, weights_dir=self.weights_dir).train()
        _resume_ppo_train_model(timesteps=self.rl_timesteps)
        promoted, message = self._promote_if_better(closed_rows, replay_rows)
        now_iso = pd.Timestamp.utcnow().isoformat()
        self.status_file.write_text(message + "\n", encoding="utf-8")
        pd.DataFrame([
            self._status_payload(
                closed_rows=closed_rows,
                replay_rows=replay_rows,
                last_trained_closed_rows=closed_rows,
                last_trained_replay_rows=replay_rows,
                last_retrained_at=now_iso,
                action=f"{message} | reason={reason}",
            )
        ]).to_csv(self.status_csv, index=False)
        return promoted


def apply_retrainer_runtime_patch():
    try:
        import retrainer as retrainer_module
        retrainer_module.Retrainer = IncrementalRetrainer
    except Exception:
        pass
    try:
        import rl_trainer as rl_trainer_module
        rl_trainer_module.train_model = _resume_ppo_train_model
    except Exception:
        pass
