from trade_lifecycle import TradeState
import os
import json
import time
import logging
import re
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

from leaderboard_scraper import run_scraper_cycle
from market_monitor import (
    fetch_btc_markets,
    save_market_snapshot,
    fetch_markets_by_slugs,
    fetch_markets_by_slug_prefix,
)
from feature_builder import FeatureBuilder
from signal_engine import SignalEngine
from whale_tracker import WhaleTracker
from alerts_engine import AlertsEngine
from trader_analytics import TraderAnalytics
from historical_dataset_builder import HistoricalDatasetBuilder
from backtester import StrategyBacktester
from autonomous_monitor import AutonomousMonitor
from retrainer import Retrainer
from execution_client import ExecutionClient
from order_manager import OrderManager
from trade_manager import TradeManager
from trade_lifecycle import TradeLifecycle
from polytrade_env import PolyTradeEnv
from model_inference import ModelInference
from stage1_inference import Stage1Inference
from stage2_temporal_inference import Stage2TemporalInference
from model_artifact_staging import snapshot_artifact_state
from return_calibration import clip_expected_return_series
from ops_state_sync import sync_ops_state_to_db
from stage3_hybrid import Stage3HybridScorer
from config import TradingConfig
from strategy_layers import EntryRuleLayer, PredictionLayer
from rl_entry_inference import EntryRLInference
from rl_position_inference import PositionRLInference
from rl_observation_schemas import prepare_entry_observation, prepare_position_observation
from shadow_purgatory import ShadowPurgatory
from live_position_book import LivePositionBook
from live_pnl import LivePnLCalculator
from reconciliation_service import ReconciliationService
from mismatch_detector import StateMismatchDetector
from db import Database
from money_manager import MoneyManager
from entry_aggression import apply_entry_cadence_boost
from orderbook_guard import OrderBookGuard
from token_utils import normalize_token_id
from order_flow_analyzer import OrderFlowAnalyzer
from technical_analyzer import TechnicalAnalyzer
from sentiment_analyzer import SentimentAnalyzer
from macro_analyzer import MacroAnalyzer
from onchain_analyzer import OnChainAnalyzer
from trade_feedback_learner import TradeFeedbackLearner
from position_telemetry import PositionTelemetry
try:
    from inference_runtime_guard import (
        reset_cycle as _reset_inference_runtime_guard,
        has_errors as _inference_guard_has_errors,
        get_errors as _inference_guard_get_errors,
    )
except Exception:
    def _reset_inference_runtime_guard():
        return None

    def _inference_guard_has_errors():
        return False

    def _inference_guard_get_errors(limit=None):
        return []

# Configure logging for zero-intervention monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure logging directory exists
os.makedirs("logs", exist_ok=True)
EXECUTION_FILE = "logs/execution_log.csv"
SIGNALS_FILE = "logs/signals.csv"
RAW_CANDIDATES_FILE = "logs/raw_candidates.csv"
MARKETS_FILE = "logs/markets.csv"
CANDIDATE_DECISIONS_FILE = "logs/candidate_decisions.csv"
CANDIDATE_CYCLE_STATS_FILE = "logs/candidate_cycle_stats.csv"
ALWAYS_ON_STATE_FILE = Path("logs/always_on_slug_state.json")
UPDOWN_SLUG_RE = re.compile(r"^btc-updown-(?P<minutes>\d+)m-(?P<epoch>\d{9,})$")


class StatefulRecurrentBrain:
    def __init__(self, model):
        self.model = model
        self.lstm_state = None
        self.episode_start = np.array([True], dtype=bool)

    def predict(self, obs, deterministic=True):
        action, self.lstm_state = self.model.predict(obs, state=self.lstm_state, episode_start=self.episode_start, deterministic=deterministic)
        self.episode_start = np.array([False], dtype=bool)
        return action, self.lstm_state

    def reset_memory(self):
        self.lstm_state = None
        self.episode_start = np.array([True], dtype=bool)


def load_brain(model_path="weights/ppo_polytrader"):
    """Loads the legacy shared Reinforcement Learning model."""
    recurrent_path = "weights/recurrent_ppo_polytrader"
    vecnorm_path = "weights/ppo_polytrader_vecnormalize.pkl"
    try:
        if RecurrentPPO is not None and os.path.exists(recurrent_path + ".zip"):
            model = RecurrentPPO.load(recurrent_path)
            logging.info(f"[+] Successfully loaded recurrent RL brain from {recurrent_path}.zip")
            return StatefulRecurrentBrain(model)

        env = None
        if os.path.exists(vecnorm_path):
            try:
                env = DummyVecEnv([lambda: PolyTradeEnv()])
                env = VecNormalize.load(vecnorm_path, env)
                env.training = False
                env.norm_reward = False
                logging.info(f"[+] Loaded RL normalization stats from {vecnorm_path}")
            except Exception as vec_exc:
                logging.warning(f"[!] Failed to load RL normalization stats from {vecnorm_path}: {vec_exc}")
                env = None

        model = PPO.load(model_path, env=env)
        logging.info(f"[+] Successfully loaded legacy shared RL brain from {model_path}.zip")
        return model
    except Exception as e:
        logging.error(f"[-] Failed to load model from {model_path}. Error: {e}")
        return None


def load_entry_brain():
    try:
        loader = EntryRLInference()
        model = loader.load()
        if model is not None:
            logging.info("[+] Successfully loaded entry RL brain.")
            return loader
    except Exception as exc:
        logging.warning(f"[!] Failed to load entry RL brain: {exc}")
    return None


def load_position_brain():
    try:
        loader = PositionRLInference()
        model = loader.load()
        if model is not None:
            logging.info("[+] Successfully loaded position RL brain.")
            return loader
    except Exception as exc:
        logging.warning(f"[!] Failed to load position RL brain: {exc}")
    return None


def prepare_observation(feature_row, legacy=False):
    """Compatibility wrapper for the shared entry observation schema."""
    return prepare_entry_observation(feature_row, legacy=legacy)


def safe_read_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def append_csv_record(path, record):
    df = pd.DataFrame([record])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def append_csv_frame(path, df):
    if df is None or df.empty:
        return
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value, default=0.0):
    try:
        num = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(num):
        return float(default)
    return num


def _parse_timestamp_utc(value):
    if value in (None, "", "nan", "None"):
        return None
    try:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric):
            numeric = float(numeric)
            if numeric > 1e17:
                ts = pd.to_datetime(numeric, utc=True, errors="coerce", unit="ns")
            elif numeric > 1e14:
                ts = pd.to_datetime(numeric, utc=True, errors="coerce", unit="us")
            elif numeric > 1e11:
                ts = pd.to_datetime(numeric, utc=True, errors="coerce", unit="ms")
            elif numeric > 1e9:
                ts = pd.to_datetime(numeric, utc=True, errors="coerce", unit="s")
            else:
                ts = pd.to_datetime(value, utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def _signal_age_seconds(signal_row: dict):
    if not isinstance(signal_row, dict):
        return None
    ts_cols = (
        "signal_observed_at",
        "market_data_timestamp",
        "timestamp",
        "updated_at",
        "market_timestamp",
        "market_quote_timestamp",
        "market_snapshot_timestamp",
        "market_updated_at",
        "market_last_trade_ts",
        "last_trade_timestamp",
    )
    for col in ts_cols:
        ts = _parse_timestamp_utc(signal_row.get(col))
        if ts is not None:
            return max(0.0, (pd.Timestamp.now(tz="UTC") - ts).total_seconds())
    return None


def _reject_category(reason: str):
    text = str(reason or "").strip().lower()
    if text in {"rule_veto"}:
        return "rule_veto"
    if "size" in text or "min" in text:
        return "min_size"
    if "freeze" in text or "kill_switch" in text or "kill-switch" in text:
        return "freeze"
    if "orderbook" in text or "liquidity" in text or "spread" in text:
        return "no_liquidity"
    if "stale" in text or "freshness" in text:
        return "stale_market_data"
    if "duplicate" in text:
        return "duplicate"
    if "model" in text:
        return "model_gate"
    return "other"


def log_raw_candidates(candidates_df):
    if candidates_df is None or candidates_df.empty:
        return
    raw_df = candidates_df.copy()
    if "timestamp" not in raw_df.columns:
        raw_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_csv_frame(RAW_CANDIDATES_FILE, raw_df)


def log_ranked_signal(signal_row):
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market": signal_row.get("market_title", "Unknown Market"),
        "wallet_copied": str(signal_row.get("trader_wallet", signal_row.get("wallet_copied", "Unknown"))),
        "token_id": signal_row.get("token_id"),
        "condition_id": signal_row.get("condition_id"),
        "order_side": signal_row.get("order_side", signal_row.get("trade_side", "BUY")),
        "trade_side": signal_row.get("trade_side", signal_row.get("order_side", "BUY")),
        "outcome_side": signal_row.get("outcome_side", signal_row.get("side", "UNKNOWN")),
        "entry_intent": signal_row.get("entry_intent", "OPEN_LONG"),
        "side": signal_row.get("outcome_side", signal_row.get("side", "UNKNOWN")),
        "signal_label": signal_row.get("signal_label", "UNKNOWN"),
        "confidence": signal_row.get("confidence", 0.0),
        "reason": signal_row.get("reason", ""),
        "market_url": signal_row.get("market_url"),
        "trader_win_rate": signal_row.get("trader_win_rate"),
        "normalized_trade_size": signal_row.get("normalized_trade_size"),
        "current_price": signal_row.get("current_price"),
        "time_left": signal_row.get("time_left"),
        "liquidity_score": signal_row.get("liquidity_score"),
        "volume_score": signal_row.get("volume_score"),
        "probability_momentum": signal_row.get("probability_momentum"),
        "volatility_score": signal_row.get("volatility_score"),
        "whale_pressure": signal_row.get("whale_pressure"),
        "market_structure_score": signal_row.get("market_structure_score"),
        "volatility_risk": signal_row.get("volatility_risk"),
        "time_decay_score": signal_row.get("time_decay_score"),
        "market_structure": signal_row.get("market_structure", "UNKNOWN"),
        "trend_score": signal_row.get("trend_score", 0.5),
        "fgi_value": signal_row.get("fgi_value", 50),
        "fgi_status": signal_row.get("fgi_status", "Neutral"),
        "btc_funding_rate": signal_row.get("btc_funding_rate", 0.0),
        "is_overheated_long": signal_row.get("is_overheated_long", False),
    }
    append_csv_record(SIGNALS_FILE, record)


def choose_action(
    signal_row,
    entry_rule: EntryRuleLayer,
    entry_brain=None,
    legacy_brain=None,
    decision_meta=None,
    precomputed_rule_eval=None,
    precomputed_rule_allows=None,
):
    """Choose a trading action for the given signal row.

    Args:
        precomputed_rule_eval:   Pre-evaluated result from entry_rule.evaluate().
                                 If supplied, the function skips a second evaluation
                                 (eliminates the double-call bug).
        precomputed_rule_allows: Corresponding bool extracted from precomputed_rule_eval.
    """
    def _is_truthy(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value) != 0.0
        text = str(value or "").strip().lower()
        return text in {"1", "true", "yes", "on"}

    if decision_meta is None:
        decision_meta = {}
    decision_meta["rule_vetoed_rl_action"] = False
    decision_meta["vetoed_action"] = None

    action_val = 0
    force_candidate = (
        _is_truthy(signal_row.get("force_candidate"))
        or str(signal_row.get("signal_source", "")).strip().lower() == "always_on_market"
    )
    if entry_brain is not None:
        try:
            _av = entry_brain.predict(signal_row)
            if _av is not None:
                action_val = int(_av)
        except Exception:
            pass
    if action_val == 0 and legacy_brain is not None:
        try:
            obs = prepare_observation(signal_row)
            action, _ = legacy_brain.predict(obs, deterministic=True)
            action_val = int(action.item() if hasattr(action, "item") else action[0])
        except Exception:
            pass

    # Apply entry rule as veto even when RL says enter.
    # FIX BUG#1: use caller-supplied precomputed evaluation to avoid a second
    # entry_rule.evaluate() call (which could diverge from the outer-loop copy).
    if precomputed_rule_eval is not None:
        rule_eval = precomputed_rule_eval
        rule_allows_entry = (
            precomputed_rule_allows
            if precomputed_rule_allows is not None
            else (
                bool(rule_eval["allow"]) if isinstance(rule_eval, dict)
                else entry_rule.should_enter(signal_row)
            )
        )
    else:
        rule_eval = entry_rule.evaluate(signal_row) if hasattr(entry_rule, "evaluate") else None
        rule_allows_entry = bool(rule_eval["allow"]) if isinstance(rule_eval, dict) else entry_rule.should_enter(signal_row)

    if action_val in (1, 2) and not rule_allows_entry:
        decision_meta["rule_vetoed_rl_action"] = True
        decision_meta["vetoed_action"] = action_val
        if isinstance(rule_eval, dict):
            logging.info(
                "Entry rule vetoed RL action=%d for %s (score=%.4f/%.4f spread=%.4f/%.4f liquidity[%s]=%s/%s)",
                action_val,
                signal_row.get("market_title", signal_row.get("market", "unknown")),
                float(rule_eval.get("score", 0.0) or 0.0),
                float(rule_eval.get("score_threshold", 0.0) or 0.0),
                float(rule_eval.get("spread", 0.0) or 0.0),
                float(rule_eval.get("spread_threshold", 0.0) or 0.0),
                rule_eval.get("liquidity_metric"),
                rule_eval.get("liquidity_value"),
                rule_eval.get("liquidity_threshold"),
            )
        else:
            logging.info(
                "Entry rule vetoed RL action=%d for %s",
                action_val,
                signal_row.get("market_title", signal_row.get("market", "unknown")),
            )
        return 0
    if action_val != 0:
        return action_val

    # For pinned always-on market candidates, allow rule-based fallback
    # even when RL models return HOLD/0, so the bot keeps attempting entries.
    if force_candidate and entry_rule.should_enter(signal_row):
        return 1

    # Optional safety valve: if RL keeps returning HOLD, allow rule fallback.
    # This prevents "stuck forever" behavior when RL is stale or overly conservative.
    allow_rule_fallback_with_rl_hold = os.getenv("ALLOW_RULE_FALLBACK_WITH_RL_HOLD", "true").strip().lower() in {"1", "true", "yes", "on"}
    rl_hold_min_confidence = float(os.getenv("RL_HOLD_FALLBACK_MIN_CONFIDENCE", "0.15") or 0.15)
    aggressive_expected_return_floor = float(getattr(TradingConfig, "ENTRY_INACTIVITY_EXPECTED_RETURN_FLOOR", -0.002))
    if _is_truthy(signal_row.get("activity_target_mode")):
        rl_hold_min_confidence = min(
            rl_hold_min_confidence,
            max(0.02, float(getattr(TradingConfig, "ENTRY_INACTIVITY_CONFIDENCE_BOOST", 0.08))),
        )
    if (entry_brain is not None or legacy_brain is not None) and allow_rule_fallback_with_rl_hold:
        confidence = _safe_float(signal_row.get("confidence", 0.0), default=float("nan"))
        if not np.isfinite(confidence) or confidence <= 0.0:
            confidence = _safe_float(PredictionLayer.select_signal_score(signal_row), default=0.0)
        if confidence >= rl_hold_min_confidence and entry_rule.should_enter(signal_row):
            expected_return = _safe_float(signal_row.get("expected_return", 0.0), default=0.0)
            if expected_return <= aggressive_expected_return_floor:
                return 0
            edge_score = _safe_float(signal_row.get("edge_score", 0.0), default=0.0)
            return 2 if edge_score >= 0.04 else 1

    # FIX V5: If RL models are loaded and fallback is disabled / conditions not met, keep HOLD.
    if entry_brain is not None or legacy_brain is not None:
        return 0

    # Fallback: rule-based decision (ONLY used if no RL models are loaded)
    if not entry_rule.should_enter(signal_row):
        return 0
    edge_score = float(signal_row.get("edge_score", 0.0) or 0.0)
    return 2 if edge_score >= 0.04 else 1


def quote_entry_price(signal_row, slippage=0.01):
    current_price = float(signal_row.get("current_price", signal_row.get("price", 0.5)))
    best_ask = signal_row.get("best_ask")
    base_price = float(best_ask) if best_ask not in [None, ""] and pd.notna(best_ask) else current_price
    return min(0.99, base_price + slippage)


def quote_exit_price(signal_row, slippage=0.01):
    current_price = float(signal_row.get("current_price", signal_row.get("price", 0.5)))
    best_bid = signal_row.get("best_bid")
    base_price = float(best_bid) if best_bid not in [None, ""] and pd.notna(best_bid) else current_price
    return max(0.01, base_price - slippage)


def execute_paper_trade(action, signal_row, fill_price=None, size_usdc=None):
    """Simulates a trade fill and logs the hypothetical position."""
    if action == 0:
        logging.info(f"Brain: IGNORE -> Skipping signal from {signal_row.get('trader_wallet', 'Unknown')[:8]}")
        return

    # FIX H1: Use MoneyManager size if provided, else fall back to fixed amounts
    size = size_usdc if size_usdc and size_usdc > 0 else (10 if action == 1 else 50)
    outcome_side = str(signal_row.get("outcome_side", signal_row.get("side", "UNKNOWN"))).upper()
    signal_price = float(signal_row.get("current_price", signal_row.get("price", 0.5)))
    fill_price = quote_entry_price(signal_row) if fill_price is None else fill_price

    trade_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market": signal_row.get("market_title", "Unknown Market"),
        "wallet_copied": str(signal_row.get("trader_wallet", signal_row.get("wallet_copied", "Unknown"))),
        "token_id": signal_row.get("token_id"),
        "condition_id": signal_row.get("condition_id"),
        "outcome_side": outcome_side,
        "order_side": signal_row.get("order_side", "BUY"),
        "signal_price": round(signal_price, 3),
        "fill_price": round(fill_price, 3),
        "size_usdc": size,
        "signal_label": signal_row.get("signal_label", "UNKNOWN"),
        "confidence": signal_row.get("confidence", 0.0),
        "action_type": "PAPER_TRADE",
    }

    logging.info(
        "Brain: FOLLOW -> Paper filled %s USDC on %s at $%.3f for '%s' | label=%s confidence=%.2f",
        size,
        outcome_side,
        fill_price,
        trade_record["market"],
        trade_record["signal_label"],
        float(trade_record["confidence"]),
    )

    try:
        append_csv_record(EXECUTION_FILE, trade_record)
    except Exception as e:
        logging.error(f"[-] Failed to write to {EXECUTION_FILE}: {e}")


def build_feature_snapshot(row):
    keys = [
        "token_id",
        "condition_id",
        "outcome_side",
        "market_title",
        "confidence",
        "edge_score",
        "expected_return",
        "p_tp_before_sl",
        "temporal_p_tp_before_sl",
        "meta_prob",
        "current_price",
        "entry_price",
        "spread",
    ]
    payload = {k: row.get(k) for k in keys if k in row}
    return json.dumps(payload, default=str)


def log_live_fill_event(signal_row, fill_price, size_usdc, action_type="LIVE_TRADE"):
    trade_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market": signal_row.get("market_title", signal_row.get("market", "Unknown Market")),
        "wallet_copied": str(signal_row.get("trader_wallet", signal_row.get("wallet_copied", "Unknown"))),
        "token_id": signal_row.get("token_id"),
        "condition_id": signal_row.get("condition_id"),
        "outcome_side": str(signal_row.get("outcome_side", signal_row.get("side", "UNKNOWN"))).upper(),
        "order_side": signal_row.get("order_side", signal_row.get("trade_side", "BUY")),
        "signal_price": round(float(signal_row.get("current_price", signal_row.get("price", fill_price)) or fill_price), 3),
        "fill_price": round(float(fill_price), 3),
        "size_usdc": float(size_usdc),
        "signal_label": signal_row.get("signal_label", "UNKNOWN"),
        "confidence": signal_row.get("confidence", 0.0),
        "action_type": action_type,
    }
    try:
        append_csv_record(EXECUTION_FILE, trade_record)
    except Exception as e:
        logging.error(f"[-] Failed to write live fill to {EXECUTION_FILE}: {e}")


def minutes_since_last_entry(db) -> float | None:
    try:
        rows = db.query_all(
            """
            SELECT created_at
            FROM candidate_decisions
            WHERE final_decision IN ('ENTRY_FILLED', 'PAPER_OPENED')
            ORDER BY COALESCE(created_at, '') DESC
            LIMIT 1
            """
        )
        if rows:
            ts = pd.to_datetime(rows[0].get("created_at"), errors="coerce", utc=True)
            if pd.notna(ts):
                return max(0.0, float((pd.Timestamp.utcnow() - ts).total_seconds() / 60.0))
    except Exception:
        pass
    try:
        if EXECUTION_FILE.exists():
            exec_df = pd.read_csv(EXECUTION_FILE, engine="python", on_bad_lines="skip")
            if not exec_df.empty and "timestamp" in exec_df.columns:
                if "action_type" in exec_df.columns:
                    exec_df = exec_df[exec_df["action_type"].astype(str).str.upper().eq("LIVE_TRADE")]
                ts = pd.to_datetime(exec_df.get("timestamp"), errors="coerce", utc=True).dropna()
                if not ts.empty:
                    return max(0.0, float((pd.Timestamp.utcnow() - ts.max()).total_seconds() / 60.0))
    except Exception:
        pass
    return None


# --- IP SAFEGUARD INTERCEPTOR ---
# Prevents Polymarket IP Bans during 5-second fast-polling
_orig_run_scraper_cycle = run_scraper_cycle
_orig_fetch_btc_markets = fetch_btc_markets
_last_scraper_time = 0
_open_market_cache = {}

def safe_run_scraper_cycle(*args, **kwargs):
    global _last_scraper_time
    import time
    import pandas as pd
    min_interval = max(0, int(os.getenv("SCRAPER_MIN_INTERVAL_SECONDS", "55")))
    now = time.time()
    if min_interval and (now - _last_scraper_time) < min_interval:
        logging.info(
            "Scraper throttle active: skipping wallet scrape for %.1fs (min_interval=%ss).",
            float(min_interval - (now - _last_scraper_time)),
            min_interval,
        )
        return pd.DataFrame()
    out = _orig_run_scraper_cycle(*args, **kwargs)
    _last_scraper_time = now
    return out

def safe_fetch_btc_markets(limit=1000, closed=False, max_offset=0):
    global _open_market_cache
    import time
    import pandas as pd
    cache_ttl = max(1, int(os.getenv("MARKET_CACHE_TTL_SECONDS", "55")))

    # Cache open snapshots by request shape so deeper scans are not silently
    # downgraded to a shallower cached page-0 result.
    cache_key = (int(limit), int(max_offset or 0))
    if not closed:
        now_ts = time.time()
        cached = _open_market_cache.get(cache_key)
        if cached is not None and (now_ts - float(cached.get("ts", 0.0))) < cache_ttl:
            cached_df = cached.get("df")
            if isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
                return cached_df

    res = _orig_fetch_btc_markets(limit=limit, closed=closed, max_offset=max_offset)
    if not closed and not res.empty:
        _open_market_cache[cache_key] = {"ts": time.time(), "df": res}
    return res

run_scraper_cycle = safe_run_scraper_cycle
fetch_btc_markets = safe_fetch_btc_markets
# --------------------------------

def main_loop():
    """The continuous autonomous loop (research + paper-trading mode)."""
    logging.info("Initializing LIVE PolyMarket Supervisor...")

    def _env_int(name: str, default: int, minimum: int = 0, maximum: int = 100000) -> int:
        try:
            value = int(os.getenv(name, str(default)) or default)
        except Exception:
            value = int(default)
        value = max(int(minimum), value)
        value = min(int(maximum), value)
        return value

    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)) or default)
        except Exception:
            return float(default)

    entry_brain = load_entry_brain()
    position_brain = load_position_brain()
    legacy_brain = None
    if entry_brain is None or position_brain is None:
        legacy_brain = load_brain()

    if entry_brain is None:
        logging.warning("Entry RL brain unavailable. Falling back to legacy shared RL or rules.")
    if position_brain is None:
        logging.warning("Position RL brain unavailable. Falling back to legacy shared RL or holds.")
    if entry_brain is None and position_brain is None and legacy_brain is None:
        logging.warning("No RL brain available. Continuing with supervised-first ranking path.")

    entry_model_name = "ppo_entry_policy" if entry_brain is not None else "ppo_polytrader_legacy_entry" if legacy_brain is not None else "no_rl_entry"
    position_model_name = "ppo_position_policy" if position_brain is not None else "ppo_polytrader_legacy_position" if legacy_brain is not None else "no_rl_position"
    entry_model_artifact = "weights/ppo_entry_policy.zip" if entry_brain is not None else "weights/ppo_polytrader.zip" if legacy_brain is not None else None
    entry_norm_artifact = "weights/ppo_entry_vecnormalize.pkl" if entry_brain is not None else "weights/ppo_polytrader_vecnormalize.pkl" if legacy_brain is not None else None
    position_model_artifact = "weights/ppo_position_policy.zip" if position_brain is not None else "weights/ppo_polytrader.zip" if legacy_brain is not None else None
    position_norm_artifact = "weights/ppo_position_vecnormalize.pkl" if position_brain is not None else "weights/ppo_polytrader_vecnormalize.pkl" if legacy_brain is not None else None

    feature_builder = FeatureBuilder()
    signal_engine = SignalEngine()
    model_inference = ModelInference()
    stage1_inference = Stage1Inference()
    stage2_inference = Stage2TemporalInference()
    hybrid_scorer = Stage3HybridScorer()
    order_flow_analyzer = OrderFlowAnalyzer(min_usd_volume=500.0, volume_imbalance_threshold=0.75, min_trades_count=3)
    technical_analyzer = TechnicalAnalyzer()
    sentiment_analyzer = SentimentAnalyzer()
    macro_analyzer = MacroAnalyzer()
    onchain_analyzer = OnChainAnalyzer()
    entry_min_score = _env_float("ENTRY_MIN_SCORE", 0.12)
    entry_max_spread = _env_float("ENTRY_MAX_SPREAD", 0.35)
    entry_min_liquidity = _env_float("ENTRY_MIN_LIQUIDITY", 0.5)
    entry_min_liquidity_score = _env_float("ENTRY_MIN_LIQUIDITY_SCORE", 0.005)
    target_entry_interval_minutes = _env_float(
        "TARGET_ENTRY_INTERVAL_MINUTES",
        float(getattr(TradingConfig, "TARGET_ENTRY_INTERVAL_MINUTES", 5)),
    )
    entry_aggression_top_k = _env_int(
        "ENTRY_AGGRESSION_TOP_K",
        int(getattr(TradingConfig, "ENTRY_AGGRESSION_TOP_K", 3)),
        minimum=1,
        maximum=25,
    )
    entry_rule = EntryRuleLayer(
        min_score=entry_min_score,
        max_spread=entry_max_spread,
        min_liquidity=entry_min_liquidity,
        min_liquidity_score=entry_min_liquidity_score,
    )
    logging.info(
        "EntryRule thresholds: min_score=%.4f max_spread=%.4f min_liquidity=%.4f min_liquidity_score=%.4f target_interval=%.1fm top_k=%d",
        entry_min_score,
        entry_max_spread,
        entry_min_liquidity,
        entry_min_liquidity_score,
        target_entry_interval_minutes,
        entry_aggression_top_k,
    )
    whale_tracker = WhaleTracker()
    alerts_engine = AlertsEngine()
    trader_analytics = TraderAnalytics()
    dataset_builder = HistoricalDatasetBuilder()
    backtester = StrategyBacktester()
    trading_mode = "live"
    logging.warning("PAPER MODE DISABLED: Bot is permanently locked into LIVE TRADING mode.")
    execution_client = ExecutionClient()
    order_manager = OrderManager()
    live_position_book = LivePositionBook()
    live_pnl = LivePnLCalculator()
    reconciliation_service = ReconciliationService(execution_client)
    mismatch_detector = StateMismatchDetector()
    trade_manager = TradeManager(logs_dir="logs")
    trade_manager.exec_client = execution_client
    orderbook_guard = OrderBookGuard(max_spread=0.20, min_bid_depth=1, min_ask_depth=1)
    # Persistent cross-cycle cache: token_id -> time.monotonic() of when we last confirmed no orderbook.
    # Avoids redundant 404 API hits every cycle for inactive/resolved markets.
    _ob_no_book_cache: dict = {}  # {token_id: monotonic_timestamp}
    _OB_NO_BOOK_TTL = 1800.0   # 30 minutes — re-check expired markets occasionally
    _money_mgr = MoneyManager()
    autonomous_monitor = AutonomousMonitor()
    retrainer = Retrainer()
    feedback_learner = TradeFeedbackLearner()
    position_telemetry = PositionTelemetry()
    def _refresh_runtime_model_handles(reason="runtime_artifacts_reloaded"):
        nonlocal entry_brain, position_brain, legacy_brain
        nonlocal entry_model_name, position_model_name
        nonlocal entry_model_artifact, entry_norm_artifact
        nonlocal position_model_artifact, position_norm_artifact
        nonlocal model_inference, stage1_inference, stage2_inference

        entry_brain = load_entry_brain()
        position_brain = load_position_brain()
        legacy_brain = None
        if entry_brain is None or position_brain is None:
            legacy_brain = load_brain()

        entry_model_name = "ppo_entry_policy" if entry_brain is not None else "ppo_polytrader_legacy_entry" if legacy_brain is not None else "no_rl_entry"
        position_model_name = "ppo_position_policy" if position_brain is not None else "ppo_polytrader_legacy_position" if legacy_brain is not None else "no_rl_position"
        entry_model_artifact = "weights/ppo_entry_policy.zip" if entry_brain is not None else "weights/ppo_polytrader.zip" if legacy_brain is not None else None
        entry_norm_artifact = "weights/ppo_entry_vecnormalize.pkl" if entry_brain is not None else "weights/ppo_polytrader_vecnormalize.pkl" if legacy_brain is not None else None
        position_model_artifact = "weights/ppo_position_policy.zip" if position_brain is not None else "weights/ppo_polytrader.zip" if legacy_brain is not None else None
        position_norm_artifact = "weights/ppo_position_vecnormalize.pkl" if position_brain is not None else "weights/ppo_polytrader_vecnormalize.pkl" if legacy_brain is not None else None

        # The supervised inference classes already read fresh artifacts from disk
        # on every run, but rebuilding them here makes promoted model activation
        # explicit and keeps startup/reload paths symmetric.
        model_inference = ModelInference()
        stage1_inference = Stage1Inference()
        stage2_inference = Stage2TemporalInference()

        active_artifacts = snapshot_artifact_state("weights")
        autonomous_monitor.write_heartbeat(
            "inference",
            status="ok",
            message=reason,
            extra={
                "active_artifact_count": len(active_artifacts),
                "active_artifacts": sorted(active_artifacts.keys()),
                "legacy_brain_loaded": legacy_brain is not None,
                "entry_brain_loaded": entry_brain is not None,
                "position_brain_loaded": position_brain is not None,
            },
        )
    previous_markets_df = None
    previous_entry_freeze_active = False
    previous_entry_freeze_reason = None
    try:
        shadow_purgatory = ShadowPurgatory()
    except Exception as exc:
        shadow_purgatory = None
        logging.warning("ShadowPurgatory unavailable at startup: %s", exc)
    db = Database()
    try:
        db_report = db.integrity_report()
        if not db_report.get("ok", True):
            logging.error("SQLite integrity check failed: %s", db_report)
            auto_reset = os.getenv("RESET_RUNTIME_STATE_ON_DB_CORRUPTION", "false").strip().lower() in {"1", "true", "yes", "on"}
            if auto_reset:
                backup_dir = db.backup_and_reset_runtime_state("logs")
                logging.warning(
                    "Runtime state reset completed (backup=%s). Model weights were preserved in weights/.",
                    backup_dir,
                )
    except Exception as exc:
        logging.warning("DB integrity preflight failed: %s", exc)


    def _make_position_key(token_id=None, condition_id=None, outcome_side=None, market=None):
        token_id = str(token_id).strip() if token_id not in [None, ""] else ""
        condition_id = str(condition_id).strip() if condition_id not in [None, ""] else ""
        outcome_side = str(outcome_side).strip() if outcome_side not in [None, ""] else ""
        market = str(market).strip() if market not in [None, ""] else ""
        if token_id or condition_id:
            return f"{token_id}|{condition_id}|{outcome_side}"
        if market and outcome_side:
            return f"{market}|{outcome_side}"
        return None
    def _trade_key_from_signal(signal_row):
        """Consistent trade key matching TradeManager._get_trade_key."""
        token_id = str(signal_row.get("token_id", "") or "").strip()
        condition_id = str(signal_row.get("condition_id", "") or "").strip()
        market = signal_row.get("market_title") or signal_row.get("market")
        outcome_side = signal_row.get("outcome_side") or signal_row.get("side")
        outcome_side = str(outcome_side or "").strip()
        if (token_id or condition_id) and outcome_side:
            return f"{token_id}|{condition_id}|{outcome_side}"
        return f"{market}|{outcome_side}" if market and outcome_side else None

    updown_market_grace_seconds = _env_int("UPDOWN_MARKET_GRACE_SECONDS", 120, minimum=0, maximum=3600)

    def _is_expired_updown_slug(slug_value) -> bool:
        slug = str(slug_value or "").strip().lower()
        if not slug:
            return False
        match = UPDOWN_SLUG_RE.match(slug)
        if not match:
            return False
        try:
            duration_minutes = max(1, int(match.group("minutes")))
            start_epoch = int(match.group("epoch"))
        except Exception:
            return False
        start_ts = pd.to_datetime(start_epoch, unit="s", utc=True, errors="coerce")
        if pd.isna(start_ts):
            return False
        end_ts = start_ts + pd.Timedelta(minutes=duration_minutes)
        expiry_cutoff = end_ts + pd.Timedelta(seconds=updown_market_grace_seconds)
        return pd.Timestamp.now(tz="UTC") > expiry_cutoff

    def _extract_open_market_universe(markets_df: pd.DataFrame):
        open_token_ids: set[str] = set()
        open_condition_ids: set[str] = set()
        open_market_slugs: set[str] = set()
        if markets_df is None or markets_df.empty:
            return open_token_ids, open_condition_ids, open_market_slugs

        def _add_token(value):
            tok = normalize_token_id(value)
            if tok:
                open_token_ids.add(str(tok))

        for _, market_row in markets_df.iterrows():
            row = market_row.to_dict()
            closed_flag = _is_truthy(row.get("closed"))
            active_raw = row.get("active")
            active_flag = True if active_raw is None or (isinstance(active_raw, float) and np.isnan(active_raw)) else _is_truthy(active_raw)
            if closed_flag or not active_flag:
                continue
            end_raw = row.get("end_date", row.get("endDate"))
            if end_raw is not None and not (isinstance(end_raw, float) and np.isnan(end_raw)):
                end_ts = pd.to_datetime(end_raw, utc=True, errors="coerce")
                if pd.notna(end_ts) and end_ts <= pd.Timestamp.now(tz="UTC"):
                    continue

            cond = str(row.get("condition_id") or "").strip().lower()
            if cond and cond != "nan":
                open_condition_ids.add(cond)

            slug = str(row.get("slug") or row.get("market_slug") or "").strip().lower()
            if slug and slug != "nan":
                if _is_expired_updown_slug(slug):
                    continue
                open_market_slugs.add(slug)

            _add_token(row.get("yes_token_id"))
            _add_token(row.get("no_token_id"))
            _add_token(row.get("token_id"))

            raw_ids = row.get("clob_token_ids")
            if raw_ids is None:
                continue
            if isinstance(raw_ids, float) and np.isnan(raw_ids):
                continue

            if isinstance(raw_ids, (list, tuple, set)):
                token_candidates = list(raw_ids)
            else:
                raw_text = str(raw_ids).strip()
                if not raw_text or raw_text.lower() == "nan":
                    continue
                if raw_text.startswith("[") and raw_text.endswith("]"):
                    try:
                        parsed = json.loads(raw_text)
                    except Exception:
                        parsed = [x.strip().strip("'\"") for x in raw_text.strip("[]").split(",") if x.strip()]
                    token_candidates = list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
                else:
                    token_candidates = [raw_text]

            for tok in token_candidates:
                _add_token(tok)

        return open_token_ids, open_condition_ids, open_market_slugs

    def _dedupe_signals_df(signals_df: pd.DataFrame):
        if signals_df is None or signals_df.empty:
            return signals_df
        work = signals_df.copy()
        if "timestamp" in work.columns:
            work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
            max_age_hours = max(1, int(os.getenv("SIGNAL_MAX_AGE_HOURS", os.getenv("SIGNAL_LOOKBACK_HOURS", "24"))))
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=max_age_hours)
            work = work[work["timestamp"].notna() & (work["timestamp"] >= cutoff)]
        dedupe_cols = [
            c for c in [
                "trade_id",
                "tx_hash",
                "trader_wallet",
                "token_id",
                "condition_id",
                "order_side",
                "outcome_side",
                "price",
                "size",
                "timestamp",
            ] if c in work.columns
        ]
        if dedupe_cols:
            work = work.drop_duplicates(subset=dedupe_cols, keep="first")
        return work.reset_index(drop=True)

    def _annotate_signal_freshness(signals_df: pd.DataFrame, observed_ts):
        if signals_df is None or signals_df.empty:
            return signals_df
        work = signals_df.copy()
        observed_iso = str(observed_ts)
        if "source_signal_timestamp" not in work.columns:
            work["source_signal_timestamp"] = work.get("signal_observed_at", work.get("timestamp"))
        if "source_market_timestamp" not in work.columns:
            work["source_market_timestamp"] = work.get("market_data_timestamp", work.get("timestamp"))
        if "last_trade_timestamp" not in work.columns:
            work["last_trade_timestamp"] = work.get("timestamp")
        else:
            work["last_trade_timestamp"] = work["last_trade_timestamp"].where(
                work["last_trade_timestamp"].notna(), work.get("timestamp")
            )
        # Use cycle observation time for freshness gates; keep source timestamps separately.
        work["signal_observed_at"] = observed_iso
        work["market_data_timestamp"] = observed_iso
        return work

    def _coerce_confidence(value, default=0.5):
        try:
            conf = float(value)
            return float(np.clip(conf, 0.0, 1.0))
        except Exception:
            return float(default)

    def _normalize_position_action(action_val):
        """
        Normalize position-policy actions for open-position management.
        Supports both 3-action heads (0/1/2 = hold/reduce/exit) and
        shared 6-action heads (3/4/5 = hold/reduce/exit).
        """
        try:
            action_val = int(action_val)
        except Exception:
            return 3
        if action_val in (3, 4, 5):
            return action_val
        if action_val in (0, 1, 2):
            return action_val + 3
        return 3

    def _build_predictive_exit_targets(scored_frame, open_trades):
        """
        Build per-position model-based TP targets from current scored signals.
        Uses expected_return + p_tp gate and enforces a minimum profitable step.
        """
        targets = {}
        if scored_frame is None or scored_frame.empty or not open_trades:
            return targets

        signal_lookup = {}
        for _, row in scored_frame.iterrows():
            row_dict = row.to_dict()
            key = _trade_key_from_signal(row_dict)
            if key:
                signal_lookup[key] = row_dict

        min_profit_step = max(0.01, float(getattr(TradingConfig, "SHADOW_TP_DELTA", 0.04)) * 0.5)
        for trade in open_trades:
            trade_key = _make_position_key(
                token_id=trade.token_id,
                condition_id=trade.condition_id,
                outcome_side=trade.outcome_side,
                market=trade.market,
            )
            if not trade_key:
                continue
            signal = signal_lookup.get(trade_key)
            if not signal:
                continue

            try:
                p_tp = float(signal.get("p_tp_before_sl", 0.0) or 0.0)
                expected_return = float(signal.get("expected_return", 0.0) or 0.0)
                entry_price = float(getattr(trade, "entry_price", 0.0) or 0.0)
                if entry_price <= 0:
                    continue
            except Exception:
                continue

            if p_tp < 0.55 or expected_return <= 0:
                continue

            # Treat expected_return as ROI-style signal, clipped to avoid unrealistic exits.
            expected_return = min(expected_return, 0.25)
            target_price = entry_price * (1.0 + expected_return)
            target_price = max(target_price, entry_price + min_profit_step)
            target_price = min(0.99, target_price)
            if target_price > entry_price:
                targets[trade_key] = target_price
        return targets

    always_on_enabled = os.getenv("ALWAYS_ON_MARKET_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    configured_always_on_slug = str(os.getenv("ALWAYS_ON_MARKET_SLUG", "btc-updown-5m-1774926600") or "").strip()
    always_on_slug = configured_always_on_slug
    always_on_only = os.getenv("ALWAYS_ON_ONLY", "false").strip().lower() in {"1", "true", "yes", "on"}
    always_on_force_entry = os.getenv("ALWAYS_ON_FORCE_ENTRY", "true").strip().lower() in {"1", "true", "yes", "on"}
    always_on_signal_size = float(os.getenv("ALWAYS_ON_SIGNAL_SIZE", "25") or 25)
    always_on_rotate_prefix = "btc-updown-5m-"
    if ALWAYS_ON_STATE_FILE.exists():
        try:
            persisted_state = json.loads(ALWAYS_ON_STATE_FILE.read_text(encoding="utf-8"))
            persisted_slug = str(persisted_state.get("resolved_slug") or "").strip()
            if (
                persisted_slug
                and configured_always_on_slug.lower().startswith(always_on_rotate_prefix)
                and persisted_slug.lower().startswith(always_on_rotate_prefix)
            ):
                always_on_slug = persisted_slug
                logging.info(
                    "Loaded persisted rotating always-on slug: configured=%s persisted=%s",
                    configured_always_on_slug,
                    always_on_slug,
                )
        except Exception as exc:
            logging.warning("Failed to load persisted always-on slug state: %s", exc)

    def _persist_always_on_slug(resolved_slug: str):
        try:
            ALWAYS_ON_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            ALWAYS_ON_STATE_FILE.write_text(
                json.dumps(
                    {
                        "configured_slug": configured_always_on_slug,
                        "resolved_slug": str(resolved_slug or ""),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logging.warning("Failed to persist always-on slug state (%s): %s", resolved_slug, exc)

    entry_max_market_staleness_sec = float(os.getenv("ENTRY_MAX_MARKET_STALENESS_SEC", "90") or 90)
    entry_cycle_staleness_grace_sec = float(os.getenv("ENTRY_CYCLE_STALENESS_GRACE_SEC", "300") or 300)
    entry_require_market_timestamp = _is_truthy(os.getenv("ENTRY_REQUIRE_MARKET_TIMESTAMP", "false"))
    # LIVE-TRADE FIX: calibration gate disabled by default — it blocked all entries when
    # there is no historical edge data in the DB (fresh account / first run).
    enable_calibration_gate = _is_truthy(os.getenv("ENABLE_CALIBRATION_GATE", "false"))
    calibration_lookback_rows = max(50, int(os.getenv("CALIBRATION_LOOKBACK_ROWS", "500") or 500))
    calibration_edge_margin = float(os.getenv("CALIBRATION_EDGE_MARGIN", "0.0015") or 0.0015)
    calibration_min_edge = float(os.getenv("CALIBRATION_MIN_EDGE", "0.0005") or 0.0005)

    # LIVE-TRADE FIX: raised drawdown limit from $8 → $50 and recon mismatches 4 → 20
    # to avoid the kill-switch firing on normal account activity / first fills.
    max_session_drawdown_usdc = float(os.getenv("SESSION_MAX_DRAWDOWN_USDC", "50.0") or 50.0)
    max_session_failed_entries = max(1, int(os.getenv("SESSION_MAX_FAILED_ENTRIES", "15") or 15))
    max_session_reconciliation_mismatches = max(1, int(os.getenv("SESSION_MAX_RECON_MISMATCHES", "20") or 20))
    session_start_balance = None
    session_peak_balance = None
    session_failed_entries = 0
    session_reconciliation_mismatch_count = 0
    session_kill_switch_active = False
    session_kill_switch_reason = None

    def _classify_precycle_reconciliation_report(recon_report: dict) -> dict:
        report = recon_report or {}
        hard_mismatch_counts = {
            "missing_remote_orders": len(report.get("missing_remote_orders", [])),
            "missing_local_orders": len(report.get("missing_local_orders", [])),
            "missing_local_trades": len(report.get("missing_local_trades", [])),
            "order_mismatches": len(report.get("order_mismatches", [])),
        }
        soft_mismatch_counts = {
            "missing_remote_trades": len(report.get("missing_remote_trades", [])),
        }
        hard_mismatch_count = sum(hard_mismatch_counts.values())
        soft_mismatch_count = sum(soft_mismatch_counts.values())
        return {
            "hard_mismatch_count": hard_mismatch_count,
            "soft_mismatch_count": soft_mismatch_count,
            "freeze_entries": hard_mismatch_count > 0,
            "hard_mismatch_counts": hard_mismatch_counts,
            "soft_mismatch_counts": soft_mismatch_counts,
        }

    def _compute_calibrated_edge(signal_row: dict):
        p_tp = _safe_float(signal_row.get("p_tp_before_sl", signal_row.get("confidence", 0.0)), default=0.0)
        expected_return = _safe_float(signal_row.get("expected_return", 0.0), default=0.0)
        calibrated_prob_edge = max(0.0, p_tp - 0.5)
        return calibrated_prob_edge * expected_return

    def _load_calibration_baseline():
        default_baseline = calibration_min_edge
        try:
            rows = db.query_all(
                """
                SELECT calibrated_edge
                FROM candidate_decisions
                WHERE calibrated_edge IS NOT NULL
                  AND calibrated_edge > 0
                ORDER BY decision_id DESC
                LIMIT ?
                """,
                (calibration_lookback_rows,),
            )
            values = []
            for r in rows:
                v = _safe_float(r.get("calibrated_edge"), default=float("nan"))
                if np.isnan(v) or v <= 0:
                    continue
                values.append(v)
            if len(values) < 20:
                return default_baseline
            return max(default_baseline, float(np.quantile(values, 0.60)))
        except Exception:
            return default_baseline

    def _update_session_drawdown(balance_now):
        nonlocal session_start_balance, session_peak_balance, session_kill_switch_active, session_kill_switch_reason
        if balance_now is None:
            return
        balance_now = _safe_float(balance_now, default=0.0)
        if balance_now <= 0:
            return
        if session_start_balance is None:
            session_start_balance = balance_now
        if session_peak_balance is None:
            session_peak_balance = balance_now
        session_peak_balance = max(session_peak_balance, balance_now)
        drawdown = session_peak_balance - balance_now
        if drawdown >= max_session_drawdown_usdc and not session_kill_switch_active:
            session_kill_switch_active = True
            session_kill_switch_reason = f"session_drawdown_limit_hit ({drawdown:.4f} >= {max_session_drawdown_usdc:.4f})"

    def _inject_always_on_signal(signals_df: pd.DataFrame, markets_df: pd.DataFrame):
        nonlocal always_on_slug
        if not always_on_enabled or not always_on_slug:
            return signals_df, markets_df
        now_iso = datetime.now(timezone.utc).isoformat()
        out_signals = signals_df.copy() if signals_df is not None else pd.DataFrame()
        out_markets = markets_df.copy() if markets_df is not None else pd.DataFrame()

        if out_markets.empty or "slug" not in out_markets.columns or always_on_slug not in set(out_markets["slug"].dropna().astype(str)):
            fetched = fetch_markets_by_slugs([always_on_slug])
            if fetched is not None and not fetched.empty:
                if out_markets is None or out_markets.empty:
                    out_markets = fetched.copy()
                else:
                    out_markets = pd.concat([out_markets, fetched], ignore_index=True)
                dedupe_col = "slug" if "slug" in out_markets.columns else "market_id" if "market_id" in out_markets.columns else None
                if dedupe_col:
                    out_markets = out_markets.drop_duplicates(subset=[dedupe_col], keep="last")
                out_markets = out_markets.loc[:, ~out_markets.columns.duplicated()]

        if out_markets is None or out_markets.empty or "slug" not in out_markets.columns:
            return out_signals, out_markets

        target_rows = out_markets[out_markets["slug"].astype(str) == always_on_slug]
        resolved_slug = always_on_slug
        if target_rows.empty:
            # BTC up/down 5m markets rotate slug timestamps.
            # If the configured slug is stale, fall back to the latest open slug with the same base prefix.
            rotating_prefix = "btc-updown-5m-"
            requested_slug = str(always_on_slug or "").strip().lower()
            if requested_slug.startswith(rotating_prefix):
                prefix_rows = out_markets[out_markets["slug"].astype(str).str.lower().str.startswith(rotating_prefix)]
                if prefix_rows is None or prefix_rows.empty:
                    fetched_prefix = fetch_markets_by_slug_prefix(
                        rotating_prefix,
                        limit=500,
                        max_offset=2000,
                        closed=False,
                    )
                    if fetched_prefix is not None and not fetched_prefix.empty:
                        out_markets = pd.concat([out_markets, fetched_prefix], ignore_index=True)
                        if "slug" in out_markets.columns:
                            out_markets = out_markets.drop_duplicates(subset=["slug"], keep="last")
                        out_markets = out_markets.loc[:, ~out_markets.columns.duplicated()]
                        prefix_rows = out_markets[out_markets["slug"].astype(str).str.lower().str.startswith(rotating_prefix)]
                if prefix_rows is not None and not prefix_rows.empty:
                    if "closed" in prefix_rows.columns:
                        open_mask = ~prefix_rows["closed"].astype(str).str.lower().isin({"1", "true", "yes"})
                        open_rows = prefix_rows[open_mask]
                        if open_rows is not None and not open_rows.empty:
                            prefix_rows = open_rows
                    if "end_date" in prefix_rows.columns:
                        _end = pd.to_datetime(prefix_rows["end_date"], utc=True, errors="coerce")
                        prefix_rows = prefix_rows.assign(_end_date_sort=_end).sort_values(
                            by="_end_date_sort", ascending=False, na_position="last"
                        )
                    target_rows = prefix_rows.head(1)
                    if target_rows is not None and not target_rows.empty:
                        resolved_slug = str(target_rows.iloc[-1].get("slug") or always_on_slug)
                        logging.info(
                            "Always-on slug fallback: requested=%s resolved=%s",
                            always_on_slug,
                            resolved_slug,
                        )
                        if resolved_slug and resolved_slug != always_on_slug:
                            always_on_slug = resolved_slug
                            _persist_always_on_slug(resolved_slug)
        if target_rows.empty:
            logging.warning("Always-on market slug not found this cycle: %s", always_on_slug)
            return out_signals, out_markets

        market_row = target_rows.iloc[-1].to_dict()
        yes_token = market_row.get("yes_token_id")
        no_token = market_row.get("no_token_id")
        condition_id = market_row.get("condition_id")
        try:
            yes_price = float(market_row.get("current_price", market_row.get("last_trade_price", 0.5)) or 0.5)
        except Exception:
            yes_price = 0.5
        yes_price = float(np.clip(yes_price, 0.01, 0.99))

        pref_side = str(os.getenv("ALWAYS_ON_MARKET_SIDE", "AUTO") or "AUTO").strip().upper()
        if pref_side not in {"YES", "NO"}:
            pref_side = "YES" if yes_price >= 0.5 else "NO"
        outcome_side = pref_side
        token_id = yes_token if outcome_side == "YES" else no_token
        if token_id in [None, ""]:
            fallback_token = yes_token or no_token
            if fallback_token in [None, ""]:
                logging.warning("Always-on market has no tradable token ids for slug=%s", always_on_slug)
                return out_signals, out_markets
            token_id = fallback_token
            outcome_side = "YES" if token_id == yes_token else "NO"

        signal_price = yes_price if outcome_side == "YES" else float(np.clip(1.0 - yes_price, 0.01, 0.99))
        synthetic_signal = {
            "trade_id": f"always_on_{resolved_slug}_{int(time.time())}",
            "tx_hash": None,
            "trader_wallet": "system_always_on",
            "market_title": market_row.get("market_title", market_row.get("question", always_on_slug)),
            "market_slug": resolved_slug,
            "token_id": str(token_id),
            "condition_id": condition_id,
            "order_side": "BUY",
            "trade_side": "BUY",
            "outcome_side": outcome_side,
            "entry_intent": "OPEN_LONG",
            "side": outcome_side,
            "price": signal_price,
            "size": always_on_signal_size,
            "timestamp": now_iso,
            "signal_source": "always_on_market",
            "force_candidate": 1 if always_on_force_entry else 0,
        }
        out_signals = pd.concat([out_signals, pd.DataFrame([synthetic_signal])], ignore_index=True)
        logging.info(
            "Always-on signal injected for slug=%s side=%s token=%s price=%.4f",
            resolved_slug,
            outcome_side,
            str(token_id)[:16],
            signal_price,
        )
        if resolved_slug and resolved_slug.lower().startswith(always_on_rotate_prefix):
            _persist_always_on_slug(resolved_slug)
        return out_signals, out_markets


    while True:
        try:
            cycle_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            live_positions_df_for_cycle = pd.DataFrame()
            trajectory_metrics = {}
            pre_cycle_entry_freeze = False
            pre_cycle_freeze_reason = None
            pre_cycle_freeze_detail = {}
            _reset_inference_runtime_guard()
            reset_failed_each_cycle = _is_truthy(os.getenv("RESET_FAILED_ORDERS_EACH_CYCLE", "false"))
            if (
                reset_failed_each_cycle
                and trading_mode == "live"
                and order_manager is not None
                and hasattr(order_manager, "risk")
                and hasattr(order_manager.risk, "reset_failed_orders")
            ):
                order_manager.risk.reset_failed_orders()

            if trading_mode == "live" and order_manager is not None:
                try:
                    bal_now, _ = order_manager._get_available_balance(
                        asset_type="COLLATERAL",
                        use_onchain_fallback=False,
                    )
                    _update_session_drawdown(bal_now)
                except Exception:
                    pass

            if session_kill_switch_active:
                pre_cycle_entry_freeze = True
                pre_cycle_freeze_reason = "session_kill_switch"
                pre_cycle_freeze_detail = {
                    "reason": session_kill_switch_reason or "session_limit_hit",
                    "failed_entries": session_failed_entries,
                    "reconciliation_mismatches": session_reconciliation_mismatch_count,
                    "max_failed_entries": max_session_failed_entries,
                    "max_reconciliation_mismatches": max_session_reconciliation_mismatches,
                    "max_drawdown_usdc": max_session_drawdown_usdc,
                    "session_start_balance": session_start_balance,
                    "session_peak_balance": session_peak_balance,
                }
                autonomous_monitor.write_heartbeat(
                    "risk",
                    status="error",
                    message="session_kill_switch_active",
                    extra=pre_cycle_freeze_detail,
                )
            if trading_mode == "live" and reconciliation_service is not None:
                try:
                    sync_summary = reconciliation_service.sync_orders_and_fills()
                    if not sync_summary: sync_summary = {} # BUG 9 FIX
                    logging.info("Exchange reconciliation synced orders=%s fills=%s", sync_summary.get("orders", 0), sync_summary.get("fills", 0))
                except Exception as exc:
                    logging.warning("Exchange reconciliation failed: %s", exc)
                try:
                    # Strict pre-cycle sync: rebuild and reconcile live open/closed state
                    # BEFORE evaluating any new entry opportunities.
                    dead_tokens = orderbook_guard.dead_token_ids()
                    if dead_tokens:
                        live_position_book.close_dead_token_positions(dead_tokens)
                        
                    live_position_book.rebuild_from_db()
                    pre_positions_df = live_position_book.get_enriched_open_positions(scored_df=None)
                    trade_manager.reconcile_live_positions(reconciled_positions_df=pre_positions_df)

                    recon_report, _, _ = reconciliation_service.reconcile()
                    recon_classification = _classify_precycle_reconciliation_report(recon_report)
                    hard_mismatch_count = recon_classification["hard_mismatch_count"]
                    soft_mismatch_count = recon_classification["soft_mismatch_count"]
                    if hard_mismatch_count > 0:
                        session_reconciliation_mismatch_count += 1
                        if (
                            session_reconciliation_mismatch_count >= max_session_reconciliation_mismatches
                            and not session_kill_switch_active
                        ):
                            session_kill_switch_active = True
                            session_kill_switch_reason = (
                                "session_reconciliation_mismatch_limit_hit "
                                f"({session_reconciliation_mismatch_count} >= {max_session_reconciliation_mismatches})"
                            )
                        pre_cycle_entry_freeze = True
                        pre_cycle_freeze_reason = "precycle_reconciliation_mismatch"
                        pre_cycle_freeze_detail = {
                            "mismatch_count": hard_mismatch_count + soft_mismatch_count,
                            "hard_mismatch_count": hard_mismatch_count,
                            "soft_mismatch_count": soft_mismatch_count,
                            "missing_remote_orders": recon_report.get("missing_remote_orders", []),
                            "missing_local_orders": recon_report.get("missing_local_orders", []),
                            "missing_remote_trades": recon_report.get("missing_remote_trades", []),
                            "missing_local_trades": recon_report.get("missing_local_trades", []),
                            "order_mismatches": recon_report.get("order_mismatches", []),
                        }
                        logging.error(
                            "Pre-cycle hard reconciliation mismatch detected. Freezing new entries for this cycle: %s",
                            pre_cycle_freeze_detail,
                        )
                        autonomous_monitor.write_heartbeat(
                            "reconciliation",
                            status="error",
                            message="precycle_reconciliation_mismatch",
                            extra=pre_cycle_freeze_detail,
                        )
                    elif soft_mismatch_count > 0:
                        session_reconciliation_mismatch_count = 0
                        pre_cycle_freeze_detail = {
                            "mismatch_count": soft_mismatch_count,
                            "hard_mismatch_count": hard_mismatch_count,
                            "soft_mismatch_count": soft_mismatch_count,
                            "missing_remote_trades": recon_report.get("missing_remote_trades", []),
                        }
                        logging.warning(
                            "Pre-cycle soft reconciliation drift detected. Continuing without entry freeze: %s",
                            pre_cycle_freeze_detail,
                        )
                        autonomous_monitor.write_heartbeat(
                            "reconciliation",
                            status="warn",
                            message="precycle_reconciliation_soft_drift",
                            extra=pre_cycle_freeze_detail,
                        )
                    else:
                        session_reconciliation_mismatch_count = 0
                        autonomous_monitor.write_heartbeat(
                            "reconciliation",
                            status="ok",
                            message="precycle_sync_ok",
                            extra={"open_positions": len(pre_positions_df) if pre_positions_df is not None else 0},
                        )
                except Exception as exc:
                    logging.warning("Pre-cycle live sync/reconcile failed: %s", exc)
            logging.info("--- Starting Research + Paper-Trading Evaluation Cycle ---")

            # 1. Gather public market context + public wallet activity
            open_markets = fetch_btc_markets(closed=False)
            closed_markets = fetch_btc_markets(closed=True)
            if not open_markets.empty and not closed_markets.empty:
                markets_df = pd.concat([open_markets, closed_markets], ignore_index=True).drop_duplicates(subset=["market_id"])
            else:
                markets_df = open_markets if not open_markets.empty else closed_markets
            autonomous_monitor.write_heartbeat("market_monitor", status="ok", message="markets_fetched", extra={"rows": len(markets_df) if markets_df is not None else 0})
            if markets_df is not None and not markets_df.empty: markets_df = markets_df.loc[:, ~markets_df.columns.duplicated()]
            save_market_snapshot(markets_df)
            if always_on_enabled and always_on_only:
                logging.info(
                    "ALWAYS_ON_ONLY active for slug=%s: skipping wallet scraper and using pinned market signal path.",
                    always_on_slug,
                )
                signals_df = pd.DataFrame()
            else:
                signals_df = run_scraper_cycle()
                
                # O-Flow Analysis: Generate synthetic signals from raw volume + liquidity
                order_flow_signals_df = order_flow_analyzer.analyze(signals_df, markets_df)
                if not order_flow_signals_df.empty:
                    if signals_df is not None and not signals_df.empty:
                        signals_df = pd.concat([signals_df, order_flow_signals_df], ignore_index=True)
                    else:
                        signals_df = order_flow_signals_df
                        
                signals_df = _dedupe_signals_df(signals_df)
            signals_df, markets_df = _inject_always_on_signal(signals_df, markets_df)
            signals_df = _dedupe_signals_df(signals_df)
            cycle_observed_iso = datetime.now(timezone.utc).isoformat()
            signals_df = _annotate_signal_freshness(signals_df, cycle_observed_iso)
            
            # --- PILLARS 1-4: The Unified Macro Footprint ---
            ta_context = technical_analyzer.analyze()
            sent_context = sentiment_analyzer.analyze()
            mach_context = macro_analyzer.analyze()
            onc_context = onchain_analyzer.analyze()
            
            macro_context = {**ta_context, **sent_context, **mach_context, **onc_context}
            if not signals_df.empty:
                for k, v in macro_context.items():
                    signals_df[k] = v
            # --------------------------------------------------
            
            if always_on_enabled and always_on_only and signals_df is not None and not signals_df.empty:
                # Keep ALWAYS_ON_ONLY robust when rotating BTC 5m slugs change every round.
                # Prefer synthetic pinned signals; fall back to slug filtering only if needed.
                if "signal_source" in signals_df.columns:
                    src = signals_df["signal_source"].astype(str).str.lower()
                    pinned = signals_df[src == "always_on_market"]
                    if not pinned.empty:
                        signals_df = pinned.reset_index(drop=True)
                if not signals_df.empty and "market_slug" in signals_df.columns:
                    configured = str(always_on_slug or "").strip().lower()
                    if configured.startswith("btc-updown-5m-"):
                        mask = signals_df["market_slug"].astype(str).str.lower().str.startswith("btc-updown-5m-")
                    else:
                        mask = signals_df["market_slug"].astype(str).str.lower() == configured
                    signals_df = signals_df[mask].reset_index(drop=True)
            autonomous_monitor.write_heartbeat("signal_engine", status="ok", message="signals_scraped", extra={"rows": len(signals_df) if signals_df is not None else 0})

            if signals_df is not None and not signals_df.empty and "market_slug" in signals_df.columns:
                scraped_slugs = {
                    str(slug).strip().lower()
                    for slug in signals_df["market_slug"].dropna().astype(str).unique()
                    if str(slug).strip()
                }
                scraped_slugs.discard("")
                known_slugs = (
                    {
                        str(slug).strip().lower()
                        for slug in markets_df["slug"].dropna().astype(str).unique()
                        if str(slug).strip()
                    }
                    if markets_df is not None and not markets_df.empty and "slug" in markets_df.columns
                    else set()
                )
                missing_slugs = scraped_slugs - known_slugs
                if missing_slugs:
                    logging.info("Universe Gap: %s slugs missing. Synchronizing...", len(missing_slugs))
                    missing_df = fetch_markets_by_slugs(sorted(missing_slugs))
                    if missing_df is not None and not missing_df.empty:
                        markets_df = pd.concat([markets_df, missing_df], ignore_index=True).drop_duplicates(subset=["slug"])
                        if markets_df is not None and not markets_df.empty: markets_df = markets_df.loc[:, ~markets_df.columns.duplicated()]
                        recovered_slugs = {
                            str(slug).strip().lower()
                            for slug in missing_df["slug"].dropna().astype(str).unique()
                            if str(slug).strip()
                        } if "slug" in missing_df.columns else set()
                        logging.info(
                            "Universe sync recovered %s/%s missing slugs.",
                            len(recovered_slugs),
                            len(missing_slugs),
                        )
                    else:
                        logging.warning("Universe sync could not recover any of %s missing slugs.", len(missing_slugs))
            save_market_snapshot(markets_df)

            if signals_df.empty:
                logging.info("No actionable signals found. Checking active positions...")
                scored_df = pd.DataFrame()  # Fallthrough to exits
            # 2. Build whale summaries and detect alerts from public data
            whale_tracker.write_summary(signals_df)
            autonomous_monitor.write_heartbeat("whale_tracker", status="ok", message="whale_summary_written", extra={"rows": len(signals_df) if signals_df is not None else 0})
            alerts_engine.process_alerts(markets_df, previous_markets_df, signals_df)
            autonomous_monitor.write_heartbeat("alerts_engine", status="ok", message="alerts_processed")
            previous_markets_df = markets_df.copy()

            # 3. Build features, run supervised inference, and score paper-trading opportunities
            if signals_df is not None and not signals_df.empty: signals_df = signals_df.loc[:, ~signals_df.columns.duplicated()]
            features_df = feature_builder.build_features(signals_df, markets_df)
            if features_df is not None: features_df = features_df.loc[:, ~features_df.columns.duplicated()].copy()
            if features_df is not None and not features_df.empty: features_df = features_df.loc[:, ~features_df.columns.duplicated()]
            log_raw_candidates(features_df)
            # LIVE-TRADE FIX: strict_inference_mode defaults to false so missing/stale
            # model artifacts no longer freeze live entries. The bot will still log
            # missing artifacts as warnings but will NOT halt trading.
            strict_inference_mode = os.getenv("STRICT_INFERENCE_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
            if trading_mode == "live" and strict_inference_mode:
                missing_model_artifacts = []
                for stage_name, inference_obj in (
                    ("model_inference", model_inference),
                    ("stage1_inference", stage1_inference),
                    ("stage2_temporal_inference", stage2_inference),
                ):
                    checker = getattr(inference_obj, "missing_artifacts", None)
                    if not callable(checker):
                        continue
                    try:
                        for item in checker():
                            missing_model_artifacts.append(
                                {
                                    "stage": stage_name,
                                    "component": str(item.get("component") or "unknown"),
                                    "path": str(item.get("path") or ""),
                                }
                            )
                    except Exception as exc:
                        logging.warning("Strict inference artifact check failed for %s: %s", stage_name, exc)
                if missing_model_artifacts:
                    pre_cycle_entry_freeze = True
                    pre_cycle_freeze_reason = "inference_model_missing"
                    pre_cycle_freeze_detail = {
                        "missing_artifact_count": len(missing_model_artifacts),
                        "missing_artifacts": missing_model_artifacts,
                    }
                    logging.error(
                        "STRICT INFERENCE MODE: Missing inference model artifacts detected. Freezing live entries this cycle: %s",
                        pre_cycle_freeze_detail,
                    )
                    autonomous_monitor.write_heartbeat(
                        "inference",
                        status="error",
                        message="strict_inference_missing_models",
                        extra=pre_cycle_freeze_detail,
                    )
            elif trading_mode == "live" and not strict_inference_mode:
                # Warn about missing artifacts but don't freeze entries
                for stage_name, inference_obj in (
                    ("model_inference", model_inference),
                    ("stage1_inference", stage1_inference),
                    ("stage2_temporal_inference", stage2_inference),
                ):
                    checker = getattr(inference_obj, "missing_artifacts", None)
                    if callable(checker):
                        try:
                            missing = list(checker())
                            if missing:
                                logging.warning(
                                    "[%s] Missing model artifacts (non-blocking): %s",
                                    stage_name,
                                    [str(m.get("path") or m.get("component")) for m in missing],
                                )
                        except Exception:
                            pass
            inferred_df = model_inference.run(features_df)
            inferred_df = stage1_inference.run(inferred_df)
            inferred_df = stage2_inference.run(inferred_df)
            if trading_mode == "live" and strict_inference_mode and _inference_guard_has_errors():
                inference_errors = _inference_guard_get_errors(limit=20)
                pre_cycle_entry_freeze = True
                runtime_error_detail = {
                    "error_count": len(inference_errors),
                    "errors": [
                        {
                            "stage": e.get("stage"),
                            "error_type": e.get("error_type"),
                            "message": e.get("message"),
                            "context": e.get("context"),
                        }
                        for e in inference_errors
                    ],
                }
                if pre_cycle_freeze_reason == "inference_model_missing":
                    merged_detail = dict(pre_cycle_freeze_detail or {})
                    merged_detail["runtime_error_count"] = runtime_error_detail["error_count"]
                    merged_detail["runtime_errors"] = runtime_error_detail["errors"]
                    pre_cycle_freeze_reason = "inference_model_missing_and_runtime_exception"
                    pre_cycle_freeze_detail = merged_detail
                else:
                    pre_cycle_freeze_reason = "inference_runtime_exception"
                    pre_cycle_freeze_detail = runtime_error_detail
                logging.error(
                    "STRICT INFERENCE MODE: Runtime inference exceptions detected. Freezing live entries this cycle: %s",
                    pre_cycle_freeze_detail,
                )
                autonomous_monitor.write_heartbeat(
                    "inference",
                    status="error",
                    message="strict_inference_freeze",
                    extra=pre_cycle_freeze_detail,
                )
            if "temporal_p_tp_before_sl" in inferred_df.columns:
                w_stage1 = float(np.clip(getattr(TradingConfig, "STAGE1_BLEND_WEIGHT", 0.65), 0.0, 1.0))
                w_stage2 = 1.0 - w_stage1
                inferred_df["p_tp_before_sl"] = (
                    inferred_df["p_tp_before_sl"].astype(float).fillna(0.0) * w_stage1
                    + inferred_df["temporal_p_tp_before_sl"].astype(float).fillna(0.0) * w_stage2
                ).clip(0.0, 1.0)
            if "expected_return" in inferred_df.columns:
                inferred_df["expected_return"] = clip_expected_return_series(inferred_df["expected_return"])
            if "temporal_expected_return" in inferred_df.columns:
                inferred_df["expected_return"] = clip_expected_return_series(
                    inferred_df[["expected_return", "temporal_expected_return"]].mean(axis=1)
                )
                inferred_df["edge_score"] = inferred_df["p_tp_before_sl"].astype(float) * inferred_df["expected_return"].astype(float)
            inferred_df = hybrid_scorer.run(inferred_df)
            if "hybrid_edge" in inferred_df.columns:
                inferred_df["edge_score"] = inferred_df["hybrid_edge"]
            log_raw_candidates(inferred_df)
            if inferred_df is not None and not inferred_df.empty: inferred_df = inferred_df.loc[:, ~inferred_df.columns.duplicated()]
            scored_df = signal_engine.score_features(inferred_df)
            scored_df = feedback_learner.apply_to_scored_df(scored_df, signal_engine)
            if scored_df is not None: scored_df = scored_df.loc[:, ~scored_df.columns.duplicated()].copy()
            if scored_df is not None and not scored_df.empty: scored_df = scored_df.loc[:, ~scored_df.columns.duplicated()]

            if shadow_purgatory is not None:
                try:
                    shadow_purgatory.resolve_purgatory()
                    autonomous_monitor.write_heartbeat("shadow_audit", status="ok", message="shadow_purgatory_resolved")
                except Exception as exc:
                    logging.warning("Shadow purgatory resolve failed: %s", exc)
                    autonomous_monitor.write_heartbeat("shadow_audit", status="warn", message="shadow_purgatory_resolve_failed", extra={"error": str(exc)})

            if shadow_purgatory is not None and not scored_df.empty:
                merge_keys = [c for c in ["token_id", "timestamp", "trader_wallet", "market_slug", "market_title"] if c in scored_df.columns and c in inferred_df.columns]
                if merge_keys:
                    shadow_candidates = scored_df.head(5).merge(inferred_df, on=merge_keys, how="left", suffixes=("", "_full"))
                else:
                    shadow_candidates = scored_df.head(5).copy()
                for _, row in shadow_candidates.iterrows():
                    try:
                        signal_payload = row.to_dict()
                        full_features = pd.DataFrame([row.to_dict()])
                        shadow_purgatory.log_intent(signal_payload, full_features)
                    except Exception as exc:
                        logging.warning("Shadow logging failed for %s: %s", row.get("market_title", row.get("market")), exc)

            if scored_df.empty:
                logging.info("No scored signals generated. Checking active positions...")
            sort_cols = [c for c in ["risk_adjusted_ev", "entry_ev", "execution_quality_score", "edge_score", "p_tp_before_sl", "confidence", "normalized_trade_size"] if c in scored_df.columns]
            scored_df = scored_df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
            calibration_baseline = _load_calibration_baseline() if enable_calibration_gate else calibration_min_edge
            open_token_ids, open_condition_ids, open_market_slugs = _extract_open_market_universe(markets_df)
            if open_market_slugs:
                logging.info(
                    "Open market universe this cycle: markets=%d tokens=%d",
                    len(open_market_slugs),
                    len(open_token_ids),
                )
            else:
                logging.warning("Open market universe is empty this cycle; skipping candidate entries.")

            # Suppress duplicate entries / repeated token spam
            if "token_id" in scored_df.columns:
                # Keep top 3 per token (was top 1 — too aggressive)
                scored_df = scored_df.groupby("token_id").head(3).reset_index(drop=True)

            # 3. Log top-ranked paper opportunities (research output, not betting advice)
            top_n = min(5, len(scored_df))
            logging.info("Top %s paper-trading opportunities this cycle:", top_n)
            print("\n=== TOP PAPER-TRADING OPPORTUNITIES ===")
            for rank, (_, ranked_row) in enumerate(scored_df.head(top_n).iterrows(), start=1):
                summary_line = (
                    f"{rank}. {ranked_row.get('signal_label')} | "
                    f"confidence={_safe_float(ranked_row.get('confidence', PredictionLayer.select_signal_score(ranked_row)), default=0.0):.2f} | "
                    f"market={ranked_row.get('market_title')} | "
                    f"side={ranked_row.get('side')}"
                )
                logging.info(" - %s", summary_line)
                print(summary_line)
                print(f"   reason: {ranked_row.get('reason')}")
                log_ranked_signal(ranked_row.to_dict())
            print("======================================\n")

            last_entry_age_minutes = minutes_since_last_entry(db)
            cadence_boost_active = (
                last_entry_age_minutes is not None
                and target_entry_interval_minutes > 0
                and last_entry_age_minutes >= target_entry_interval_minutes
            )
            if cadence_boost_active:
                logging.info(
                    "Trade cadence booster active: last entry %.1f minutes ago (target %.1f minutes). Relaxing top %d candidate gates this cycle.",
                    last_entry_age_minutes,
                    target_entry_interval_minutes,
                    entry_aggression_top_k,
                )

            # 4A. Candidate-entry path for new signals
            current_active_trades = []
            active_trade_keys = set()
            current_open_exposure = 0.0

            def _refresh_local_active_trade_state():
                nonlocal current_active_trades, active_trade_keys, current_open_exposure
                current_active_trades = [
                    trade
                    for trade in trade_manager.active_trades.values()
                    if getattr(trade, "state", None) != TradeState.CLOSED
                ]
                active_trade_keys = {
                    _trade_key_from_signal({
                        "token_id": trade.token_id,
                        "condition_id": trade.condition_id,
                        "market_title": trade.market,
                        "outcome_side": trade.outcome_side,
                    })
                    for trade in current_active_trades if trade.market or trade.token_id
                }
                current_open_exposure = sum(
                    float(getattr(trade, "size_usdc", 0) or 0)
                    for trade in current_active_trades
                )

            _refresh_local_active_trade_state()

            cached_entry_available_balance: float | None = None

            def _invalidate_entry_available_balance():
                nonlocal cached_entry_available_balance
                cached_entry_available_balance = None

            def _get_entry_available_balance(force_refresh: bool = False) -> float:
                nonlocal cached_entry_available_balance
                if (
                    not force_refresh
                    and cached_entry_available_balance is not None
                    and cached_entry_available_balance > 0
                ):
                    return float(cached_entry_available_balance)

                available_balance = 0.0
                if order_manager is not None:
                    try:
                        available_balance, _ = order_manager._get_available_balance(
                            asset_type="COLLATERAL",
                            use_onchain_fallback=False,
                        )
                    except Exception:
                        pass
                if available_balance <= 0 and execution_client is not None and trading_mode != "live":
                    try:
                        available_balance = execution_client.get_available_balance(asset_type="COLLATERAL")
                    except Exception:
                        pass
                if available_balance <= 0 and trading_mode == "paper":
                    try:
                        _sim_bal = float(os.getenv("SIMULATED_STARTING_BALANCE", "1000"))
                        available_balance = max(0.0, _sim_bal - current_open_exposure)
                    except Exception:
                        available_balance = 0.0

                if available_balance > 0:
                    cached_entry_available_balance = float(available_balance)
                return float(available_balance or 0.0)

            def _get_live_available_token_shares(token_id: str) -> float:
                if trading_mode != "live" or order_manager is None or not token_id:
                    return 0.0
                try:
                    readiness = order_manager.check_readiness(asset_type="CONDITIONAL", token_id=token_id)
                except Exception:
                    readiness = None
                raw_balance = None
                if isinstance(readiness, dict):
                    for key in ["balance", "available", "available_balance", "amount"]:
                        if readiness.get(key) is not None:
                            raw_balance = readiness[key]
                            break
                try:
                    normalized_balance = float(order_manager._normalize_balance(raw_balance))
                except Exception:
                    try:
                        normalized_balance = float(raw_balance or 0.0)
                    except Exception:
                        normalized_balance = 0.0
                try:
                    return float(order_manager._round_down_shares(max(0.0, normalized_balance)))
                except Exception:
                    return float(max(0.0, normalized_balance))

            def _build_live_exit_price_ladder(token_id: str, fallback_price: float, aggressive: bool = False):
                tick_size = max(0.001, float(os.getenv("LIVE_EXIT_TICK_SIZE", "0.01") or 0.01))
                max_attempts = max(2, int(os.getenv("LIVE_EXIT_MAX_ATTEMPTS", "4") or 4))
                analysis = {}
                try:
                    if orderbook_guard is not None:
                        analysis = orderbook_guard.analyze_book(token_id, depth=max(5, max_attempts))
                except Exception:
                    analysis = {}

                ladder = []
                seen_prices = set()

                def _append_price(raw_price):
                    try:
                        price_val = round(float(raw_price), 4)
                    except Exception:
                        return
                    if not (0.009 < price_val < 0.991):
                        return
                    if price_val in seen_prices:
                        return
                    seen_prices.add(price_val)
                    ladder.append(price_val)

                for level in (analysis.get("top_bids") or []):
                    _append_price(level.get("price"))

                best_bid = analysis.get("best_bid")
                if best_bid is not None:
                    for step in ([1, 2, 3, 4] if aggressive else [1, 2]):
                        _append_price(float(best_bid) - (tick_size * step))

                _append_price(fallback_price)
                if not ladder:
                    _append_price(quote_exit_price({"current_price": fallback_price}, slippage=0.03 if aggressive else 0.01))

                return ladder[:max_attempts], analysis

            def _execute_live_sell_ladder(token_id: str, requested_shares: float, condition_id=None, outcome_side=None, reference_price: float = 0.0, close_reason: str = "rl_exit"):
                aggressive_reasons = {
                    "stop_loss",
                    "trailing_stop",
                    "time_stop",
                    "trajectory_panic_exit",
                    "trajectory_reversal_exit",
                    "ai_close_long",
                }
                aggressive_exit = str(close_reason or "").strip().lower() in aggressive_reasons
                fallback_price = max(0.01, min(0.99, float(reference_price or 0.0)))
                initial_available_shares = _get_live_available_token_shares(token_id)
                requested_shares = max(0.0, float(requested_shares or 0.0))
                remaining_target_shares = min(requested_shares, initial_available_shares)
                if remaining_target_shares <= 0:
                    return {
                        "status": "no_inventory",
                        "filled_shares": 0.0,
                        "avg_price": fallback_price,
                        "remaining_exchange_shares": initial_available_shares,
                        "attempts": [],
                        "analysis": {},
                    }

                per_attempt_timeout = max(2.0, float(os.getenv("LIVE_EXIT_ATTEMPT_TIMEOUT_SECONDS", "6") or 6))
                ladder_prices, analysis = _build_live_exit_price_ladder(token_id, fallback_price=fallback_price, aggressive=aggressive_exit)
                attempts = []
                filled_total_shares = 0.0
                filled_total_notional = 0.0
                available_before = initial_available_shares

                for attempt_index, attempt_price in enumerate(ladder_prices, start=1):
                    remaining_target_shares = min(max(requested_shares - filled_total_shares, 0.0), available_before)
                    if remaining_target_shares <= 1e-6:
                        break

                    exit_row, exit_response = order_manager.submit_entry(
                        token_id=token_id,
                        price=attempt_price,
                        size=remaining_target_shares,
                        side="SELL",
                        condition_id=condition_id,
                        outcome_side=outcome_side,
                        order_type="GTC",
                        post_only=False,
                        execution_style="taker",
                        bypass_risk_checks=True,
                    )
                    exit_order_id = (exit_row or {}).get("order_id") or (exit_response or {}).get("orderID") or (exit_response or {}).get("order_id") or (exit_response or {}).get("id")
                    attempt_record = {
                        "attempt": attempt_index,
                        "price": attempt_price,
                        "requested_shares": round(remaining_target_shares, 6),
                        "order_id": exit_order_id,
                    }

                    if not exit_order_id:
                        attempt_record["result"] = "rejected"
                        attempt_record["reason"] = (exit_row or {}).get("reason") or (exit_response or {}).get("reason")
                        attempts.append(attempt_record)
                        continue

                    fill_result = order_manager.wait_for_fill(exit_order_id, timeout_seconds=per_attempt_timeout, poll_seconds=1)
                    if not fill_result.get("filled"):
                        logging.warning(
                            "Live SELL attempt %s/%s not fully filled for %s; attempting cancel for order_id=%s at %.4f",
                            attempt_index,
                            len(ladder_prices),
                            token_id,
                            exit_order_id,
                            attempt_price,
                        )
                        try:
                            order_manager.cancel_stale_order(exit_order_id)
                        except Exception as exc:
                            logging.warning("Failed to cancel stale live exit order %s: %s", exit_order_id, exc)

                    fill_payload = fill_result.get("response") or {}
                    available_after = _get_live_available_token_shares(token_id)
                    actual_filled_shares = max(0.0, available_before - available_after)
                    reported_fill_size = 0.0
                    try:
                        reported_fill_size = float(fill_payload.get("size", 0.0) or 0.0)
                    except Exception:
                        reported_fill_size = 0.0
                    if actual_filled_shares <= 1e-6 and fill_result.get("filled") and reported_fill_size > 0:
                        actual_filled_shares = min(reported_fill_size, remaining_target_shares)
                        available_after = max(0.0, available_before - actual_filled_shares)

                    if actual_filled_shares > 1e-6:
                        try:
                            actual_fill_price = float(fill_payload.get("price", attempt_price) or attempt_price)
                        except Exception:
                            actual_fill_price = float(attempt_price)
                        filled_total_shares += actual_filled_shares
                        filled_total_notional += actual_filled_shares * actual_fill_price
                        attempt_record["result"] = "filled" if fill_result.get("filled") else "partial_after_timeout"
                        attempt_record["filled_shares"] = round(actual_filled_shares, 6)
                        attempt_record["fill_price"] = round(actual_fill_price, 6)
                    else:
                        attempt_record["result"] = "unfilled"

                    attempts.append(attempt_record)
                    available_before = available_after

                remaining_exchange_shares = _get_live_available_token_shares(token_id)
                avg_fill_price = (filled_total_notional / filled_total_shares) if filled_total_shares > 0 else fallback_price
                return {
                    "status": "filled" if filled_total_shares >= (requested_shares - 1e-6) else ("partial" if filled_total_shares > 0 else "unfilled"),
                    "filled_shares": float(filled_total_shares),
                    "avg_price": float(avg_fill_price),
                    "remaining_exchange_shares": float(remaining_exchange_shares),
                    "attempts": attempts,
                    "analysis": analysis,
                }
            live_entry_freeze = pre_cycle_entry_freeze
            freeze_reason = pre_cycle_freeze_reason
            freeze_detail = pre_cycle_freeze_detail or {}
            if trading_mode == "live" and live_position_book is not None:
                live_position_book.rebuild_from_db()
                reconciled_positions_df = live_position_book.get_enriched_open_positions(scored_df=scored_df)
                if mismatch_detector is not None:
                    mismatch_summary = mismatch_detector.detect(current_active_trades, reconciled_positions_df)
                    if mismatch_summary.get("freeze_entries"):
                        session_reconciliation_mismatch_count += 1
                        if (
                            session_reconciliation_mismatch_count >= max_session_reconciliation_mismatches
                            and not session_kill_switch_active
                        ):
                            session_kill_switch_active = True
                            session_kill_switch_reason = (
                                "session_reconciliation_mismatch_limit_hit "
                                f"({session_reconciliation_mismatch_count} >= {max_session_reconciliation_mismatches})"
                            )
                        live_entry_freeze = True
                        freeze_reason = mismatch_summary.get("source") or "state_mismatch"
                        freeze_detail = mismatch_summary.get("detail", {}) or {}
                        mismatch_detector.record(mismatch_summary)
                        logging.error(
                            "Live entry freeze active (reason=%s, local_count=%s, live_count=%s, local_only=%s, live_only=%s)",
                            freeze_reason,
                            freeze_detail.get("local_count"),
                            freeze_detail.get("live_count"),
                            freeze_detail.get("local_only"),
                            freeze_detail.get("live_only"),
                        )
                        autonomous_monitor.write_heartbeat("reconciliation", status="error", message="entry_freeze_state_mismatch", extra=freeze_detail)
                    elif previous_entry_freeze_active:
                        session_reconciliation_mismatch_count = 0
                        logging.info("Entry freeze cleared (previous_reason=%s)", previous_entry_freeze_reason or "state_mismatch")

            previous_entry_freeze_active = live_entry_freeze
            previous_entry_freeze_reason = freeze_reason

            _max_pos = getattr(TradingConfig, 'MAX_CONCURRENT_POSITIONS', 5)
            _candidate_skip_logging = os.getenv("ENABLE_CANDIDATE_SKIP_LOGGING", "true").strip().lower() in {"1", "true", "yes", "on"}
            _candidate_skip_counts = {}
            _candidate_stats = {
                "candidates_seen": 0,
                "candidates_tradable": 0,
                "candidates_rejected": 0,
                "entries_sent": 0,
                "fills_received": 0,
            }
            _candidate_decision_rows = []
            # Seed per-cycle unavailable set from persistent cache (skip API calls for known-404 tokens)
            import time as _time
            _now_mono = _time.monotonic()
            _ob_no_book_cache = {k: v for k, v in _ob_no_book_cache.items() if _now_mono - v < _OB_NO_BOOK_TTL}
            _orderbook_unavailable_tokens = set(_ob_no_book_cache.keys())

            def _record_candidate_decision(
                signal_row: dict,
                *,
                final_decision: str,
                reject_reason: str | None = None,
                gate: str | None = None,
                model_action: str | None = None,
                proposed_size_usdc: float | None = None,
                final_size_usdc: float | None = None,
                available_balance: float | None = None,
                order_id: str | None = None,
                precomputed_calibrated_edge: float | None = None,
                **extra,
            ):
                nonlocal session_failed_entries, session_kill_switch_active, session_kill_switch_reason

                reject_reason_norm = str(reject_reason or "").strip().lower() or None
                reject_category = _reject_category(reject_reason_norm) if reject_reason_norm else None
                # FIX BUG#2: use caller-supplied calibrated_edge when available to
                # avoid a redundant _compute_calibrated_edge() call.
                calibrated_edge = (
                    precomputed_calibrated_edge
                    if precomputed_calibrated_edge is not None
                    else _compute_calibrated_edge(signal_row)
                )
                payload = {
                    "cycle_id": cycle_id,
                    "candidate_id": f"{cycle_id}:{len(_candidate_decision_rows) + 1}",
                    "token_id": normalize_token_id(signal_row.get("token_id")),
                    "condition_id": signal_row.get("condition_id"),
                    "outcome_side": signal_row.get("outcome_side", signal_row.get("side")),
                    "market": signal_row.get("market_title", signal_row.get("market")),
                    "market_slug": signal_row.get("market_slug"),
                    "trader_wallet": signal_row.get("trader_wallet", signal_row.get("wallet_copied")),
                    "entry_intent": signal_row.get("entry_intent"),
                    "model_action": model_action,
                    "final_decision": str(final_decision or "SKIPPED"),
                    "reject_reason": reject_reason_norm,
                    "reject_category": reject_category,
                    "gate": gate,
                    "confidence": _safe_float(signal_row.get("confidence", 0.0), default=0.0),
                    "p_tp_before_sl": _safe_float(signal_row.get("p_tp_before_sl", 0.0), default=0.0),
                    "expected_return": _safe_float(signal_row.get("expected_return", 0.0), default=0.0),
                    "edge_score": _safe_float(signal_row.get("edge_score", 0.0), default=0.0),
                    "calibrated_edge": calibrated_edge,
                    "calibrated_baseline": calibration_baseline,
                    "proposed_size_usdc": _safe_float(proposed_size_usdc, default=0.0) if proposed_size_usdc is not None else None,
                    "final_size_usdc": _safe_float(final_size_usdc, default=0.0) if final_size_usdc is not None else None,
                    "available_balance": _safe_float(available_balance, default=0.0) if available_balance is not None else None,
                    "order_id": str(order_id) if order_id is not None else None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                details_json = json.dumps(extra or {}, default=str, separators=(",", ":"))
                try:
                    db.execute(
                        """
                        INSERT INTO candidate_decisions (
                            cycle_id, candidate_id, token_id, condition_id, outcome_side, market,
                            market_slug, trader_wallet, entry_intent, model_action, final_decision,
                            reject_reason, reject_category, gate, confidence, p_tp_before_sl,
                            expected_return, edge_score, calibrated_edge, calibrated_baseline,
                            proposed_size_usdc, final_size_usdc, available_balance, order_id, details_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            payload["cycle_id"],
                            payload["candidate_id"],
                            payload["token_id"],
                            payload["condition_id"],
                            payload["outcome_side"],
                            payload["market"],
                            payload["market_slug"],
                            payload["trader_wallet"],
                            payload["entry_intent"],
                            payload["model_action"],
                            payload["final_decision"],
                            payload["reject_reason"],
                            payload["reject_category"],
                            payload["gate"],
                            payload["confidence"],
                            payload["p_tp_before_sl"],
                            payload["expected_return"],
                            payload["edge_score"],
                            payload["calibrated_edge"],
                            payload["calibrated_baseline"],
                            payload["proposed_size_usdc"],
                            payload["final_size_usdc"],
                            payload["available_balance"],
                            payload["order_id"],
                            details_json,
                            payload["created_at"],
                        ),
                    )
                except Exception as exc:
                    logging.warning("Candidate decision DB logging failed: %s", exc)

                csv_payload = dict(payload)
                csv_payload["details_json"] = details_json
                _candidate_decision_rows.append(csv_payload)

                if reject_reason_norm:
                    _candidate_skip_counts[reject_reason_norm] = _candidate_skip_counts.get(reject_reason_norm, 0) + 1
                    _candidate_stats["candidates_rejected"] += 1
                    if reject_category == "no_liquidity":
                        _candidate_stats.setdefault("rejected_no_liquidity", 0)
                        _candidate_stats["rejected_no_liquidity"] += 1
                if payload["order_id"]:
                    _candidate_stats["entries_sent"] += 1
                if payload["final_decision"] == "ENTRY_FILLED":
                    _candidate_stats["fills_received"] += 1

                if reject_reason_norm in {
                    "live_entry_rejected_or_missing_order_id",
                    "live_entry_unfilled_after_cancel",
                    "live_entry_submit_exception",
                    "live_wait_for_fill_exception",
                }:
                    session_failed_entries += 1
                    if session_failed_entries >= max_session_failed_entries and not session_kill_switch_active:
                        session_kill_switch_active = True
                        session_kill_switch_reason = (
                            "session_failed_entry_limit_hit "
                            f"({session_failed_entries} >= {max_session_failed_entries})"
                        )

                if _candidate_skip_logging and payload["final_decision"] == "REJECTED":
                    try:
                        logging.info("CANDIDATE_REJECT %s", json.dumps(payload, default=str, separators=(",", ":")))
                    except Exception:
                        logging.info(
                            "CANDIDATE_REJECT reason=%s token=%s market=%s",
                            reject_reason_norm,
                            payload.get("token_id"),
                            payload.get("market"),
                        )

                return payload

            def _log_candidate_skip(signal_row: dict, reason: str, gate: str | None = None, **extra):
                payload = _record_candidate_decision(
                    signal_row,
                    final_decision="SKIPPED",
                    reject_reason=reason,
                    gate=gate or "entry_gate",
                    **extra,
                )
                if not _candidate_skip_logging:
                    return
                try:
                    logging.info("CANDIDATE_SKIP %s", json.dumps(payload, default=str, separators=(",", ":")))
                except Exception:
                    logging.info("CANDIDATE_SKIP reason=%s token=%s market=%s", reason, payload.get("token_id"), payload.get("market"))
            
            # FIX 1A: Process all AI exits FIRST, ignoring max_pos restrictions
            for _, row in scored_df.iterrows():
                s_row = row.to_dict()
                if str(s_row.get("entry_intent", "")).upper() == "CLOSE_LONG":
                    m_key = _trade_key_from_signal(s_row)
                    if m_key in trade_manager.active_trades:
                        logging.warning("AI Veto! CLOSE_LONG received for %s. Forcing exit.", m_key)
                        _trade = trade_manager.active_trades[m_key]
                        _px = float(getattr(_trade, "current_price", 0.0) or getattr(_trade, "entry_price", 0.0) or 0.0)
                        if _px > 0:
                            _trade.close(exit_price=_px, reason="ai_close_long")
                        else:
                            _trade.state = TradeState.CLOSED
                            _trade.close_reason = "ai_close_long"

            # FIX 1B: Normal entry loop
            for candidate_rank, (_, row) in enumerate(scored_df.iterrows(), start=1):
                signal_row = row.to_dict()
                if cadence_boost_active and candidate_rank <= entry_aggression_top_k:
                    signal_row = apply_entry_cadence_boost(
                        signal_row,
                        last_entry_age_minutes,
                        target_entry_interval_minutes,
                        candidate_rank,
                    )
                _candidate_stats["candidates_seen"] += 1
                token_id_norm = normalize_token_id(signal_row.get("token_id"))
                token_id = str(token_id_norm or "")
                entry_intent = str(signal_row.get("entry_intent", "OPEN_LONG") or "OPEN_LONG").upper()
                market_key = _trade_key_from_signal(signal_row)

                market_age_sec = _signal_age_seconds(signal_row)
                if market_age_sec is None and entry_require_market_timestamp:
                    _log_candidate_skip(signal_row, "missing_market_timestamp", gate="freshness")
                    continue
                effective_entry_max_market_staleness_sec = entry_max_market_staleness_sec
                if trading_mode == "live":
                    # Current cycles can run long when the bot is reconciling positions,
                    # checking balances, and walking many candidates. Avoid aging out
                    # otherwise-current signals purely because this cycle is still busy.
                    effective_entry_max_market_staleness_sec = max(
                        effective_entry_max_market_staleness_sec,
                        entry_cycle_staleness_grace_sec,
                    )
                if market_age_sec is not None and market_age_sec > effective_entry_max_market_staleness_sec:
                    _log_candidate_skip(
                        signal_row,
                        "stale_market_data",
                        gate="freshness",
                        market_age_seconds=round(market_age_sec, 3),
                        max_age_seconds=effective_entry_max_market_staleness_sec,
                    )
                    continue

                if len(current_active_trades) >= _max_pos:
                    _log_candidate_skip(
                        signal_row,
                        "max_concurrent_positions_reached",
                        gate="capacity",
                        active_positions=len(current_active_trades),
                        max_positions=_max_pos,
                    )
                    continue
                
                if not market_key or not token_id or entry_intent == "CLOSE_LONG":
                    _log_candidate_skip(
                        signal_row,
                        "invalid_candidate_identity_or_intent",
                        gate="identity",
                        has_market_key=bool(market_key),
                        has_token_id=bool(token_id),
                        entry_intent=entry_intent,
                    )
                    continue

                signal_condition_id = str(signal_row.get("condition_id") or "").strip().lower()
                signal_slug = str(signal_row.get("market_slug") or "").strip().lower()
                if signal_slug and _is_expired_updown_slug(signal_slug):
                    _log_candidate_skip(
                        signal_row,
                        "expired_market_slug",
                        gate="freshness",
                        market_slug=signal_slug,
                    )
                    continue
                if not open_market_slugs:
                    _log_candidate_skip(
                        signal_row,
                        "no_open_market_universe",
                        gate="market_universe",
                    )
                    continue
                signal_has_token = bool(token_id)
                if signal_has_token:
                    # Token ID is present: require it to be explicitly in the open set.
                    # Do NOT allow slug/condition_id to rescue a dead token — that's the
                    # leak that lets resolved-market tokens reach the orderbook guard.
                    in_open_universe = token_id in open_token_ids
                    if not in_open_universe:
                        # Fallback: check if condition_id or slug is open AND the token_id
                        # is one of that market's clob_token_ids in markets_df.
                        # This handles the (rare) case where the scraper saw an older snapshot
                        # and the token_id was not yet added to open_token_ids this cycle.
                        slug_or_cond_open = (
                            (signal_condition_id and signal_condition_id in open_condition_ids)
                            or (signal_slug and signal_slug in open_market_slugs)
                        )
                        if slug_or_cond_open:
                            # Accept it — the parent market is open, likely a snapshot lag.
                            in_open_universe = True
                            logging.debug(
                                "Universe fallback: token_id %s not in open_token_ids but "
                                "slug/condition_id is open (snapshot lag?) — allowing",
                                token_id[:16] if len(token_id) > 16 else token_id,
                            )
                        else:
                            logging.debug(
                                "Universe gap: token_id %s has no open slug/condition_id either "
                                "— dead market candidate",
                                token_id[:16] if len(token_id) > 16 else token_id,
                            )
                else:
                    # No token_id on signal: fall back to slug/condition_id only.
                    in_open_universe = (
                        (signal_condition_id and signal_condition_id in open_condition_ids)
                        or (signal_slug and signal_slug in open_market_slugs)
                    )
                if not in_open_universe:
                    _log_candidate_skip(
                        signal_row,
                        "token_not_in_open_market_universe",
                        gate="market_universe",
                        open_market_count=len(open_market_slugs),
                        open_token_count=len(open_token_ids),
                    )
                    continue

                # FIX: Check dynamic active_trades to prevent Triple-Buy duplicates in the same loop
                if market_key in active_trade_keys:
                    logging.info("Prevented duplicate entry: Trade already open for %s.", market_key)
                    _log_candidate_skip(signal_row, "duplicate_active_trade", gate="dedupe", market_key=market_key)
                    continue
                if token_id in _orderbook_unavailable_tokens:
                    _log_candidate_skip(
                        signal_row,
                        "orderbook_not_available_cached",
                        gate="liquidity",
                        token_id_cached=True,
                    )
                    continue
                if trading_mode == "live" and live_entry_freeze:
                    logging.warning(
                        "Skipping new live entry for %s because entry freeze is active (reason=%s, local_count=%s, live_count=%s)",
                        token_id,
                        freeze_reason or "state_mismatch",
                        freeze_detail.get("local_count"),
                        freeze_detail.get("live_count"),
                    )
                    _log_candidate_skip(
                        signal_row,
                        "freeze",
                        gate="freeze",
                        freeze_reason=freeze_reason or "state_mismatch",
                        local_count=freeze_detail.get("local_count"),
                        live_count=freeze_detail.get("live_count"),
                    )
                    continue
                if session_kill_switch_active:
                    _log_candidate_skip(
                        signal_row,
                        "freeze",
                        gate="kill_switch",
                        freeze_reason=session_kill_switch_reason or "session_limit_hit",
                    )
                    continue

                # FIX BUG#1: evaluate rule once here and pass the result into
                # choose_action so it does NOT call evaluate() a second time.
                rule_eval = entry_rule.evaluate(signal_row) if hasattr(entry_rule, "evaluate") else None
                rule_allows_entry = bool(rule_eval["allow"]) if isinstance(rule_eval, dict) else entry_rule.should_enter(signal_row)
                _action_meta = {}
                action_val = choose_action(
                    signal_row,
                    entry_rule,
                    entry_brain=entry_brain,
                    legacy_brain=legacy_brain,
                    decision_meta=_action_meta,
                    precomputed_rule_eval=rule_eval,
                    precomputed_rule_allows=rule_allows_entry,
                )
                if action_val not in [0, 1, 2]:
                    action_val = 0

                # FIX BUG#2 + BUG#4: compute calibrated_edge once here so it can
                # be reused by both the skip log and _record_candidate_decision
                # without a second call.  Also do a cheap opportunistic balance
                # fetch so that IGNORE/veto skip records include available_balance.
                _pre_calibrated_edge = _compute_calibrated_edge(signal_row)
                _pre_available_bal: float | None = None
                if action_val == 0:
                    # Balance fetch for observability on IGNORE/veto paths only;
                    # the real fetch (with paper fallbacks) still happens in the
                    # action_val != 0 block so sizing logic is unchanged.
                    try:
                        _pre_available_bal = _get_entry_available_balance()
                    except Exception:
                        pass
                    if (_pre_available_bal is None or _pre_available_bal <= 0) and trading_mode == "paper":
                        try:
                            _sim_bal = float(os.getenv("SIMULATED_STARTING_BALANCE", "1000"))
                            _pre_available_bal = max(0.0, _sim_bal - current_open_exposure)
                        except Exception:
                            pass

                action_map = {0: "IGNORE", 1: "SMALL_BUY", 2: "LARGE_BUY"}
                try:
                    db.execute(
                        "INSERT INTO model_decisions (token_id, condition_id, outcome_side, model_name, score, action, feature_snapshot, model_artifact, normalization_artifact) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            token_id,
                            signal_row.get("condition_id"),
                            signal_row.get("outcome_side", signal_row.get("side")),
                            entry_model_name,
                            _safe_float(signal_row.get("confidence", 0.0), default=0.0),
                            action_map.get(action_val, "UNKNOWN"),
                            build_feature_snapshot(signal_row),
                            entry_model_artifact,
                            entry_norm_artifact,
                        ),
                    )
                except Exception as exc:
                    logging.warning("Model decision logging failed for %s: %s", token_id, exc)

                if action_val != 0:
                    # FIX BUG#2: reuse pre-computed calibrated_edge (no second call).
                    calibrated_edge = _pre_calibrated_edge
                    calibrated_required = max(calibration_min_edge, calibration_baseline + calibration_edge_margin)
                    if enable_calibration_gate and calibrated_edge < calibrated_required:
                        _log_candidate_skip(
                            signal_row,
                            "calibration_edge_below_baseline",
                            gate="calibration",
                            model_action=action_map.get(action_val, "UNKNOWN"),
                            calibrated_edge=round(calibrated_edge, 8),
                            calibrated_baseline=round(calibration_baseline, 8),
                            calibrated_required=round(calibrated_required, 8),
                        )
                        continue
                    confidence = _safe_float(signal_row.get("confidence", 0.0), default=0.0)

                    # ── Get balance (with paper mode fallback) ──
                    _available_bal = _get_entry_available_balance()
                    # ── BUGFIX: Paper mode needs simulated balance ──
                    if _available_bal <= 0 and trading_mode == "paper":
                        _sim_bal = float(os.getenv("SIMULATED_STARTING_BALANCE", "1000"))
                        _available_bal = max(0.0, _sim_bal - current_open_exposure)

                    _current_exposure = current_open_exposure
                    size_usdc = _money_mgr.calculate_bet_size(
                        available_balance=_available_bal,
                        confidence=confidence,
                        current_exposure=_current_exposure,
                    )
                    min_bet_usdc = float(getattr(TradingConfig, "MIN_BET_USDC", 1.0))
                    configured_min_entry = max(
                        min_bet_usdc,
                        float(getattr(TradingConfig, "MIN_BET_USDC", 1.0)),
                        float(getattr(TradingConfig, "MIN_ENTRY_USDC", getattr(TradingConfig, "MIN_BET_USDC", 1.0))),
                    )
                    reserve_pct = max(0.0, min(0.95, float(getattr(TradingConfig, "CAPITAL_RESERVE_PCT", 0.20))))
                    tradable_balance = max(0.0, _available_bal - (_available_bal * reserve_pct))
                    hard_max_bet = max(min_bet_usdc, float(getattr(TradingConfig, "HARD_MAX_BET_USDC", 250.0)))
                    risk_capped_max_entry = min(
                        tradable_balance * float(getattr(TradingConfig, "MAX_RISK_PER_TRADE_PCT", 0.15)),
                        hard_max_bet,
                    )
                    effective_min_entry = configured_min_entry
                    risk_capped_below_strategy_floor = (
                        risk_capped_max_entry + 1e-9 < configured_min_entry and size_usdc + 1e-9 >= min_bet_usdc
                    )
                    if risk_capped_below_strategy_floor:
                        # Small live accounts can otherwise deadlock forever here:
                        # MoneyManager correctly respects max-risk, but the static
                        # strategic floor rejects every entry. Fall back to the
                        # exchange minimum when the risk cap itself sits below the
                        # strategic anti-micro threshold.
                        effective_min_entry = min_bet_usdc
                    if size_usdc <= 0:
                        logging.info(
                            "MoneyManager: skip trade (balance=$%.2f, conf=%.2f, exposure=$%.2f)",
                            _available_bal, confidence, _current_exposure,
                        )
                        _log_candidate_skip(
                            signal_row,
                            "min_size",
                            gate="sizing",
                            model_action=action_map.get(action_val, "UNKNOWN"),
                            available_balance=round(_available_bal, 6),
                            current_exposure=round(_current_exposure, 6),
                        )
                        continue
                    if size_usdc + 1e-9 < effective_min_entry:
                        logging.info(
                            "Anti-micro guard: skipping tiny entry for %s (size=$%.2f < min_entry=$%.2f)",
                            token_id[:16],
                            size_usdc,
                            effective_min_entry,
                        )
                        _log_candidate_skip(
                            signal_row,
                            "anti_micro_min_entry",
                            gate="sizing",
                            model_action=action_map.get(action_val, "UNKNOWN"),
                            available_balance=round(_available_bal, 6),
                            current_exposure=round(_current_exposure, 6),
                            strategic_min_entry=round(configured_min_entry, 6),
                            effective_min_entry=round(effective_min_entry, 6),
                            risk_capped_max_entry=round(risk_capped_max_entry, 6),
                            proposed_size_usdc=round(size_usdc, 6),
                        )
                        continue
                    if risk_capped_below_strategy_floor and size_usdc + 1e-9 < configured_min_entry:
                        logging.info(
                            "Anti-micro guard: allowing risk-capped small-account entry for %s (size=$%.2f, configured_min=$%.2f, effective_min=$%.2f, risk_cap=$%.2f)",
                            token_id[:16],
                            size_usdc,
                            configured_min_entry,
                            effective_min_entry,
                            risk_capped_max_entry,
                        )
                    # ── Order book guard: check spread/depth before entry ──
                    try:
                        ob_check = orderbook_guard.check_before_entry(
                            token_id=token_id, side="BUY", intended_size_usdc=size_usdc,
                        )
                        if not ob_check["tradable"]:
                            logging.info("OrderBookGuard BLOCKED %s: %s", token_id[:16], ob_check["reason"])
                            ob_reason = str(ob_check.get("reason") or "").strip().lower()
                            if ob_reason == "orderbook_not_available":
                                _orderbook_unavailable_tokens.add(token_id)
                                _ob_no_book_cache[token_id] = _time.monotonic()  # persist across cycles
                            _log_candidate_skip(
                                signal_row,
                                "orderbook_not_available" if ob_reason == "orderbook_not_available" else "no_liquidity",
                                gate="liquidity",
                                model_action=action_map.get(action_val, "UNKNOWN"),
                                ob_reason=ob_reason,
                            )
                            continue
                        fill_price = ob_check.get("recommended_entry_price") or quote_entry_price(signal_row)
                        for _w in ob_check.get("warnings", []):
                            logging.warning("OrderBookGuard %s: %s", token_id[:16], _w)
                    except Exception as _ob_exc:
                        logging.warning("OrderBookGuard failed for %s: %s (using fallback price)", token_id[:16], _ob_exc)
                        fill_price = quote_entry_price(signal_row)
                    
                    if fill_price is None or pd.isna(fill_price):
                        logging.warning("Skipping signal with missing fill price for token_id=%s", token_id)
                        _log_candidate_skip(
                            signal_row,
                            "missing_fill_price",
                            gate="pricing",
                            model_action=action_map.get(action_val, "UNKNOWN"),
                        )
                        continue

                    _candidate_stats["candidates_tradable"] += 1


                    if trading_mode == "live" and order_manager is not None:
                        # For live mode: submit order first, register trade only on fill
                        from pnl_engine import PNLEngine as _PNLEngine
                        _order_shares = _PNLEngine.shares_from_capital(size_usdc, fill_price)
                        try:
                            if getattr(TradingConfig, "USE_MARKET_ORDERS", False):
                                entry_row, entry_response = order_manager.submit_market_entry(
                                    token_id=token_id,
                                    amount=size_usdc,
                                    side=signal_row.get("order_side", "BUY"),
                                    condition_id=signal_row.get("condition_id"),
                                    outcome_side=signal_row.get("outcome_side", signal_row.get("side")),
                                )
                            else:
                                entry_row, entry_response = order_manager.submit_entry(
                                    token_id=token_id,
                                    price=fill_price,
                                    size=size_usdc,
                                    side=signal_row.get("order_side", "BUY"),
                                    condition_id=signal_row.get("condition_id"),
                                    outcome_side=signal_row.get("outcome_side", signal_row.get("side")),
                                )
                        except Exception as exc:
                            logging.warning("Live entry submit exception for token_id=%s: %s", token_id, exc)
                            _record_candidate_decision(
                                signal_row,
                                final_decision="REJECTED",
                                reject_reason="live_entry_submit_exception",
                                gate="execution",
                                model_action=action_map.get(action_val, "UNKNOWN"),
                                proposed_size_usdc=size_usdc,
                                available_balance=_available_bal,
                                submit_error=str(exc),
                            )
                            continue
                        entry_order_id = (entry_row or {}).get("order_id") or (entry_response or {}).get("orderID") or (entry_response or {}).get("order_id") or (entry_response or {}).get("id")
                        if not entry_order_id:
                            logging.info("Live entry rejected/skipped for token_id=%s reason=%s", token_id, (entry_row or {}).get("reason"))
                            # If order is rejected/skipped, we should potentially revert trade creation in TradeManager
                            # For simplicity, we'll let process_exits handle cleanup later if trade is not filled
                            _record_candidate_decision(
                                signal_row,
                                final_decision="REJECTED",
                                reject_reason="live_entry_rejected_or_missing_order_id",
                                gate="execution",
                                model_action=action_map.get(action_val, "UNKNOWN"),
                                proposed_size_usdc=size_usdc,
                                available_balance=_available_bal,
                                exchange_reason=(entry_row or {}).get("reason"),
                            )
                            continue
                        
                        try:
                            fill_result = order_manager.wait_for_fill(entry_order_id)
                        except Exception as exc:
                            _record_candidate_decision(
                                signal_row,
                                final_decision="REJECTED",
                                reject_reason="live_wait_for_fill_exception",
                                gate="execution",
                                model_action=action_map.get(action_val, "UNKNOWN"),
                                proposed_size_usdc=size_usdc,
                                available_balance=_available_bal,
                                order_id=entry_order_id,
                                wait_error=str(exc),
                            )
                            continue
                        if not fill_result.get("filled"):
                            logging.info("Live entry not filled for token_id=%s; attempting cancel for order_id=%s", token_id, entry_order_id)
                            try:
                                cancel_resp = order_manager.cancel_stale_order(entry_order_id)
                                # FIX V5: If cancel fails because it ALREADY FILLED in the background, catch it!
                                if "already filled" in str(cancel_resp).lower() or "not found" in str(cancel_resp).lower():
                                    logging.warning("Order %s filled right after timeout! Fetching state...", entry_order_id)
                                    fill_result = order_manager.wait_for_fill(entry_order_id, timeout_seconds=2)
                            except Exception as exc:
                                logging.warning("Failed to cancel stale live entry order %s: %s", entry_order_id, exc)
                            
                            # If it STILL isn't filled after the race-condition check, safely skip
                            if not fill_result.get("filled"):
                                _record_candidate_decision(
                                    signal_row,
                                    final_decision="REJECTED",
                                    reject_reason="live_entry_unfilled_after_cancel",
                                    gate="execution",
                                    model_action=action_map.get(action_val, "UNKNOWN"),
                                    proposed_size_usdc=size_usdc,
                                    available_balance=_available_bal,
                                    order_id=entry_order_id,
                                )
                                continue
                        
                        fill_payload = fill_result.get("response") or {}
                        actual_fill_price = float(fill_payload.get("price", fill_price) or fill_price)
                        sz = fill_payload.get("size", _order_shares)
                        actual_fill_size = float(sz) if (sz is not None and str(sz).strip() and float(sz) > 0) else _order_shares # BUG 8 FIX
                        
                        log_live_fill_event(signal_row, actual_fill_price, size_usdc, action_type="LIVE_TRADE")
                        # Register trade AFTER confirmed fill (not before)
                        trade = TradeLifecycle(
                            market=signal_row.get("market_title", signal_row.get("market", "Unknown")),
                            token_id=token_id,
                            condition_id=signal_row.get("condition_id"),
                            outcome_side=signal_row.get("outcome_side", signal_row.get("side", "YES")),
                        )
                        trade.confidence_at_entry = confidence # BUG 6 FIX
                        trade.signal_label = signal_row.get("signal_label", "UNKNOWN")
                        trade.enter(size_usdc=size_usdc, entry_price=actual_fill_price)
                        trade.shares = actual_fill_size
                        trade_manager.active_trades[market_key] = trade
                        _refresh_local_active_trade_state()
                        _invalidate_entry_available_balance()
                        logging.info("Live trade filled for %s at %s. Shares: %s", token_id, actual_fill_price, actual_fill_size)
                        _record_candidate_decision(
                            signal_row,
                            final_decision="ENTRY_FILLED",
                            gate="execution",
                            model_action=action_map.get(action_val, "UNKNOWN"),
                            proposed_size_usdc=size_usdc,
                            final_size_usdc=actual_fill_size * actual_fill_price,
                            available_balance=_available_bal,
                            order_id=entry_order_id,
                            fill_price=actual_fill_price,
                            fill_shares=actual_fill_size,
                        )
                    else:
                        # Paper trade: register in TradeManager (which logs to execution_log.csv)
                        # FIX C7: Don't also call execute_paper_trade — it creates duplicate log entries
                        trade = trade_manager.handle_signal(signal_row=pd.Series(signal_row), confidence=confidence, size_usdc=size_usdc, entry_price_override=fill_price)
                        if trade is not None:
                            _refresh_local_active_trade_state()
                            _invalidate_entry_available_balance()
                            logging.info(
                                "Brain: FOLLOW -> Paper filled %s USDC on %s at $%.3f for '%s' | label=%s confidence=%.2f",
                                size_usdc,
                                signal_row.get("outcome_side", "?"),
                                fill_price,
                                signal_row.get("market_title", "Unknown"),
                                signal_row.get("signal_label", "UNKNOWN"),
                                confidence,
                            )
                            _record_candidate_decision(
                                signal_row,
                                final_decision="PAPER_OPENED",
                                gate="paper_execution",
                                model_action=action_map.get(action_val, "UNKNOWN"),
                                proposed_size_usdc=size_usdc,
                                final_size_usdc=size_usdc,
                                available_balance=_available_bal,
                                paper_fill_price=fill_price,
                            )
                        else:
                            _record_candidate_decision(
                                signal_row,
                                final_decision="REJECTED",
                                reject_reason="paper_trade_manager_rejected",
                                gate="paper_execution",
                                model_action=action_map.get(action_val, "UNKNOWN"),
                                proposed_size_usdc=size_usdc,
                                available_balance=_available_bal,
                            )
                else:
                    # FIX BUG#4: pass available_balance and precomputed_calibrated_edge
                    # so IGNORE/veto skip records are no longer null for those fields.
                    vetoed_action = _action_meta.get("vetoed_action")
                    if _action_meta.get("rule_vetoed_rl_action"):
                        ignore_reason = "rule_veto"
                        log_model_action = action_map.get(vetoed_action, action_map.get(action_val, "UNKNOWN"))
                    else:
                        ignore_reason = "model_action_ignore_rule_blocked" if not rule_allows_entry else "model_action_ignore"
                        log_model_action = action_map.get(action_val, "UNKNOWN")
                    _log_candidate_skip(
                        signal_row,
                        ignore_reason,
                        gate="model_policy",
                        model_action=log_model_action,
                        available_balance=round(_pre_available_bal, 6) if _pre_available_bal is not None else None,
                        precomputed_calibrated_edge=_pre_calibrated_edge,
                        rule_allows_entry=rule_allows_entry,
                        rule_score=rule_eval.get("score") if isinstance(rule_eval, dict) else None,
                        rule_score_threshold=rule_eval.get("score_threshold") if isinstance(rule_eval, dict) else None,
                        rule_spread=rule_eval.get("spread") if isinstance(rule_eval, dict) else None,
                        rule_spread_threshold=rule_eval.get("spread_threshold") if isinstance(rule_eval, dict) else None,
                        rule_liquidity_metric=rule_eval.get("liquidity_metric") if isinstance(rule_eval, dict) else None,
                        rule_liquidity_value=rule_eval.get("liquidity_value") if isinstance(rule_eval, dict) else None,
                        rule_liquidity_threshold=rule_eval.get("liquidity_threshold") if isinstance(rule_eval, dict) else None,
                        rule_vetoed_rl_action=bool(_action_meta.get("rule_vetoed_rl_action")),
                        rule_vetoed_action=vetoed_action,
                    )

            if _candidate_skip_counts:
                logging.info("CANDIDATE_SKIP_SUMMARY %s", json.dumps(_candidate_skip_counts, sort_keys=True))
            if _candidate_decision_rows:
                try:
                    append_csv_frame(CANDIDATE_DECISIONS_FILE, pd.DataFrame(_candidate_decision_rows))
                except Exception as exc:
                    logging.warning("Failed to append candidate decision CSV: %s", exc)
            cycle_stats_row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle_id": cycle_id,
                "candidates_seen": _candidate_stats.get("candidates_seen", 0),
                "candidates_tradable": _candidate_stats.get("candidates_tradable", 0),
                "candidates_rejected": _candidate_stats.get("candidates_rejected", 0),
                "entries_sent": _candidate_stats.get("entries_sent", 0),
                "fills_received": _candidate_stats.get("fills_received", 0),
                "reject_breakdown": json.dumps(_candidate_skip_counts, sort_keys=True),
            }
            append_csv_record(CANDIDATE_CYCLE_STATS_FILE, cycle_stats_row)
            logging.info("CANDIDATE_CYCLE_SUMMARY %s", json.dumps(cycle_stats_row, separators=(",", ":")))

            # 4B. Open-position management path for hold / reduce / exit
            # FIX H3: Build price map by token_id for reliable matching
            _token_price_map = {}
            if not scored_df.empty and "token_id" in scored_df.columns:
                for _, _pr in scored_df.iterrows():
                    _tid = str(_pr.get("token_id", ""))
                    _cp = _pr.get("current_price", _pr.get("market_last_trade_price"))
                    if _tid and _cp is not None:
                        try: _token_price_map[_tid] = float(_cp)
                        except (TypeError, ValueError): pass
            # Also build market-title map as fallback
            market_price_key = next((c for c in ["market_title", "market", "question"] if c in markets_df.columns), None)
            if market_price_key and "current_price" in markets_df.columns:
                market_prices = markets_df.set_index(market_price_key)["current_price"].dropna().to_dict()
            else:
                market_prices = {}
            # Update trades: try token_id first, then market title fallback
            for _tk, _tr in list(trade_manager.active_trades.items()):
                _tid = str(_tr.token_id or "")
                updated = False
                if _tid in _token_price_map:
                    _tr.update_market(_token_price_map[_tid])
                    updated = True
                elif _tr.market in market_prices:
                    _tr.update_market(market_prices[_tr.market])
                    updated = True
                    
                # FIX 3: Prevent "Blind Trades" by forcing an API ping if market dropped off scraper
                if not updated and trading_mode == "live":
                    try:
                        _ob = orderbook_guard.analyze_book(_tid, depth=1)
                        if _ob.get("best_bid"):
                            _tr.update_market(_ob["best_bid"])
                    except: pass

            # If in live mode, reconcile with exchange before making decisions
            if trading_mode == "live" and execution_client is not None:
                live_position_book.rebuild_from_db()
                try:
                    live_position_book.rebuild_from_db()
                    reconciled_positions_df = live_position_book.get_enriched_open_positions(scored_df=scored_df)
                    reconciled_positions_df = position_telemetry.enrich_with_live_marks(
                        reconciled_positions_df,
                        orderbook_guard=orderbook_guard,
                        fallback_price_map=_token_price_map,
                    )
                    trade_manager.reconcile_live_positions(reconciled_positions_df=reconciled_positions_df)
                    live_positions_df_for_cycle = live_pnl.enrich_positions(reconciled_positions_df)
                    trajectory_metrics = position_telemetry.build_trajectory_metrics(live_positions_df_for_cycle)
                except Exception as exc:
                    logging.warning("Live trade reconciliation failed before management decisions: %s", exc)
            current_open_trades = [
                trade
                for trade in trade_manager.active_trades.values()
                if getattr(trade, "state", None) != TradeState.CLOSED
            ]
            if current_open_trades and (position_brain is not None or legacy_brain is not None):
                scored_lookup = {}
                if scored_df is not None and not scored_df.empty:
                    for _, sr in scored_df.iterrows():
                        sr_dict = sr.to_dict()
                        sr_key = _trade_key_from_signal(sr_dict)
                        if sr_key:
                            scored_lookup[sr_key] = sr_dict
                for trade in current_open_trades:
                    trade_key = _make_position_key(
                        token_id=trade.token_id,
                        condition_id=trade.condition_id,
                        outcome_side=trade.outcome_side,
                        market=trade.market,
                    )
                    signal_match = scored_lookup.get(trade_key, {}) if trade_key else {}
                    current_price = float(getattr(trade, "current_price", 0.0) or 0.0)
                    entry_price = float(getattr(trade, "entry_price", 0.0) or 0.0)
                    shares = float(getattr(trade, "shares", 0.0) or 0.0)
                    unrealized_pnl = float(getattr(trade, "unrealized_pnl", 0.0) or 0.0)
                    market_value = float(shares * current_price)
                    model_confidence = _coerce_confidence(
                        signal_match.get("confidence", getattr(trade, "confidence_at_entry", 0.5))
                    )
                    pos_dict = {
                        "token_id": trade.token_id,
                        "condition_id": trade.condition_id,
                        "outcome_side": trade.outcome_side,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "size_usdc": float(getattr(trade, "size_usdc", 0.0) or 0.0),
                        "shares": shares,
                        "market_value": market_value,
                        "unrealized_pnl": unrealized_pnl,
                        "market_title": trade.market,
                        "confidence": model_confidence,
                    }
                    token_id = str(trade.token_id or "")
                    try:
                        if position_brain is not None:
                            pos_action_val = position_brain.predict(pos_dict)
                        else:
                            obs = prepare_position_observation(pos_dict)
                            pos_action, _ = legacy_brain.predict(obs, deterministic=True)
                            pos_action_val = int(pos_action.item() if hasattr(pos_action, "item") else pos_action[0])
                    except Exception as e:
                        logging.warning("Position brain prediction failed for %s: %s. Defaulting to HOLD.", token_id, e)
                        pos_action_val = 3 # HOLD

                    pos_action_val = _normalize_position_action(pos_action_val)
                    trajectory_signal = trajectory_metrics.get(trade_key, {}) if trade_key else {}
                    pos_dict.update(trajectory_signal)
                    if pos_action_val == 3 and trajectory_signal.get("panic_exit_signal"):
                        pos_action_val = 5
                    elif pos_action_val == 3 and trajectory_signal.get("reversal_exit_signal"):
                        pos_action_val = 5
                    elif pos_action_val == 3 and trajectory_signal.get("liquidity_stress_signal") and unrealized_pnl > 0:
                        pos_action_val = 4
                    elif pos_action_val == 3 and trajectory_signal.get("profit_lock_signal"):
                        pos_action_val = 4

                    if pos_action_val == 4:
                        if getattr(trade, 'has_been_reduced', False):
                            continue
                        min_reduce_notional = max(
                            float(getattr(TradingConfig, "MIN_BET_USDC", 1.0)),
                            float(getattr(TradingConfig, "MIN_REDUCE_NOTIONAL_USDC", 2.5)),
                        )
                        min_remainder_notional = max(
                            0.0,
                            float(getattr(TradingConfig, "MIN_POSITION_REMAINDER_USDC", min_reduce_notional)),
                        )
                        projected_exit_price = max(float(current_price or entry_price or 0.0), 0.0)
                        projected_reduce_shares = max(float(trade.shares or 0.0) * 0.5, 0.0)
                        projected_reduce_notional = projected_reduce_shares * projected_exit_price
                        projected_remaining_shares = max(float(trade.shares or 0.0) - projected_reduce_shares, 0.0)
                        projected_remaining_notional = projected_remaining_shares * projected_exit_price
                        if (
                            projected_reduce_notional > 0
                            and (
                                projected_reduce_notional < min_reduce_notional
                                or (projected_remaining_shares > 0 and projected_remaining_notional < min_remainder_notional)
                            )
                        ):
                            logging.info(
                                "Anti-micro guard: converting REDUCE -> EXIT for %s (reduce=$%.2f remainder=$%.2f min_reduce=$%.2f min_remainder=$%.2f)",
                                token_id,
                                projected_reduce_notional,
                                projected_remaining_notional,
                                min_reduce_notional,
                                min_remainder_notional,
                            )
                            pos_action_val = 5

                    pos_action_map = {3: "HOLD", 4: "REDUCE", 5: "EXIT"}
                    action_str = pos_action_map.get(pos_action_val, "HOLD")
                    try:
                        db.execute(
                            "INSERT INTO model_decisions (token_id, condition_id, outcome_side, model_name, score, action, feature_snapshot, model_artifact, normalization_artifact) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                token_id,
                                trade.condition_id,
                                trade.outcome_side,
                                position_model_name,
                                float(pos_dict.get("confidence", 0.5) or 0.5),
                                action_str,
                                build_feature_snapshot(pos_dict),
                                position_model_artifact,
                                position_norm_artifact,
                            ),
                        )
                    except Exception as exc:
                        logging.warning("Failed to log management decision for %s: %s", token_id, exc)

                    if pos_action_val == 4: # REDUCE
                        # FIX 2: Stop Reduce Death Spiral. Only reduce if not already reduced.
                        if getattr(trade, 'has_been_reduced', False):
                            continue
                        logging.info("[%s] Reducing position for %s", trading_mode.upper(), token_id)
                        if trading_mode == "live" and order_manager is not None:
                            try:
                                _ob_exit = orderbook_guard.analyze_book(token_id, depth=5)
                                if _ob_exit.get("best_bid") is not None and float(_ob_exit.get("best_bid")) > 0: # BUG 7 FIX: Prevent $0 limit sells
                                    exit_price = _ob_exit["best_bid"]
                                else:
                                    exit_price = quote_exit_price(pos_dict)
                            except Exception:
                                exit_price = quote_exit_price(pos_dict)
                            exit_shares = trade.shares * 0.5
                            if exit_price is not None and exit_shares > 0:
                                pre_reduce_shares = max(float(trade.shares or 0.0), 0.0)
                                reduce_result = _execute_live_sell_ladder(
                                    token_id=token_id,
                                    requested_shares=exit_shares,
                                    condition_id=trade.condition_id,
                                    outcome_side=trade.outcome_side,
                                    reference_price=exit_price,
                                    close_reason="rl_reduce",
                                )
                                actual_fill_size = min(float(reduce_result.get("filled_shares", 0.0) or 0.0), pre_reduce_shares)
                                if actual_fill_size > 1e-6:
                                    actual_fill_price = float(reduce_result.get("avg_price", exit_price) or exit_price)
                                    log_live_fill_event(pos_dict, actual_fill_price, actual_fill_size, action_type="LIVE_REDUCE")
                                    trade.partial_exit(
                                        fraction=min(1.0, actual_fill_size / max(pre_reduce_shares, 1e-9)),
                                        exit_price=actual_fill_price,
                                    )
                                    trade.has_been_reduced = True
                                    logging.info(
                                        "Live REDUCE filled %.6f/%.6f shares for %s across %s attempt(s).",
                                        actual_fill_size,
                                        pre_reduce_shares,
                                        token_id,
                                        len(reduce_result.get("attempts", [])),
                                    )
                                else:
                                    logging.warning(
                                        "Live REDUCE remained unfilled for %s after %s attempt(s).",
                                        token_id,
                                        len(reduce_result.get("attempts", [])),
                                    )
                            else:
                                logging.warning("Live REDUCE skipped for %s due to invalid exit price/size", token_id)
                        else:
                            trade.has_been_reduced = True
                            trade.partial_exit(fraction=0.5, exit_price=trade.current_price) # Paper reduce
                            logging.info("Paper REDUCE for %s. Current PnL: %.2f (reason=rl_reduce)", token_id, trade.realized_pnl)
                    elif pos_action_val == 5: # EXIT
                        logging.info("[%s] Exiting position for %s", trading_mode.upper(), token_id)
                        if trading_mode == "live" and order_manager is not None:
                            try:
                                _ob_exit = orderbook_guard.analyze_book(token_id, depth=5)
                                if _ob_exit.get("best_bid") is not None and float(_ob_exit.get("best_bid")) > 0: # BUG 7 FIX: Prevent $0 limit sells
                                    exit_price = _ob_exit["best_bid"]
                                else:
                                    exit_price = quote_exit_price(pos_dict)
                            except Exception:
                                exit_price = quote_exit_price(pos_dict)
                            exit_shares = trade.shares
                            if exit_price is not None and exit_shares > 0:
                                pre_exit_shares = max(float(trade.shares or 0.0), 0.0)
                                exit_result = _execute_live_sell_ladder(
                                    token_id=token_id,
                                    requested_shares=exit_shares,
                                    condition_id=trade.condition_id,
                                    outcome_side=trade.outcome_side,
                                    reference_price=exit_price,
                                    close_reason="rl_exit",
                                )
                                actual_fill_size = min(float(exit_result.get("filled_shares", 0.0) or 0.0), pre_exit_shares)
                                if actual_fill_size > 1e-6:
                                    actual_fill_price = float(exit_result.get("avg_price", exit_price) or exit_price)
                                    log_live_fill_event(pos_dict, actual_fill_price, actual_fill_size, action_type="LIVE_EXIT")
                                    if actual_fill_size >= pre_exit_shares - 1e-6:
                                        trade.close(exit_price=actual_fill_price, reason="rl_exit") # Update TradeLifecycle
                                        trade_manager.persist_closed_trades([trade])
                                        trade_manager.active_trades.pop(_make_position_key(token_id=trade.token_id, condition_id=trade.condition_id, outcome_side=trade.outcome_side, market=trade.market), None) # Remove from active trades
                                    else:
                                        trade.partial_exit(
                                            fraction=min(1.0, actual_fill_size / max(pre_exit_shares, 1e-9)),
                                            exit_price=actual_fill_price,
                                        )
                                        logging.warning(
                                            "Live EXIT partially filled for %s (filled=%.6f remaining=%.6f).",
                                            token_id,
                                            actual_fill_size,
                                            float(trade.shares or 0.0),
                                        )
                                else:
                                    logging.warning(
                                        "Live EXIT remained unfilled for %s after %s attempt(s).",
                                        token_id,
                                        len(exit_result.get("attempts", [])),
                                    )
                            else:
                                logging.warning("Live EXIT skipped for %s due to invalid exit price/size", token_id)
                        else:
                            trade.close(exit_price=trade.current_price, reason="rl_exit") # FIX M2: real reason
                            logging.info("Paper EXIT for %s. Realized PnL: %.2f", token_id, trade.realized_pnl)
                            trade_manager.active_trades.pop(_make_position_key(token_id=trade.token_id, condition_id=trade.condition_id, outcome_side=trade.outcome_side, market=trade.market), None) # Remove from active trades

            # Process any pending exits (e.g., from CLOSE_LONG signals or internal rules)
            predictive_exit_targets = _build_predictive_exit_targets(
                scored_df,
                [
                    trade
                    for trade in trade_manager.active_trades.values()
                    if getattr(trade, "state", None) != TradeState.CLOSED
                ],
            )
            closed_trades = trade_manager.process_exits(
                datetime.now(timezone.utc),
                persist_closed=(trading_mode != "live"),
                predictive_exit_targets=predictive_exit_targets,
                trajectory_metrics=trajectory_metrics,
            )
            if closed_trades:
                logging.info("[%s] Processed %s closed trades.", trading_mode.upper(), len(closed_trades))

                # CLEANED LIVE EXIT & MONEY MANAGER BLOCK
            if trading_mode == "live" and order_manager is not None:
                for ct in closed_trades:
                    if str(getattr(ct, "close_reason", "") or "").strip().lower() == "external_manual_close":
                        logging.info(
                            "Skipping SELL for externally-closed trade token=%s reason=%s",
                            str(getattr(ct, "token_id", "") or "")[:16],
                            getattr(ct, "close_reason", None),
                        )
                        try:
                            trade_manager.persist_closed_trades([ct])
                        except Exception:
                            pass
                        continue
                    _ct_shares = float(getattr(ct, "shares", 0.0) or 0.0)
                    _ct_px = float(getattr(ct, "current_price", 0.0) or getattr(ct, "entry_price", 0.0) or 0.0)
                    if _ct_shares <= 0:
                        continue
                    if (_ct_shares * _ct_px) < 0.01:
                        logging.info(
                            "Skipping SELL for dust closed trade token=%s shares=%.8f notional=$%.6f reason=%s",
                            str(getattr(ct, "token_id", "") or "")[:16],
                            _ct_shares,
                            _ct_shares * _ct_px,
                            getattr(ct, "close_reason", None),
                        )
                        try:
                            trade_manager.persist_closed_trades([ct])
                        except Exception:
                            pass
                        continue
                    _ct_token = str(ct.token_id or "")
                    if not _ct_token: continue
                    
                    try:
                        _ob = orderbook_guard.analyze_book(_ct_token, depth=5)
                        _exit_p = _ob.get("best_bid") or ct.current_price
                    except Exception:
                        _exit_p = ct.current_price
                        
                    if _exit_p and _exit_p > 0:
                        logging.info("Submitting SELL for rule exit: token=%s shares=%.2f price=%.4f reason=%s", 
                                     _ct_token[:16], ct.shares, _exit_p, ct.close_reason)
                        try:
                            _pre_ct_shares = max(float(ct.shares or 0.0), 0.0)
                            _exit_result = _execute_live_sell_ladder(
                                token_id=_ct_token,
                                requested_shares=_pre_ct_shares,
                                condition_id=ct.condition_id,
                                outcome_side=ct.outcome_side,
                                reference_price=_exit_p,
                                close_reason=getattr(ct, "close_reason", None) or "policy_exit",
                            )
                            _filled_ct_shares = min(float(_exit_result.get("filled_shares", 0.0) or 0.0), _pre_ct_shares)
                            if _filled_ct_shares > 1e-6:
                                _actual_exit_price = float(_exit_result.get("avg_price", _exit_p) or _exit_p)
                                log_live_fill_event(
                                    {"token_id": _ct_token, "market_title": ct.market, "outcome_side": ct.outcome_side, "current_price": _actual_exit_price},
                                    _actual_exit_price,
                                    _filled_ct_shares,
                                    action_type=f"LIVE_EXIT_{ct.close_reason}",
                                )
                                if _filled_ct_shares >= _pre_ct_shares - 1e-6:
                                    trade_manager.persist_closed_trades([ct])
                                else:
                                    trade_manager.active_trades[_make_position_key(token_id=ct.token_id, condition_id=ct.condition_id, outcome_side=ct.outcome_side, market=ct.market)] = ct
                                    ct.state = TradeState.OPEN
                                    ct.closed_at = None
                                    ct.close_reason = None
                                    ct.partial_exit(
                                        fraction=min(1.0, _filled_ct_shares / max(_pre_ct_shares, 1e-9)),
                                        exit_price=_actual_exit_price,
                                    )
                                    logging.warning(
                                        "Live SELL partially filled for %s (filled=%.6f remaining=%.6f). Restored remainder to active tracking.",
                                        _ct_token[:16],
                                        _filled_ct_shares,
                                        float(ct.shares or 0.0),
                                    )
                            else:
                                trade_manager.active_trades[_make_position_key(token_id=ct.token_id, condition_id=ct.condition_id, outcome_side=ct.outcome_side, market=ct.market)] = ct
                                ct.state = TradeState.OPEN
                                ct.close_reason = None
                                logging.warning("Live SELL failed for %s after %s attempt(s). Restored to active tracking.", _ct_token[:16], len(_exit_result.get("attempts", [])))
                        except Exception as _exc:
                            logging.error("Failed SELL for %s: %s", _ct_token[:16], _exc)

            # SINGLE MoneyManager Update
            for ct in closed_trades:
                # Only record if it wasn't rolled back into OPEN state
                if getattr(ct, 'state', None) == TradeState.CLOSED:
                    if ct.realized_pnl >= 0:
                        _money_mgr.record_win(ct.realized_pnl)
                    else:
                        _money_mgr.record_loss(ct.realized_pnl)
            finalized_closed_trades = [ct for ct in closed_trades if getattr(ct, "state", None) == TradeState.CLOSED]
            closed_trade_feedback_count = feedback_learner.record_closed_trades(finalized_closed_trades)

            # 5. Phase 2 analytics outputs
            trades_df = safe_read_csv(EXECUTION_FILE)
            if scored_df is not None: scored_df = scored_df.loc[:, ~scored_df.columns.duplicated()].copy()
            trader_signals_df = scored_df.copy()
            if "wallet_copied" not in trader_signals_df.columns and "trader_wallet" in trader_signals_df.columns:
                trader_signals_df = trader_signals_df.rename(columns={"trader_wallet": "wallet_copied"})
            if "market" in trader_signals_df.columns and "market_title" in trader_signals_df.columns:
                trader_signals_df["market"] = trader_signals_df["market"].fillna(trader_signals_df["market_title"])
                trader_signals_df = trader_signals_df.drop(columns=["market_title"])
            elif "market" not in trader_signals_df.columns and "market_title" in trader_signals_df.columns:
                trader_signals_df = trader_signals_df.rename(columns={"market_title": "market"})
            trader_signals_df = trader_signals_df.loc[:, ~trader_signals_df.columns.duplicated()].copy()
            trader_analytics.write(trader_signals_df, trades_df)
            backtester.write(trader_signals_df)
            dataset_builder.write()

            alerts_df = safe_read_csv("logs/alerts.csv")

            if trading_mode == "live" and live_position_book is not None and live_pnl is not None:
                live_position_book.rebuild_from_db()
                open_positions_df_for_status = live_position_book.get_enriched_open_positions(scored_df=scored_df)
                open_positions_df_for_status = position_telemetry.enrich_with_live_marks(
                    open_positions_df_for_status,
                    orderbook_guard=orderbook_guard,
                    fallback_price_map=_token_price_map if "_token_price_map" in locals() else {},
                )
                open_positions_df_for_status = live_pnl.enrich_positions(open_positions_df_for_status)
                if not open_positions_df_for_status.empty:
                    try:
                        latest_trajectory_metrics = position_telemetry.build_trajectory_metrics(open_positions_df_for_status)
                        open_positions_df_for_status = position_telemetry.apply_trajectory_metrics(
                            open_positions_df_for_status,
                            latest_trajectory_metrics,
                        )
                        position_telemetry.capture_positions(open_positions_df_for_status)
                    except Exception as exc:
                        logging.warning("Position telemetry capture failed at status stage: %s", exc)
                pnl_summary = live_pnl.summarize_portfolio(open_positions_df_for_status)
                autonomous_monitor.write_heartbeat(
                    "trade_manager",
                    status="ok",
                    message="live_positions_reconciled",
                    extra={
                        "open_positions": pnl_summary.get("open_positions", 0),
                        "gross_market_value": pnl_summary.get("gross_market_value", 0.0),
                        "realized_pnl": pnl_summary.get("realized_pnl", 0.0),
                        "unrealized_pnl": pnl_summary.get("unrealized_pnl", 0.0),
                        "total_pnl": pnl_summary.get("total_pnl", 0.0),
                        "pnl_source": pnl_summary.get("pnl_source", "live_reconciled"),
                    },
                )
            else:
                open_positions_for_status = trade_manager.get_open_positions()
                open_positions_df_for_status = pd.DataFrame([trade.__dict__ for trade in open_positions_for_status]) if open_positions_for_status else pd.DataFrame()
                autonomous_monitor.write_heartbeat("trade_manager", status="ok", message="trades_updated", extra={"open_positions": len(open_positions_for_status)})

            autonomous_monitor.write_status(trader_signals_df, trades_df, alerts_df, open_positions_df_for_status)
            if trading_mode == "live":
                trade_manager.persist_open_positions(reconciled_positions_df=open_positions_df_for_status)
            else:
                trade_manager.persist_open_positions()
            try:
                sync_ops_state_to_db("logs")
            except Exception as exc:
                logging.warning("Ops state sync to DB failed: %s", exc)
            # LIVE-TRADE FIX: enable live retraining by default so the RL model
            # actually learns from wins and losses during real trading.
            allow_live_retrain = os.getenv("ENABLE_LIVE_RETRAIN", "true").strip().lower() in {"1", "true", "yes", "on"}
            retrain_promoted = False
            if trading_mode != "live" or allow_live_retrain:
                retrain_promoted = bool(retrainer.maybe_retrain(
                    force=closed_trade_feedback_count > 0,
                    reason="closed_trade_feedback" if closed_trade_feedback_count > 0 else "scheduled_cycle_check",
                ))
                if retrain_promoted:
                    _refresh_runtime_model_handles(reason="retrain_promoted_models_activated")
                autonomous_monitor.write_heartbeat("retrainer", status="ok", message="retrain_checked")
            else:
                autonomous_monitor.write_heartbeat("retrainer", status="ok", message="retrain_skipped_live")

            open_positions_count_for_sleep = 0
            if trading_mode == "live":
                open_positions_count_for_sleep = len(open_positions_df_for_status) if open_positions_df_for_status is not None else 0
            else:
                open_positions_count_for_sleep = len(open_positions_for_status)

            if open_positions_count_for_sleep > 0:
                logging.info("Cycle complete. Fast-polling active trades. Sleeping for 5 seconds...")
                time.sleep(5)
            else:
                logging.info("Cycle complete. Sleeping for 60 seconds...")
                time.sleep(60)

        except KeyboardInterrupt:
            logging.info("Supervisor halted manually by user.")
            break
        except Exception as e:
            autonomous_monitor.write_failure("supervisor", str(e))
            logging.error(f"Critical error in main loop: {e}. Relaxing for 60 seconds to respect API limits (Memory state is completely preserved).")
            time.sleep(60)


if __name__ == "__main__":
    main_loop()

