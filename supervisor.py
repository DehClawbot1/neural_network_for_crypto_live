from trade_lifecycle import TradeState
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import json
import signal
import time
import logging
import re
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from btc_regime_router import apply_regime_model_blend
from btc_threshold_guard import find_conflicting_btc_price_threshold_position
from weather_temperature_guard import find_conflicting_weather_temperature_position

try:
    from sb3_contrib import RecurrentPPO
except (ImportError, ModuleNotFoundError):
    RecurrentPPO = None

from leaderboard_scraper import run_scraper_cycle
from market_monitor import (
    fetch_btc_markets,
    fetch_btc_updown_markets,
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
from btc_forecast_model import BTCForecastModel
from btc_multitimeframe import BTCMultiTimeframeForecaster
from btc_forecast_eval import BTCForecastEvaluator
from supervisor_btc_pipeline import apply_btc_pipeline
from sentiment_analyzer import SentimentAnalyzer
from macro_analyzer import MacroAnalyzer
from onchain_analyzer import OnChainAnalyzer
from trade_feedback_learner import TradeFeedbackLearner
from position_telemetry import PositionTelemetry
from wallet_state_engine import (
    resolve_source_wallet_reduce_fraction,
    should_convert_reduce_to_exit,
    source_wallet_signal_matches_trade,
)
from performance_governor import PerformanceGovernor
from trade_lifecycle_audit import TradeLifecycleAuditor
from benchmark_strategy import BenchmarkStrategy
from trade_quality import build_quality_context, resolve_entry_signal_label
from trading_mode_preset import select_trading_mode, apply_preset, PRESETS
from btc_trade_feedback import BTCTradeFeedback
from weather_temperature_strategy import WeatherTemperatureStrategy
from weather_temperature_markets import fetch_weather_temperature_markets
from brain_paths import active_runtime_identity, resolve_brain_context
from brain_log_routing import append_csv_with_brain_mirrors
from model_registry import ModelRegistry
try:
    from inference_runtime_guard import (
        reset_cycle as _reset_inference_runtime_guard,
        has_errors as _inference_guard_has_errors,
        get_errors as _inference_guard_get_errors,
    )
except (ImportError, ModuleNotFoundError):
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
SOURCE_WALLET_LOG_COLUMNS = [
    "wallet_state_gate_pass",
    "wallet_state_gate_reason",
    "wallet_agreement_score",
    "wallet_conflict_with_stronger",
    "wallet_stronger_conflict_score",
    "wallet_support_strength",
    "source_wallet_direction_confidence",
    "source_wallet_position_event",
    "source_wallet_net_position_increased",
    "source_wallet_current_net_exposure",
    "source_wallet_average_entry",
    "source_wallet_last_add",
    "source_wallet_last_reduce",
    "source_wallet_last_close",
    "source_wallet_current_direction",
    "source_wallet_reduce_fraction",
    "source_wallet_state_freshness_seconds",
    "source_wallet_freshness_score",
    "source_wallet_fresh",
    "source_wallet_exit_signal",
    "source_wallet_reduce_signal",
    "source_wallet_reversal_signal",
]
WEATHER_LOG_COLUMNS = [
    "market_family",
    "wallet_temp_hit_rate_90d",
    "wallet_temp_realized_pnl_90d",
    "wallet_region_score",
    "wallet_temp_range_skill",
    "wallet_temp_threshold_skill",
    "weather_parseable",
    "weather_parse_error",
    "weather_location",
    "weather_country",
    "weather_event_date_local",
    "weather_resolution_timezone",
    "weather_question_type",
    "weather_temp_unit",
    "weather_lower_c",
    "weather_upper_c",
    "weather_interval_width_c",
    "weather_cluster_key",
    "forecast_ready",
    "forecast_stale",
    "forecast_missing_reason",
    "forecast_source",
    "forecast_max_temp_c",
    "forecast_p_hit_interval",
    "forecast_margin_to_lower_c",
    "forecast_margin_to_upper_c",
    "forecast_uncertainty_c",
    "forecast_last_update_ts",
    "forecast_drift_c",
    "weather_fair_probability_yes",
    "weather_fair_probability_side",
    "weather_market_probability",
    "weather_forecast_edge",
    "weather_forecast_confirms_direction",
    "weather_forecast_margin_score",
    "weather_forecast_stability_score",
    "weather_entry_allowed_by_forecast",
    "analytics_only",
    "analytics_only_reason",
]
BTC_CONTEXT_LOG_COLUMNS = [
    "btc_live_price",
    "btc_live_spot_price",
    "btc_live_index_price",
    "btc_live_mark_price",
    "btc_live_price_kalman",
    "btc_live_spot_price_kalman",
    "btc_live_index_price_kalman",
    "btc_live_mark_price_kalman",
    "btc_live_funding_rate",
    "btc_live_source_quality",
    "btc_live_source_quality_score",
    "btc_live_source_divergence_bps",
    "btc_live_spot_index_basis_bps",
    "btc_live_mark_index_basis_bps",
    "btc_live_mark_spot_basis_bps",
    "btc_live_spot_index_basis_bps_kalman",
    "btc_live_mark_index_basis_bps_kalman",
    "btc_live_mark_spot_basis_bps_kalman",
    "btc_live_return_1m",
    "btc_live_return_5m",
    "btc_live_return_15m",
    "btc_live_return_1h",
    "btc_live_return_1m_kalman",
    "btc_live_return_5m_kalman",
    "btc_live_return_15m_kalman",
    "btc_live_return_1h_kalman",
    "btc_live_volatility_proxy",
    "btc_live_bias",
    "btc_live_confluence",
    "btc_live_confluence_kalman",
    "btc_live_index_ready",
    "btc_live_index_feed_available",
    "btc_live_mark_feed_available",
    "market_structure",
    "trend_score",
    "btc_trend_bias",
    "btc_trend_confluence",
    "btc_volatility_regime",
    "btc_volatility_regime_score",
    "btc_momentum_regime",
    "btc_momentum_confluence",
    "btc_market_regime_label",
    "btc_market_regime_score",
    "btc_market_regime_trend_score",
    "btc_market_regime_volatility_score",
    "btc_market_regime_chaos_score",
    "btc_market_regime_stability_score",
    "btc_market_regime_is_calm",
    "btc_market_regime_is_trend",
    "btc_market_regime_is_volatile",
    "btc_market_regime_is_chaotic",
    "btc_market_regime_primary_model",
    "btc_market_regime_confidence_multiplier",
    "btc_market_regime_weight_legacy",
    "btc_market_regime_weight_stage1",
    "btc_market_regime_weight_stage2",
    "legacy_p_tp_before_sl",
    "legacy_expected_return",
    "legacy_edge_score",
    "stage1_p_tp_before_sl",
    "stage1_expected_return",
    "stage1_edge_score",
    "stage1_lower_confidence_bound",
    "stage1_return_std",
    "temporal_p_tp_before_sl",
    "temporal_expected_return",
    "temporal_edge_score",
    "regime_blended_p_tp_before_sl",
    "regime_blended_expected_return",
    "regime_blended_conservative_expected_return",
    "regime_blended_edge_score",
]
BRAIN_RUNTIME_LOG_COLUMNS = [
    "brain_id",
    "active_model_group",
    "active_model_kind",
    "active_regime",
]
RAW_CANDIDATE_LOG_COLUMNS = [
    "timestamp",
    "trader_wallet",
    "market_title",
    "condition_id",
    "token_id",
    "market_slug",
    "order_side",
    "trade_side",
    "outcome_side",
    "entry_intent",
    "side",
    "entry_price",
    "best_bid",
    "best_ask",
    "spread",
    "trader_win_rate",
    "wallet_trade_count_30d",
    "wallet_avg_size_30d",
    "wallet_winrate_30d",
    "wallet_alpha_30d",
    "wallet_avg_forward_return_15m",
    "wallet_signal_precision_tp",
    "wallet_recent_streak",
    "wallet_same_market_history",
    "normalized_trade_size",
    "current_price",
    "time_left",
    "market_liquidity",
    "market_volume",
    "liquidity_score",
    "volume_score",
    "market_last_trade_price",
    "probability_momentum",
    "volatility_score",
    "whale_consensus_score",
    "whale_pressure",
    "market_structure_score",
    "volatility_risk",
    "time_decay_score",
    "open_positions_count",
    "open_positions_negotiated_value_total",
    "open_positions_max_payout_total",
    "open_positions_current_value_total",
    "open_positions_unrealized_pnl_total",
    "open_positions_unrealized_pnl_pct_total",
    "open_positions_avg_to_now_price_change_pct_mean",
    "open_positions_avg_to_now_price_change_pct_min",
    "open_positions_avg_to_now_price_change_pct_max",
    "open_positions_winner_count",
    "open_positions_loser_count",
    "raw_size",
    "market_url",
] + BRAIN_RUNTIME_LOG_COLUMNS + SOURCE_WALLET_LOG_COLUMNS + WEATHER_LOG_COLUMNS + BTC_CONTEXT_LOG_COLUMNS
RANKED_SIGNAL_LOG_COLUMNS = [
    "timestamp",
    "market",
    "market_title",
    "market_slug",
    "wallet_copied",
    "wallet_short",
    "trader_wallet",
    "token_id",
    "condition_id",
    "order_side",
    "trade_side",
    "outcome_side",
    "entry_intent",
    "side",
    "signal_label",
    "confidence",
    "reason",
    "reason_summary",
    "recommended_action",
    "action",
    "market_url",
    "trader_win_rate",
    "normalized_trade_size",
    "current_price",
    "market_last_trade_price",
    "price",
    "best_bid",
    "best_ask",
    "time_left",
    "liquidity_score",
    "volume_score",
    "probability_momentum",
    "volatility_score",
    "whale_pressure",
    "market_structure_score",
    "volatility_risk",
    "time_decay_score",
    "open_positions_count",
    "open_positions_negotiated_value_total",
    "open_positions_max_payout_total",
    "open_positions_current_value_total",
    "open_positions_unrealized_pnl_total",
    "open_positions_unrealized_pnl_pct_total",
    "open_positions_avg_to_now_price_change_pct_mean",
    "open_positions_avg_to_now_price_change_pct_min",
    "open_positions_avg_to_now_price_change_pct_max",
    "open_positions_winner_count",
    "open_positions_loser_count",
    "brain_id",
    "active_model_group",
    "active_model_kind",
    "active_regime",
    "edge_score",
    "expected_return",
    "p_tp_before_sl",
    "risk_adjusted_ev",
    "entry_ev",
    "execution_quality_score",
] + SOURCE_WALLET_LOG_COLUMNS + WEATHER_LOG_COLUMNS + BTC_CONTEXT_LOG_COLUMNS


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
    append_csv_frame(path, pd.DataFrame([record]))


def append_csv_frame(path, df):
    if df is None or df.empty:
        return
    append_csv_with_brain_mirrors(path, df, shared_logs_dir="logs", shared_weights_dir="weights")


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


def _frame_column_as_series(frame: pd.DataFrame, column_name: str, default_value=np.nan) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=object)
    if column_name not in frame.columns:
        return pd.Series([default_value] * len(frame), index=frame.index)

    raw = frame.loc[:, column_name]
    if isinstance(raw, pd.DataFrame):
        if raw.empty:
            return pd.Series([default_value] * len(frame), index=frame.index)
        values = []
        for _, row in raw.iterrows():
            picked = default_value
            for item in row.tolist():
                if pd.notna(item):
                    picked = item
                    break
            values.append(picked)
        series = pd.Series(values, index=raw.index)
    elif isinstance(raw, pd.Series):
        series = raw
    else:
        try:
            series = pd.Series(raw, index=frame.index)
        except Exception:
            values = list(raw) if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)) else [raw] * len(frame)
            series = pd.Series(values[: len(frame)])

    series = series.reset_index(drop=True)
    if len(series) < len(frame):
        series = series.reindex(range(len(frame)), fill_value=default_value)
    elif len(series) > len(frame):
        series = series.iloc[: len(frame)]
    series.index = frame.index
    return series


def _frame_numeric_series(frame: pd.DataFrame, column_name: str, default_value=0.0) -> pd.Series:
    series = pd.to_numeric(_frame_column_as_series(frame, column_name, np.nan), errors="coerce")
    if isinstance(default_value, pd.Series):
        fallback = pd.to_numeric(default_value, errors="coerce").reset_index(drop=True)
        if len(fallback) < len(frame):
            fallback = fallback.reindex(range(len(frame)), fill_value=np.nan)
        elif len(fallback) > len(frame):
            fallback = fallback.iloc[: len(frame)]
        fallback.index = frame.index
        return series.where(series.notna(), fallback)
    return series.fillna(float(default_value))


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
    if "performance_governor" in text:
        return "performance_governor"
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
    raw_df = raw_df.reindex(columns=RAW_CANDIDATE_LOG_COLUMNS)
    append_csv_frame(RAW_CANDIDATES_FILE, raw_df)


def log_ranked_signal(signal_row):
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market": signal_row.get("market_title", "Unknown Market"),
        "market_title": signal_row.get("market_title", signal_row.get("market", "Unknown Market")),
        "market_slug": signal_row.get("market_slug"),
        "wallet_copied": str(signal_row.get("trader_wallet", signal_row.get("wallet_copied", "Unknown"))),
        "wallet_short": signal_row.get("wallet_short", str(signal_row.get("trader_wallet", signal_row.get("wallet_copied", "Unknown")))[:8]),
        "trader_wallet": signal_row.get("trader_wallet", signal_row.get("wallet_copied")),
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
        "reason_summary": signal_row.get("reason_summary", signal_row.get("reason", "")),
        "recommended_action": signal_row.get("recommended_action", signal_row.get("entry_intent", "OPEN_LONG")),
        "action": signal_row.get("action", signal_row.get("entry_intent", "OPEN_LONG")),
        "market_url": signal_row.get("market_url"),
        "trader_win_rate": signal_row.get("trader_win_rate"),
        "normalized_trade_size": signal_row.get("normalized_trade_size"),
        "current_price": signal_row.get("current_price"),
        "market_last_trade_price": signal_row.get("market_last_trade_price"),
        "price": signal_row.get("price", signal_row.get("entry_price", signal_row.get("current_price"))),
        "best_bid": signal_row.get("best_bid"),
        "best_ask": signal_row.get("best_ask"),
        "time_left": signal_row.get("time_left"),
        "liquidity_score": signal_row.get("liquidity_score"),
        "volume_score": signal_row.get("volume_score"),
        "probability_momentum": signal_row.get("probability_momentum"),
        "volatility_score": signal_row.get("volatility_score"),
        "whale_pressure": signal_row.get("whale_pressure"),
        "market_structure_score": signal_row.get("market_structure_score"),
        "volatility_risk": signal_row.get("volatility_risk"),
        "time_decay_score": signal_row.get("time_decay_score"),
        "open_positions_count": signal_row.get("open_positions_count"),
        "open_positions_negotiated_value_total": signal_row.get("open_positions_negotiated_value_total"),
        "open_positions_max_payout_total": signal_row.get("open_positions_max_payout_total"),
        "open_positions_current_value_total": signal_row.get("open_positions_current_value_total"),
        "open_positions_unrealized_pnl_total": signal_row.get("open_positions_unrealized_pnl_total"),
        "open_positions_unrealized_pnl_pct_total": signal_row.get("open_positions_unrealized_pnl_pct_total"),
        "open_positions_avg_to_now_price_change_pct_mean": signal_row.get("open_positions_avg_to_now_price_change_pct_mean"),
        "open_positions_avg_to_now_price_change_pct_min": signal_row.get("open_positions_avg_to_now_price_change_pct_min"),
        "open_positions_avg_to_now_price_change_pct_max": signal_row.get("open_positions_avg_to_now_price_change_pct_max"),
        "open_positions_winner_count": signal_row.get("open_positions_winner_count"),
        "open_positions_loser_count": signal_row.get("open_positions_loser_count"),
        "brain_id": signal_row.get("brain_id"),
        "active_model_group": signal_row.get("active_model_group"),
        "active_model_kind": signal_row.get("active_model_kind"),
        "active_regime": signal_row.get("active_regime"),
        "edge_score": signal_row.get("edge_score"),
        "expected_return": signal_row.get("expected_return"),
        "p_tp_before_sl": signal_row.get("p_tp_before_sl"),
        "risk_adjusted_ev": signal_row.get("risk_adjusted_ev"),
        "entry_ev": signal_row.get("entry_ev"),
        "execution_quality_score": signal_row.get("execution_quality_score"),
    }
    for extra_key in SOURCE_WALLET_LOG_COLUMNS + WEATHER_LOG_COLUMNS + BTC_CONTEXT_LOG_COLUMNS:
        record[extra_key] = signal_row.get(extra_key)
    record = {key: record.get(key) for key in RANKED_SIGNAL_LOG_COLUMNS}
    append_csv_record(SIGNALS_FILE, record)


def _active_trades_to_positions_frame(active_trades):
    rows = []
    for trade in active_trades or []:
        entry_price = _safe_float(getattr(trade, "entry_price", 0.0), default=0.0)
        current_price = _safe_float(getattr(trade, "current_price", entry_price), default=entry_price)
        shares = _safe_float(getattr(trade, "shares", 0.0), default=0.0)
        negotiated_value_usdc = _safe_float(getattr(trade, "size_usdc", entry_price * shares), default=entry_price * shares)
        current_value_usdc = _safe_float(getattr(trade, "market_value", current_price * shares), default=current_price * shares)
        unrealized_pnl_total = _safe_float(
            getattr(trade, "unrealized_pnl", current_value_usdc - negotiated_value_usdc),
            default=current_value_usdc - negotiated_value_usdc,
        )
        avg_to_now_price_change = current_price - entry_price
        rows.append(
            {
                "negotiated_value_usdc": negotiated_value_usdc,
                "max_payout_usdc": _safe_float(getattr(trade, "max_payout_usdc", shares), default=shares),
                "current_value_usdc": current_value_usdc,
                "unrealized_pnl": unrealized_pnl_total,
                "unrealized_pnl_pct": _safe_float(
                    getattr(
                        trade,
                        "unrealized_pnl_pct",
                        (unrealized_pnl_total / negotiated_value_usdc) if negotiated_value_usdc > 0 else 0.0,
                    ),
                    default=(unrealized_pnl_total / negotiated_value_usdc) if negotiated_value_usdc > 0 else 0.0,
                ),
                "avg_to_now_price_change_pct": _safe_float(
                    getattr(
                        trade,
                        "avg_to_now_price_change_pct",
                        (avg_to_now_price_change / entry_price) if entry_price > 0 else 0.0,
                    ),
                    default=(avg_to_now_price_change / entry_price) if entry_price > 0 else 0.0,
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_open_position_context(positions_df: pd.DataFrame | None = None, active_trades=None) -> dict:
    work = positions_df.copy() if positions_df is not None and not positions_df.empty else _active_trades_to_positions_frame(active_trades)
    if work is None or work.empty:
        return {
            "open_positions_count": 0,
            "open_positions_negotiated_value_total": 0.0,
            "open_positions_max_payout_total": 0.0,
            "open_positions_current_value_total": 0.0,
            "open_positions_unrealized_pnl_total": 0.0,
            "open_positions_unrealized_pnl_pct_total": 0.0,
            "open_positions_avg_to_now_price_change_pct_mean": 0.0,
            "open_positions_avg_to_now_price_change_pct_min": 0.0,
            "open_positions_avg_to_now_price_change_pct_max": 0.0,
            "open_positions_winner_count": 0,
            "open_positions_loser_count": 0,
        }

    def _series_from_column(column_name: str, default_value):
        if column_name not in work.columns:
            return pd.Series([default_value] * len(work), index=work.index)

        raw = work.loc[:, column_name]
        if isinstance(raw, pd.DataFrame):
            if raw.empty:
                return pd.Series([default_value] * len(work), index=work.index)
            series = raw.apply(
                lambda row: next((value for value in row if not pd.isna(value)), default_value),
                axis=1,
            )
        elif isinstance(raw, pd.Series):
            series = raw
        else:
            try:
                series = pd.Series(raw, index=work.index)
            except Exception:
                values = list(raw) if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)) else [raw] * len(work)
                series = pd.Series(values)

        series = series.reset_index(drop=True)
        if len(series) < len(work):
            series = series.reindex(range(len(work)), fill_value=default_value)
        elif len(series) > len(work):
            series = series.iloc[: len(work)]
        series.index = work.index
        return series

    def _numeric_series(column_name: str, default_value):
        series = pd.to_numeric(_series_from_column(column_name, np.nan), errors="coerce")
        if isinstance(default_value, pd.Series):
            fallback = pd.to_numeric(default_value, errors="coerce")
            fallback = fallback.reset_index(drop=True)
            if len(fallback) < len(work):
                fallback = fallback.reindex(range(len(work)), fill_value=np.nan)
            elif len(fallback) > len(work):
                fallback = fallback.iloc[: len(work)]
            fallback.index = work.index
            return series.where(series.notna(), fallback)
        return series.fillna(float(default_value))

    shares = _numeric_series("shares", 0.0)
    avg_entry_price = _numeric_series("avg_entry_price", 0.0)
    entry_price = _numeric_series("entry_price", avg_entry_price).fillna(0.0)
    mark_price = _numeric_series("mark_price", entry_price)
    current_price = _numeric_series("current_price", mark_price).fillna(entry_price)
    negotiated_fallback = _numeric_series("size_usdc", entry_price * shares)
    negotiated = _numeric_series("negotiated_value_usdc", negotiated_fallback).fillna(entry_price * shares)
    max_payout = _numeric_series("max_payout_usdc", shares).fillna(shares)
    current_value_fallback = _numeric_series("market_value", current_price * shares)
    current_value = _numeric_series("current_value_usdc", current_value_fallback).fillna(current_price * shares)
    unrealized = _numeric_series("unrealized_pnl", current_value - negotiated).fillna(0.0)
    avg_change_fallback = _numeric_series(
        "price_change_pct",
        pd.Series(np.where(entry_price > 0, (current_price - entry_price) / entry_price, 0.0), index=work.index),
    )
    avg_change_pct = _numeric_series("avg_to_now_price_change_pct", avg_change_fallback).fillna(0.0)
    unrealized_pct = _numeric_series(
        "unrealized_pnl_pct",
        pd.Series(np.where(negotiated > 0, unrealized / negotiated, 0.0), index=work.index),
    ).fillna(0.0)

    return {
        "open_positions_count": int(len(work)),
        "open_positions_negotiated_value_total": float(negotiated.sum()),
        "open_positions_max_payout_total": float(max_payout.sum()),
        "open_positions_current_value_total": float(current_value.sum()),
        "open_positions_unrealized_pnl_total": float(unrealized.sum()),
        "open_positions_unrealized_pnl_pct_total": float(unrealized.sum() / negotiated.sum()) if float(negotiated.sum()) > 0 else 0.0,
        "open_positions_avg_to_now_price_change_pct_mean": float(avg_change_pct.mean()) if len(avg_change_pct) else 0.0,
        "open_positions_avg_to_now_price_change_pct_min": float(avg_change_pct.min()) if len(avg_change_pct) else 0.0,
        "open_positions_avg_to_now_price_change_pct_max": float(avg_change_pct.max()) if len(avg_change_pct) else 0.0,
        "open_positions_winner_count": int((unrealized_pct > 0).sum()),
        "open_positions_loser_count": int((unrealized_pct < 0).sum()),
    }


def attach_open_position_context(signals_df: pd.DataFrame, positions_df: pd.DataFrame | None = None, active_trades=None):
    if signals_df is None or signals_df.empty:
        return signals_df, summarize_open_position_context(positions_df=positions_df, active_trades=active_trades)
    context = summarize_open_position_context(positions_df=positions_df, active_trades=active_trades)
    work = signals_df.copy()
    for key, value in context.items():
        work[key] = value
    return work, context


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
    market_family = str(signal_row.get("market_family", "") or "").strip().lower()
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

    if market_family.startswith("weather_temperature"):
        decision_meta["weather_rule_only"] = True
        if str(signal_row.get("entry_intent", "OPEN_LONG") or "OPEN_LONG").upper() == "CLOSE_LONG":
            return 1
        return 1 if rule_allows_entry else 0

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
    if force_candidate and rule_allows_entry:
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
        if confidence >= rl_hold_min_confidence and rule_allows_entry:
            expected_return = _safe_float(signal_row.get("expected_return", 0.0), default=0.0)
            if expected_return <= aggressive_expected_return_floor:
                return 0
            edge_score = _safe_float(signal_row.get("edge_score", 0.0), default=0.0)
            return 2 if edge_score >= 0.04 else 1

    # FIX V5: If RL models are loaded and fallback is disabled / conditions not met, keep HOLD.
    if entry_brain is not None or legacy_brain is not None:
        return 0

    # Fallback: rule-based decision (ONLY used if no RL models are loaded)
    if not rule_allows_entry:
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

    normalized_signal_label = resolve_entry_signal_label(signal_row)
    trade_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market": signal_row.get("market_title", "Unknown Market"),
        "wallet_copied": str(signal_row.get("trader_wallet", signal_row.get("wallet_copied", "Unknown"))),
        "token_id": signal_row.get("token_id"),
        "condition_id": signal_row.get("condition_id"),
        "market_family": signal_row.get("market_family", "other"),
        "brain_id": signal_row.get("brain_id", ""),
        "active_model_group": signal_row.get("active_model_group", ""),
        "active_model_kind": signal_row.get("active_model_kind", ""),
        "active_regime": signal_row.get("active_regime", ""),
        "outcome_side": outcome_side,
        "order_side": signal_row.get("order_side", "BUY"),
        "signal_price": round(signal_price, 3),
        "fill_price": round(fill_price, 3),
        "size_usdc": size,
        "signal_label": normalized_signal_label,
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
    normalized_signal_label = resolve_entry_signal_label(signal_row)
    trade_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market": signal_row.get("market_title", signal_row.get("market", "Unknown Market")),
        "wallet_copied": str(signal_row.get("trader_wallet", signal_row.get("wallet_copied", "Unknown"))),
        "token_id": signal_row.get("token_id"),
        "condition_id": signal_row.get("condition_id"),
        "market_family": signal_row.get("market_family", "other"),
        "brain_id": signal_row.get("brain_id", ""),
        "active_model_group": signal_row.get("active_model_group", ""),
        "active_model_kind": signal_row.get("active_model_kind", ""),
        "active_regime": signal_row.get("active_regime", ""),
        "outcome_side": str(signal_row.get("outcome_side", signal_row.get("side", "UNKNOWN"))).upper(),
        "order_side": signal_row.get("order_side", signal_row.get("trade_side", "BUY")),
        "signal_price": round(float(signal_row.get("current_price", signal_row.get("price", fill_price)) or fill_price), 3),
        "fill_price": round(float(fill_price), 3),
        "size_usdc": float(size_usdc),
        "signal_label": normalized_signal_label,
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

def choose_cycle_sleep_interval(
    *,
    open_positions_count: int,
    entry_freeze_active: bool,
    active_position_poll_seconds: float,
    idle_poll_seconds: float,
    entry_freeze_poll_seconds: float,
) -> tuple[float, str]:
    if int(open_positions_count or 0) > 0:
        return max(1.0, float(active_position_poll_seconds)), "fast-polling active trades"
    if bool(entry_freeze_active):
        return max(1.0, float(entry_freeze_poll_seconds)), "entry freeze active; rechecking soon"
    return max(1.0, float(idle_poll_seconds)), "idle market scan"


def performance_governor_top_signal_decision(governor_state: dict, consumed_count: int) -> bool:
    """
    In governor top-signal mode, allow entries until one actual entry opens in-cycle.
    Failed execution attempts must not consume the slot.
    """
    if not bool((governor_state or {}).get("top_signal_only")):
        return True
    return int(consumed_count or 0) <= 0


def performance_governor_consume_top_signal_slot(governor_state: dict, consumed_count: int) -> int:
    if not bool((governor_state or {}).get("top_signal_only")):
        return int(consumed_count or 0)
    return int(consumed_count or 0) + 1

_shutdown_requested = False


def _request_shutdown(signum, frame):
    global _shutdown_requested
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    logging.info("Received %s — finishing current cycle then shutting down...", sig_name)
    _shutdown_requested = True


def _sleep_until_shutdown_or_timeout(total_seconds, step_seconds=0.25):
    """
    Sleep in short chunks so Ctrl+C / SIGINT can stop the supervisor promptly
    even during backoff or idle waits on Windows.
    Returns True when shutdown was requested during the wait.
    """
    global _shutdown_requested
    try:
        remaining = max(0.0, float(total_seconds or 0.0))
    except Exception:
        remaining = 0.0
    if remaining <= 0:
        return bool(_shutdown_requested)

    step = max(0.05, min(float(step_seconds or 0.25), 1.0))
    while remaining > 0 and not _shutdown_requested:
        chunk = min(step, remaining)
        try:
            time.sleep(chunk)
        except KeyboardInterrupt:
            _request_shutdown(getattr(signal, "SIGINT", 2), None)
            return True
        remaining -= chunk
    return bool(_shutdown_requested)

def compute_leaderboard_consensus(signals_df: pd.DataFrame, market_slug_prefix: str = "btc-updown-") -> dict:
    """
    Analyse leaderboard/scraper signals for BTC up-down markets and
    return a consensus dict.

    Used by _inject_always_on_signal to enrich the always-on entry with
    top-trader agreement/disagreement data.

    Returns:
        dict with keys: leaderboard_n_yes, leaderboard_n_no, leaderboard_n_total,
        leaderboard_bias (+1 YES majority, -1 NO majority, 0 balanced),
        leaderboard_agreement (0-1, how unanimous),
        leaderboard_vol_yes, leaderboard_vol_no (total size USDC each side)
    """
    result = {
        "leaderboard_n_yes": 0,
        "leaderboard_n_no": 0,
        "leaderboard_n_total": 0,
        "leaderboard_bias": 0,
        "leaderboard_agreement": 0.0,
        "leaderboard_vol_yes": 0.0,
        "leaderboard_vol_no": 0.0,
    }
    if signals_df is None or signals_df.empty:
        return result
    # Filter to BTC up-down market trades from leaderboard wallets
    slug_col = "market_slug" if "market_slug" in signals_df.columns else None
    if slug_col is None:
        return result
    mask = signals_df[slug_col].astype(str).str.lower().str.startswith(market_slug_prefix)
    btc_signals = signals_df[mask]
    if btc_signals.empty:
        return result
    # Exclude our own synthetic signals
    if "signal_source" in btc_signals.columns:
        btc_signals = btc_signals[btc_signals["signal_source"].astype(str).str.lower() != "always_on_market"]
    if btc_signals.empty:
        return result
    side_col = "outcome_side" if "outcome_side" in btc_signals.columns else "side"
    if side_col not in btc_signals.columns:
        return result
    sides = btc_signals[side_col].astype(str).str.upper()
    sizes = btc_signals["size"].astype(float).fillna(0) if "size" in btc_signals.columns else pd.Series([0] * len(btc_signals))
    n_yes = int((sides == "YES").sum())
    n_no = int((sides == "NO").sum())
    n_total = n_yes + n_no
    vol_yes = float(sizes[sides == "YES"].sum())
    vol_no = float(sizes[sides == "NO"].sum())
    if n_total == 0:
        return result
    majority_pct = max(n_yes, n_no) / n_total
    result.update({
        "leaderboard_n_yes": n_yes,
        "leaderboard_n_no": n_no,
        "leaderboard_n_total": n_total,
        "leaderboard_bias": 1 if n_yes > n_no else (-1 if n_no > n_yes else 0),
        "leaderboard_agreement": round(majority_pct, 4),
        "leaderboard_vol_yes": round(vol_yes, 2),
        "leaderboard_vol_no": round(vol_no, 2),
    })
    return result


def split_entry_pipeline_signals(signals_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Keep analytics and consensus signals upstream, but exclude analytics-only
    sources from the live entry scoring funnel.
    """
    def _column_as_series(frame: pd.DataFrame, column_name: str, default_value):
        if column_name not in frame.columns:
            return pd.Series([default_value] * len(frame), index=frame.index)

        raw = frame.loc[:, column_name]
        if isinstance(raw, pd.DataFrame):
            if raw.empty:
                return pd.Series([default_value] * len(frame), index=frame.index)
            series = raw.apply(
                lambda row: next((value for value in row if not pd.isna(value)), default_value),
                axis=1,
            )
        elif isinstance(raw, pd.Series):
            series = raw
        else:
            try:
                series = pd.Series(raw, index=frame.index)
            except Exception:
                values = list(raw) if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)) else [raw] * len(frame)
                series = pd.Series(values)

        series = series.reset_index(drop=True)
        if len(series) < len(frame):
            series = series.reindex(range(len(frame)), fill_value=default_value)
        elif len(series) > len(frame):
            series = series.iloc[: len(frame)]
        series.index = frame.index
        return series

    def _stringify_series(series: pd.Series, default_value: str = "", *, upper: bool = False, lower: bool = False) -> pd.Series:
        def _coerce(value):
            if pd.isna(value):
                text = default_value
            else:
                text = str(value)
            text = text.strip()
            if lower:
                return text.lower()
            if upper:
                return text.upper()
            return text

        return series.map(_coerce)

    def _boolify_series(series: pd.Series, default_value: bool = False) -> pd.Series:
        def _coerce(value):
            if isinstance(value, bool):
                return value
            if pd.isna(value):
                return default_value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off", "", "nan", "none", "null"}:
                return False
            return bool(value)

        return series.map(_coerce).astype(bool)

    if signals_df is None or signals_df.empty:
        return signals_df, {"dropped_rows": 0, "dropped_global_btc_scan": 0, "dropped_stale_wallet_entries": 0, "dropped_analytics_only": 0}
    if "signal_source" not in signals_df.columns:
        signals_df = signals_df.copy()
        analytics_only = _boolify_series(_column_as_series(signals_df, "analytics_only", False), default_value=False)
        if not analytics_only.any():
            return signals_df, {"dropped_rows": 0, "dropped_global_btc_scan": 0, "dropped_stale_wallet_entries": 0, "dropped_analytics_only": 0}
        filtered = signals_df.loc[~analytics_only].reset_index(drop=True)
        return filtered, {
            "dropped_rows": int(analytics_only.sum()),
            "dropped_global_btc_scan": 0,
            "dropped_stale_wallet_entries": 0,
            "dropped_analytics_only": int(analytics_only.sum()),
        }

    work = signals_df.copy()
    signal_source = _stringify_series(_column_as_series(work, "signal_source", ""), default_value="", lower=True)
    analytics_only_mask = signal_source.eq("global_btc_scan")
    generic_analytics_only_mask = _boolify_series(_column_as_series(work, "analytics_only", False), default_value=False)

    if "entry_intent" in work.columns:
        entry_intent = _stringify_series(_column_as_series(work, "entry_intent", ""), default_value="", upper=True)
    else:
        entry_intent = pd.Series([""] * len(work), index=work.index)
    if "source_wallet_fresh" in work.columns:
        source_wallet_fresh = _boolify_series(_column_as_series(work, "source_wallet_fresh", False), default_value=False)
    else:
        source_wallet_fresh = pd.Series([True] * len(work), index=work.index)
    stale_wallet_entry_mask = entry_intent.eq("OPEN_LONG") & ~source_wallet_fresh

    drop_mask = analytics_only_mask | stale_wallet_entry_mask | generic_analytics_only_mask
    dropped_rows = int(drop_mask.sum())
    filtered = work.loc[~drop_mask].reset_index(drop=True)
    return filtered, {
        "dropped_rows": dropped_rows,
        "dropped_global_btc_scan": int(analytics_only_mask.sum()),
        "dropped_stale_wallet_entries": int(stale_wallet_entry_mask.sum()),
        "dropped_analytics_only": int(generic_analytics_only_mask.sum()),
    }


def should_soften_wallet_state_conflict(signal_row) -> bool:
    entry_intent = str(signal_row.get("entry_intent", "") or "").upper()
    if entry_intent != "OPEN_LONG":
        return False
    if not bool(signal_row.get("wallet_conflict_with_stronger", False)):
        return False
    position_event = str(signal_row.get("source_wallet_position_event", "") or "").upper()
    if position_event != "SCALE_IN":
        return False
    raw_reason = str(signal_row.get("wallet_state_gate_reason") or "").strip().replace("|", ",")
    reason_tokens = {
        token.strip()
        for token in raw_reason.split(",")
        if token and token.strip()
    }
    return bool(reason_tokens) and reason_tokens == {"conflict_with_stronger_wallet"}


def main_loop():
    """The continuous autonomous loop (research + paper-trading mode)."""
    global _shutdown_requested
    _shutdown_requested = False
    # Register graceful shutdown handlers for SIGINT and SIGTERM
    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _request_shutdown)
    logging.info("Initializing LIVE PolyMarket Supervisor...")

    # ---- Trading mode selection (1-4) ----
    trading_mode_id = select_trading_mode(default_mode=2)
    active_preset = apply_preset(trading_mode_id)
    preset_name = active_preset.get("name", "Unknown") if active_preset else "Unknown"
    logging.info(
        "Trading mode: %d (%s) | MaxRisk=%.0f%% | Reserve=%.0f%% | HardMax=$%.0f | MaxPositions=%d",
        trading_mode_id,
        preset_name,
        TradingConfig.MAX_RISK_PER_TRADE_PCT * 100,
        TradingConfig.CAPITAL_RESERVE_PCT * 100,
        TradingConfig.HARD_MAX_BET_USDC,
        TradingConfig.MAX_CONCURRENT_POSITIONS,
    )
    logging.info(
        "Governor confidence thresholds: level1=%s level2=%s",
        os.getenv("GOV_LEVEL1_MIN_ENTRY_CONFIDENCE", "0.15"),
        os.getenv("GOV_LEVEL2_MIN_ENTRY_CONFIDENCE", "0.10"),
    )

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

    btc_brain_context = resolve_brain_context("btc", shared_logs_dir="logs", shared_weights_dir="weights")
    weather_brain_context = resolve_brain_context("weather_temperature", shared_logs_dir="logs", shared_weights_dir="weights")

    feature_builder = FeatureBuilder()
    signal_engine = SignalEngine()
    model_inference = ModelInference(brain_context=btc_brain_context)
    stage1_inference = Stage1Inference(brain_context=btc_brain_context)
    stage2_inference = Stage2TemporalInference(brain_context=btc_brain_context)
    hybrid_scorer = Stage3HybridScorer()
    order_flow_analyzer = OrderFlowAnalyzer(min_usd_volume=500.0, volume_imbalance_threshold=0.75, min_trades_count=3)
    technical_analyzer = TechnicalAnalyzer()
    btc_forecast_model = BTCForecastModel()  # single 15m fallback
    btc_mtf_forecaster = BTCMultiTimeframeForecaster()  # multi-timeframe (15m/1h/4h)
    btc_forecast_evaluator = BTCForecastEvaluator()  # walk-forward live eval
    sentiment_analyzer = SentimentAnalyzer()
    macro_analyzer = MacroAnalyzer()
    onchain_analyzer = OnChainAnalyzer()
    weather_temperature_strategy = WeatherTemperatureStrategy(logs_dir="logs")
    entry_min_score = _env_float("ENTRY_MIN_SCORE", 0.04)
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
    active_position_poll_seconds = _env_float("ACTIVE_POSITION_POLL_SECONDS", 5.0)
    idle_poll_seconds = _env_float("IDLE_POLL_SECONDS", 15.0)
    entry_freeze_poll_seconds = _env_float(
        "ENTRY_FREEZE_POLL_SECONDS",
        min(idle_poll_seconds, 10.0),
    )
    error_backoff_seconds = _env_float("ERROR_BACKOFF_SECONDS", 15.0)
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
    logging.info(
        "Loop cadence configured: active_poll=%.1fs idle_poll=%.1fs freeze_poll=%.1fs error_backoff=%.1fs",
        active_position_poll_seconds,
        idle_poll_seconds,
        entry_freeze_poll_seconds,
        error_backoff_seconds,
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
    performance_governor = PerformanceGovernor(logs_dir="logs")
    lifecycle_auditor = TradeLifecycleAuditor(logs_dir="logs")
    benchmark_strategy = BenchmarkStrategy(logs_dir="logs")
    btc_trade_feedback = BTCTradeFeedback(logs_dir="logs")
    latest_open_positions_snapshot_for_shutdown = pd.DataFrame()
    weather_watchlist_warned = False

    def _flush_runtime_state(reason="shutdown"):
        nonlocal latest_open_positions_snapshot_for_shutdown
        persisted_open_positions = 0
        try:
            if trading_mode == "live":
                snapshot_df = (
                    latest_open_positions_snapshot_for_shutdown.copy()
                    if latest_open_positions_snapshot_for_shutdown is not None
                    and hasattr(latest_open_positions_snapshot_for_shutdown, "empty")
                    and not latest_open_positions_snapshot_for_shutdown.empty
                    else pd.DataFrame()
                )
                if snapshot_df.empty:
                    try:
                        snapshot_df = live_position_book.get_open_positions()
                    except Exception as exc:
                        logging.warning("Shutdown snapshot reload failed: %s", exc)
                        snapshot_df = pd.DataFrame()
                if snapshot_df is not None and not snapshot_df.empty:
                    trade_manager.persist_open_positions(reconciled_positions_df=snapshot_df)
                    persisted_open_positions = len(snapshot_df.index)
                else:
                    trade_manager.persist_open_positions()
                    persisted_open_positions = len(trade_manager.get_open_positions())
            else:
                trade_manager.persist_open_positions()
                persisted_open_positions = len(trade_manager.get_open_positions())
        except Exception as exc:
            logging.warning("Shutdown persistence failed for open positions: %s", exc)

        try:
            sync_ops_state_to_db("logs")
        except Exception as exc:
            logging.warning("Shutdown ops state sync failed: %s", exc)

        try:
            autonomous_monitor.write_heartbeat(
                "supervisor",
                status="ok",
                message="shutdown_state_persisted",
                extra={"reason": reason, "open_positions": persisted_open_positions},
            )
        except Exception as exc:
            logging.warning("Shutdown heartbeat write failed: %s", exc)
    def _resolve_brain_context_for_family(market_family: str | None):
        family_text = str(market_family or "").strip().lower()
        if family_text.startswith("weather_temperature"):
            return weather_brain_context
        return btc_brain_context

    def _get_active_model_version(market_family: str = "btc"):
        context = _resolve_brain_context_for_family(market_family)
        table = ModelRegistry(brain_context=context).comparison_table()
        if table.empty:
            return "weather_v1" if context.market_family.startswith("weather_temperature") else ""
        work = table.copy()
        if "market_family" in work.columns:
            work = work[work["market_family"].fillna("").astype(str).str.startswith(context.market_family)].copy()
        if work.empty:
            return "weather_v1" if context.market_family.startswith("weather_temperature") else ""
        champions = work[work.get("is_champion", pd.Series(dtype=bool)) == True].copy()
        if champions.empty:
            champions = work[
                work.get("promotion_status", pd.Series(dtype=str)).fillna("").astype(str).str.lower() == "promoted"
            ].copy()
        latest = champions.iloc[-1].to_dict() if not champions.empty else work.iloc[-1].to_dict()
        version = str(latest.get("run_id") or latest.get("registered_at") or "").strip()
        if version:
            return version
        return "weather_v1" if context.market_family.startswith("weather_temperature") else ""

    def _runtime_identity_from_row(signal_row):
        row_dict = signal_row.to_dict() if hasattr(signal_row, "to_dict") else dict(signal_row or {})
        context = _resolve_brain_context_for_family(row_dict.get("market_family"))
        if context.market_family.startswith("weather_temperature"):
            active_model_group = str(row_dict.get("active_model_group", "") or "weather_temperature_brain_hybrid")
            active_model_kind = str(row_dict.get("active_model_kind", "") or "weather_temperature_hybrid")
            active_regime = str(row_dict.get("active_regime", "") or "forecast_driven")
        else:
            active_model_group = str(row_dict.get("active_model_group", "") or "btc_brain_runtime_stack")
            active_model_kind = str(
                row_dict.get("active_model_kind", "")
                or row_dict.get("btc_market_regime_primary_model", "")
                or "hybrid_stack"
            )
            active_regime = str(
                row_dict.get("active_regime", "")
                or row_dict.get("btc_market_regime_label", "")
                or "calm"
            )
        return active_runtime_identity(
            context,
            active_model_group=active_model_group,
            active_model_kind=active_model_kind,
            active_regime=active_regime,
        )

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
        model_inference = ModelInference(brain_context=btc_brain_context)
        stage1_inference = Stage1Inference(brain_context=btc_brain_context)
        stage2_inference = Stage2TemporalInference(brain_context=btc_brain_context)

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
        market = str(signal_row.get("market_title") or signal_row.get("market") or "").strip()
        outcome_side = str(signal_row.get("outcome_side") or signal_row.get("side") or "").strip()
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

    def _inject_always_on_signal(signals_df: pd.DataFrame, markets_df: pd.DataFrame, btc_context: dict | None = None):
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

        # --- Side selection: BTC forecast → leaderboard consensus → price fallback ---
        btc_ctx = btc_context or {}
        btc_direction = int(btc_ctx.get("btc_predicted_direction", 0) or 0)
        btc_confidence = float(btc_ctx.get("btc_forecast_confidence", 0.0) or 0.0)
        btc_ready = bool(btc_ctx.get("btc_forecast_ready", False))
        leaderboard_consensus = compute_leaderboard_consensus(out_signals)
        lb_bias = leaderboard_consensus.get("leaderboard_bias", 0)
        lb_agreement = leaderboard_consensus.get("leaderboard_agreement", 0.0)
        lb_n_total = leaderboard_consensus.get("leaderboard_n_total", 0)

        pref_side = str(os.getenv("ALWAYS_ON_MARKET_SIDE", "AUTO") or "AUTO").strip().upper()
        side_source = "env_override"
        if pref_side not in {"YES", "NO"}:
            if btc_ready and btc_direction != 0 and btc_confidence >= 0.52:
                # PRIMARY: use the ML BTC forecast to pick the side
                pref_side = "YES" if btc_direction == 1 else "NO"
                side_source = "btc_forecast"
                # Check if leaderboard traders agree or disagree
                if lb_n_total >= 3 and lb_agreement >= 0.60:
                    lb_side = "YES" if lb_bias == 1 else "NO"
                    if lb_side == pref_side:
                        # Leaderboard confirms → boost confidence
                        btc_confidence = min(1.0, btc_confidence + 0.05)
                        side_source = "btc_forecast+leaderboard_confirm"
                        logging.info(
                            "Always-on side: BTC forecast %s confirmed by leaderboard (%d/%d traders, %.0f%% agreement). Boosted confidence → %.3f",
                            pref_side, lb_n_total, lb_n_total, lb_agreement * 100, btc_confidence,
                        )
                    else:
                        # Leaderboard disagrees → reduce confidence but still trust ML
                        btc_confidence = max(0.50, btc_confidence - 0.05)
                        side_source = "btc_forecast+leaderboard_disagree"
                        logging.warning(
                            "Always-on side: BTC forecast %s DISAGREES with leaderboard bias %s (%d traders, %.0f%% agreement). Reduced confidence → %.3f",
                            pref_side, lb_side, lb_n_total, lb_agreement * 100, btc_confidence,
                        )
            elif lb_n_total >= 5 and lb_agreement >= 0.65:
                # SECONDARY: no forecast available but strong leaderboard consensus
                pref_side = "YES" if lb_bias == 1 else "NO"
                btc_confidence = lb_agreement * 0.7  # lower confidence since it's not ML-backed
                side_source = "leaderboard_consensus"
                logging.info(
                    "Always-on side: No BTC forecast available. Using leaderboard consensus %s (%d traders, %.0f%% agreement, conf=%.3f)",
                    pref_side, lb_n_total, lb_agreement * 100, btc_confidence,
                )
            else:
                # FALLBACK: neither forecast nor leaderboard consensus → price heuristic
                pref_side = "YES" if yes_price >= 0.5 else "NO"
                side_source = "price_fallback"
                logging.info(
                    "Always-on side: No BTC forecast (ready=%s) and weak leaderboard (%d traders). Falling back to price-based side: %s",
                    btc_ready, lb_n_total, pref_side,
                )
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
            # BTC forecast metadata attached to the signal
            "btc_predicted_direction": btc_direction,
            "btc_forecast_confidence": btc_confidence,
            "btc_forecast_ready": btc_ready,
            "side_source": side_source,
            # Leaderboard consensus metadata
            **{f"ao_{k}": v for k, v in leaderboard_consensus.items()},
        }
        out_signals = pd.concat([out_signals, pd.DataFrame([synthetic_signal])], ignore_index=True)
        logging.info(
            "Always-on signal injected for slug=%s side=%s token=%s price=%.4f source=%s btc_dir=%s btc_conf=%.3f lb_bias=%s lb_n=%d",
            resolved_slug,
            outcome_side,
            str(token_id)[:16],
            signal_price,
            side_source,
            btc_direction,
            btc_confidence,
            lb_bias,
            lb_n_total,
        )
        if resolved_slug and resolved_slug.lower().startswith(always_on_rotate_prefix):
            _persist_always_on_slug(resolved_slug)
        return out_signals, out_markets


    while not _shutdown_requested:
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
                        
                    live_position_book.rebuild_from_db(force=True)
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
            # Also fetch btc-updown rotating markets (they don't appear in bulk API)
            for _updown_prefix in ("btc-updown-5m-", "btc-updown-15m-", "btc-updown-4h-"):
                try:
                    _updown_df = fetch_btc_updown_markets(prefix=_updown_prefix, closed=False)
                    if _updown_df is not None and not _updown_df.empty:
                        if markets_df is None or markets_df.empty:
                            markets_df = _updown_df
                        else:
                            markets_df = pd.concat([markets_df, _updown_df], ignore_index=True)
                            if "market_id" in markets_df.columns:
                                markets_df = markets_df.drop_duplicates(subset=["market_id"], keep="last")
                        logging.info("Added %d %s rotating markets to universe.", len(_updown_df), _updown_prefix)
                except Exception as _updown_exc:
                    logging.warning("Failed to fetch %s rotating markets: %s", _updown_prefix, _updown_exc)
            if os.getenv("ENABLE_WEATHER_TEMPERATURE_STRATEGY", "true").strip().lower() in {"1", "true", "yes", "on"}:
                try:
                    weather_markets_df = fetch_weather_temperature_markets(limit=500, closed=False, max_offset=None)
                    if weather_markets_df is not None and not weather_markets_df.empty:
                        if markets_df is None or markets_df.empty:
                            markets_df = weather_markets_df
                        else:
                            markets_df = pd.concat([markets_df, weather_markets_df], ignore_index=True)
                            dedupe_field = "market_id" if "market_id" in markets_df.columns else "condition_id"
                            if dedupe_field in markets_df.columns:
                                markets_df = markets_df.drop_duplicates(subset=[dedupe_field], keep="last")
                        logging.info("Added %d weather temperature markets to universe.", len(weather_markets_df))
                except Exception as weather_markets_exc:
                    logging.warning("Failed to fetch weather temperature markets: %s", weather_markets_exc)
            autonomous_monitor.write_heartbeat("market_monitor", status="ok", message="markets_fetched", extra={"rows": len(markets_df) if markets_df is not None else 0})
            if markets_df is not None and not markets_df.empty: markets_df = markets_df.loc[:, ~markets_df.columns.duplicated()]
            save_market_snapshot(markets_df)
            # --- PILLARS 1-4: The Unified Macro Footprint ---
            # Run BEFORE signals so BTC forecast is available for side selection
            ta_context = technical_analyzer.analyze()
            sent_context = sentiment_analyzer.analyze()
            mach_context = macro_analyzer.analyze()
            onc_context = onchain_analyzer.analyze()

            macro_context = {**ta_context, **sent_context, **mach_context, **onc_context}

            # --- BTC Price Forecast (Pillar 5) — Multi-Timeframe ---
            macro_context, _btc_pipeline_audit = apply_btc_pipeline(
                macro_context,
                technical_analyzer,
                btc_mtf_forecaster,
                btc_forecast_model,
                btc_forecast_evaluator,
            )

            # --- Gather leaderboard signals (used as extra info, not sole source) ---
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

            if os.getenv("ENABLE_WEATHER_TEMPERATURE_STRATEGY", "true").strip().lower() in {"1", "true", "yes", "on"}:
                try:
                    weather_watchlist_df = weather_temperature_strategy.load_watchlist()
                    if weather_watchlist_df.empty:
                        if not weather_watchlist_warned:
                            logging.warning(
                                "Weather temperature strategy is enabled but no approved weather wallets are configured. "
                                "Populate %s or set WEATHER_APPROVED_WALLETS to enable live weather signals.",
                                weather_temperature_strategy.watchlist_path,
                            )
                            weather_watchlist_warned = True
                        autonomous_monitor.write_heartbeat(
                            "weather_strategy",
                            status="warn",
                            message="weather_watchlist_empty",
                            extra={"watchlist_path": str(weather_temperature_strategy.watchlist_path)},
                        )
                    weather_signals_df = weather_temperature_strategy.build_cycle_signals(markets_df)
                    if weather_signals_df is not None and not weather_signals_df.empty:
                        weather_watchlist_warned = False
                        if signals_df is None or signals_df.empty:
                            signals_df = weather_signals_df
                        else:
                            signals_df = pd.concat([signals_df, weather_signals_df], ignore_index=True)
                        signals_df = _dedupe_signals_df(signals_df)
                        logging.info("Added %d weather temperature wallet-state signals.", len(weather_signals_df))
                        autonomous_monitor.write_heartbeat(
                            "weather_strategy",
                            status="ok",
                            message="weather_signals_built",
                            extra={
                                "watchlist_wallets": int(len(weather_watchlist_df.index)) if weather_watchlist_df is not None else 0,
                                "signal_rows": int(len(weather_signals_df.index)),
                            },
                        )
                except Exception as weather_signal_exc:
                    logging.warning("Weather temperature strategy signal build failed: %s", weather_signal_exc)

            # Inject always-on signal with BTC forecast driving the YES/NO side
            # and leaderboard consensus used as extra confirmation info
            signals_df, markets_df = _inject_always_on_signal(signals_df, markets_df, btc_context=macro_context)
            signals_df = _dedupe_signals_df(signals_df)
            cycle_observed_iso = datetime.now(timezone.utc).isoformat()
            signals_df = _annotate_signal_freshness(signals_df, cycle_observed_iso)

            # --------------------------------------------------
            if not signals_df.empty and macro_context:
                _scalar_types = (int, float, str, bool, type(None))
                _safe_ctx = {}
                for _mk, _mv in macro_context.items():
                    if isinstance(_mv, _scalar_types) or (hasattr(np, "integer") and isinstance(_mv, (np.integer, np.floating, np.bool_))):
                        _safe_ctx[_mk] = _mv
                    else:
                        logging.debug("macro_context key %r has non-scalar type %s — skipped for signal injection.", _mk, type(_mv).__name__)
                if _safe_ctx:
                    signals_df = pd.concat([signals_df, pd.DataFrame({k: [v] * len(signals_df) for k, v in _safe_ctx.items()}, index=signals_df.index)], axis=1)
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
            entry_signals_df, entry_signal_filter_stats = split_entry_pipeline_signals(signals_df)
            if entry_signal_filter_stats.get("dropped_rows", 0) > 0:
                logging.info(
                    "Entry pipeline dropped %d analytics-only scraper signals before feature scoring.",
                    entry_signal_filter_stats.get("dropped_rows", 0),
                )
            autonomous_monitor.write_heartbeat(
                "signal_engine",
                status="ok",
                message="entry_signal_filter_applied",
                extra=entry_signal_filter_stats,
            )

            pre_cycle_positions_df = locals().get("pre_positions_df")
            if pre_cycle_positions_df is None or (hasattr(pre_cycle_positions_df, "empty") and pre_cycle_positions_df.empty):
                pre_cycle_positions_df = pd.DataFrame([trade.__dict__ for trade in trade_manager.active_trades.values()]) if trade_manager.active_trades else pd.DataFrame()
            entry_signals_df, open_position_context = attach_open_position_context(
                entry_signals_df,
                positions_df=pre_cycle_positions_df,
                active_trades=list(trade_manager.active_trades.values()),
            )
            autonomous_monitor.write_heartbeat(
                "signal_engine",
                status="ok",
                message="entry_position_context_applied",
                extra=open_position_context,
            )

            if entry_signals_df is not None and not entry_signals_df.empty:
                entry_signals_df = entry_signals_df.loc[:, ~entry_signals_df.columns.duplicated()].copy()
                if "market_family" not in entry_signals_df.columns:
                    entry_signals_df["market_family"] = ""
                missing_market_family = entry_signals_df["market_family"].astype(str).str.strip().eq("")
                if missing_market_family.any():
                    entry_signals_df.loc[missing_market_family, "market_family"] = entry_signals_df.loc[missing_market_family].apply(
                        lambda row: build_quality_context(row.to_dict()).get("market_family", "other"),
                        axis=1,
                    )

            def _annotate_scored_candidates(frame, *, default_model_family: str, default_model_version: str, brain_context):
                if frame is None or frame.empty:
                    return pd.DataFrame()
                out = frame.loc[:, ~frame.columns.duplicated()].copy()
                if "entry_model_family" not in out.columns:
                    out["entry_model_family"] = default_model_family
                else:
                    out["entry_model_family"] = out["entry_model_family"].replace("", pd.NA).fillna(default_model_family)
                if "entry_model_version" not in out.columns:
                    out["entry_model_version"] = default_model_version
                else:
                    out["entry_model_version"] = out["entry_model_version"].replace("", pd.NA).fillna(default_model_version)
                if "performance_governor_level" not in out.columns:
                    out["performance_governor_level"] = 0
                else:
                    out["performance_governor_level"] = pd.to_numeric(out["performance_governor_level"], errors="coerce").fillna(0).astype(int)
                out["market_family"] = out.apply(lambda row: build_quality_context(row.to_dict()).get("market_family", row.get("market_family", "other")), axis=1)
                out["horizon_bucket"] = out.apply(lambda row: build_quality_context(row.to_dict()).get("horizon_bucket", row.get("horizon_bucket", "unknown")), axis=1)
                out["liquidity_bucket"] = out.apply(lambda row: build_quality_context(row.to_dict()).get("liquidity_bucket", row.get("liquidity_bucket", "unknown")), axis=1)
                out["volatility_bucket"] = out.apply(lambda row: build_quality_context(row.to_dict()).get("volatility_bucket", row.get("volatility_bucket", "unknown")), axis=1)
                out["technical_regime_bucket"] = out.apply(lambda row: build_quality_context(row.to_dict()).get("technical_regime_bucket", row.get("technical_regime_bucket", "neutral")), axis=1)
                out["brain_id"] = brain_context.brain_id
                if brain_context.market_family.startswith("weather_temperature"):
                    out["active_model_group"] = out.get("active_model_group", pd.Series(index=out.index, dtype=object)).replace("", pd.NA).fillna("weather_temperature_brain_hybrid")
                    out["active_model_kind"] = out.get("active_model_kind", pd.Series(index=out.index, dtype=object)).replace("", pd.NA).fillna("weather_temperature_hybrid")
                    out["active_regime"] = out.get("active_regime", pd.Series(index=out.index, dtype=object)).replace("", pd.NA).fillna("forecast_driven")
                else:
                    primary_model = out.get("btc_market_regime_primary_model", pd.Series(index=out.index, dtype=object)).replace("", pd.NA).fillna("hybrid_stack")
                    regime_label = out.get("btc_market_regime_label", pd.Series(index=out.index, dtype=object)).replace("", pd.NA).fillna("calm")
                    out["active_model_group"] = out.get("active_model_group", pd.Series(index=out.index, dtype=object)).replace("", pd.NA).fillna("btc_brain_runtime_stack")
                    out["active_model_kind"] = out.get("active_model_kind", pd.Series(index=out.index, dtype=object)).replace("", pd.NA).fillna(primary_model)
                    out["active_regime"] = out.get("active_regime", pd.Series(index=out.index, dtype=object)).replace("", pd.NA).fillna(regime_label)
                return out.loc[:, ~out.columns.duplicated()].copy()

            features_df = pd.DataFrame()
            inferred_df = pd.DataFrame()
            btc_scored_df = pd.DataFrame()
            weather_scored_df = pd.DataFrame()
            btc_entry_signals_df = pd.DataFrame()
            weather_entry_signals_df = pd.DataFrame()

            if entry_signals_df is not None and not entry_signals_df.empty:
                weather_entry_mask = entry_signals_df["market_family"].astype(str).str.startswith("weather_temperature")
                weather_entry_signals_df = entry_signals_df[weather_entry_mask].copy()
                btc_entry_signals_df = entry_signals_df[~weather_entry_mask].copy()

            if btc_entry_signals_df is not None and not btc_entry_signals_df.empty:
                features_df = feature_builder.build_features(btc_entry_signals_df, markets_df)
                if features_df is not None:
                    features_df = features_df.loc[:, ~features_df.columns.duplicated()].copy()
                if features_df is not None and not features_df.empty:
                    features_df = features_df.loc[:, ~features_df.columns.duplicated()]
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
                elif trading_mode == "live":
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
                inferred_df["legacy_p_tp_before_sl"] = _frame_numeric_series(
                    inferred_df, "p_tp_before_sl", 0.0
                ).fillna(0.0)
                inferred_df["legacy_expected_return"] = clip_expected_return_series(
                    _frame_numeric_series(inferred_df, "expected_return", 0.0).fillna(0.0)
                )
                inferred_df["legacy_edge_score"] = _frame_numeric_series(
                    inferred_df,
                    "edge_score",
                    inferred_df["legacy_p_tp_before_sl"] * inferred_df["legacy_expected_return"],
                ).fillna(0.0)

                inferred_df = stage1_inference.run(inferred_df)
                inferred_df["stage1_p_tp_before_sl"] = _frame_numeric_series(
                    inferred_df, "p_tp_before_sl", 0.0
                ).fillna(0.0)
                inferred_df["stage1_expected_return"] = clip_expected_return_series(
                    _frame_numeric_series(inferred_df, "expected_return", 0.0).fillna(0.0)
                )
                inferred_df["stage1_edge_score"] = _frame_numeric_series(
                    inferred_df,
                    "edge_score",
                    inferred_df["stage1_p_tp_before_sl"] * inferred_df["stage1_expected_return"],
                ).fillna(0.0)
                inferred_df["stage1_lower_confidence_bound"] = _frame_numeric_series(
                    inferred_df,
                    "lower_confidence_bound",
                    inferred_df["stage1_expected_return"],
                ).fillna(inferred_df["stage1_expected_return"])
                inferred_df["stage1_return_std"] = _frame_numeric_series(
                    inferred_df, "return_std", 0.0
                ).fillna(0.0)

                inferred_df = stage2_inference.run(inferred_df)
                inferred_df["temporal_expected_return"] = clip_expected_return_series(
                    _frame_numeric_series(inferred_df, "temporal_expected_return", 0.0).fillna(0.0)
                )
                inferred_df["temporal_edge_score"] = (
                    _frame_numeric_series(inferred_df, "temporal_p_tp_before_sl", 0.0).fillna(0.0)
                    * inferred_df["temporal_expected_return"]
                )
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
                inferred_df = apply_regime_model_blend(inferred_df)
                inferred_df = hybrid_scorer.run(inferred_df)
                if "hybrid_edge" in inferred_df.columns:
                    inferred_df["edge_score"] = inferred_df["hybrid_edge"]
                log_raw_candidates(inferred_df)
                if inferred_df is not None and not inferred_df.empty:
                    inferred_df = inferred_df.loc[:, ~inferred_df.columns.duplicated()]
                btc_scored_df = signal_engine.score_features(inferred_df)
                btc_scored_df = feedback_learner.apply_to_scored_df(btc_scored_df, signal_engine)
                btc_scored_df = _annotate_scored_candidates(
                    btc_scored_df,
                    default_model_family="runtime_live_stack",
                    default_model_version=_get_active_model_version("btc"),
                    brain_context=btc_brain_context,
                )

            if weather_entry_signals_df is not None and not weather_entry_signals_df.empty:
                log_raw_candidates(weather_entry_signals_df)
                weather_scored_df = weather_temperature_strategy.score_candidates(weather_entry_signals_df, markets_df)
                weather_scored_df = _annotate_scored_candidates(
                    weather_scored_df,
                    default_model_family="weather_temperature_hybrid",
                    default_model_version=_get_active_model_version("weather_temperature"),
                    brain_context=weather_brain_context,
                )
                log_raw_candidates(weather_scored_df)

            scored_frames = [
                frame.loc[:, ~frame.columns.duplicated()].copy()
                for frame in (btc_scored_df, weather_scored_df)
                if frame is not None and not frame.empty
            ]
            if scored_frames:
                scored_df = pd.concat(scored_frames, ignore_index=True, sort=False)
                scored_df = scored_df.loc[:, ~scored_df.columns.duplicated()].copy()
            else:
                scored_df = pd.DataFrame()

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

            # 4A. Candidate-entry path for new signals
            current_active_trades = []
            active_trade_keys = set()
            current_open_exposure = 0.0
            current_weather_active_trade_count = 0

            def _refresh_local_active_trade_state():
                nonlocal current_active_trades, active_trade_keys, current_open_exposure, current_weather_active_trade_count
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
                current_weather_active_trade_count = sum(
                    1
                    for trade in current_active_trades
                    if str(getattr(trade, "market_family", "") or "").strip().lower().startswith("weather_temperature")
                )

            _refresh_local_active_trade_state()
            governor_state = performance_governor.evaluate()
            btc_active_model_version = _get_active_model_version("btc")
            weather_active_model_version = _get_active_model_version("weather_temperature")
            try:
                benchmark_strategy.evaluate_cycle(ta_context, governor_state=governor_state)
            except Exception as exc:
                logging.warning("Benchmark/governor cycle failed: %s", exc)

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

            def _entry_balance_sizing_context(available_balance: float) -> dict:
                available_balance = max(0.0, float(available_balance or 0.0))
                reserve_pct = max(0.0, min(0.95, float(getattr(TradingConfig, "CAPITAL_RESERVE_PCT", 0.20))))
                tradable_balance = max(0.0, available_balance - (available_balance * reserve_pct))
                min_bet_usdc = float(getattr(TradingConfig, "MIN_BET_USDC", 1.0))
                configured_min_entry = max(
                    min_bet_usdc,
                    float(getattr(TradingConfig, "MIN_BET_USDC", 1.0)),
                    float(getattr(TradingConfig, "MIN_ENTRY_USDC", getattr(TradingConfig, "MIN_BET_USDC", 1.0))),
                )
                hard_max_bet = max(min_bet_usdc, float(getattr(TradingConfig, "HARD_MAX_BET_USDC", 250.0)))
                max_risk_per_trade_pct = max(0.0, float(getattr(TradingConfig, "MAX_RISK_PER_TRADE_PCT", 0.15)))
                risk_capped_max_entry = min(tradable_balance * max_risk_per_trade_pct, hard_max_bet)
                floor_support_pct = max(0.0, (1.0 - reserve_pct) * max_risk_per_trade_pct)
                min_balance_for_exchange_floor = (
                    (min_bet_usdc / floor_support_pct) if floor_support_pct > 0 else float("inf")
                )
                low_balance_pause = (
                    tradable_balance + 1e-9 < min_bet_usdc
                    or risk_capped_max_entry + 1e-9 < min_bet_usdc
                )
                return {
                    "reserve_pct": reserve_pct,
                    "tradable_balance": tradable_balance,
                    "min_bet_usdc": min_bet_usdc,
                    "configured_min_entry": configured_min_entry,
                    "hard_max_bet": hard_max_bet,
                    "max_risk_per_trade_pct": max_risk_per_trade_pct,
                    "risk_capped_max_entry": risk_capped_max_entry,
                    "min_balance_for_exchange_floor": min_balance_for_exchange_floor,
                    "low_balance_pause": low_balance_pause,
                }

            def _reason_indicates_dead_orderbook(value) -> bool:
                reason = str(value or "").strip().lower()
                if not reason:
                    return False
                return (
                    "orderbook_not_found" in reason
                    or "orderbook_not_available" in reason
                    or ("orderbook" in reason and "does not exist" in reason)
                    or ("404" in reason and "orderbook" in reason)
                    or "no orderbook exists" in reason
                )

            def _mark_dead_orderbook_token(token_id: str, reason: str | None = None, persist_close: bool = True):
                token_id = str(normalize_token_id(token_id) or "")
                if not token_id:
                    return 0
                _ob_no_book_cache[token_id] = time.monotonic()
                try:
                    _orderbook_unavailable_tokens.add(token_id)
                except Exception:
                    pass
                if orderbook_guard is not None and hasattr(orderbook_guard, "_no_book_cache"):
                    try:
                        orderbook_guard._no_book_cache[token_id] = time.monotonic() + _OB_NO_BOOK_TTL
                    except Exception:
                        pass
                if reason:
                    logging.warning(
                        "Dead-orderbook guard: tombstoning token %s (%s).",
                        token_id,
                        reason,
                    )
                closed_rows = 0
                if persist_close and live_position_book is not None:
                    try:
                        closed_rows = int(live_position_book.close_dead_token_positions({token_id}) or 0)
                        if closed_rows > 0:
                            live_position_book.rebuild_from_db(force=True)
                    except Exception as exc:
                        logging.warning("Dead-orderbook safeguard failed for %s: %s", token_id, exc)
                return closed_rows

            def _force_local_dead_orderbook_close(trade, intended_reason: str, exit_result: dict | None = None):
                if trade is None:
                    return False
                if getattr(trade, "state", None) == TradeState.CLOSED:
                    return True
                reference_price = max(
                    float(getattr(trade, "current_price", 0.0) or 0.0),
                    float(getattr(trade, "entry_price", 0.0) or 0.0),
                    0.01,
                )
                _apply_exit_execution_metrics(trade, exit_result or {"status": "dead_orderbook"}, intended_reason, reference_price)
                trade.actual_execution_path = "dead_orderbook_tombstone"
                trade.intended_exit_reason = intended_reason
                trade.close(exit_price=reference_price, reason="external_manual_close", exit_btc_price=btc_live_price)
                trade_manager.persist_closed_trades([trade])
                trade_manager.active_trades.pop(
                    _make_position_key(
                        token_id=trade.token_id,
                        condition_id=trade.condition_id,
                        outcome_side=trade.outcome_side,
                        market=trade.market,
                    ),
                    None,
                )
                return True

            cadence_boost_blockers = []
            if cadence_boost_active:
                governor_level = int(governor_state.get("governor_level", 0) or 0)
                min_profit_factor_for_boost = float(os.getenv("ENTRY_CADENCE_MIN_PROFIT_FACTOR", "1.0") or 1.0)
                if governor_level > 0:
                    cadence_boost_blockers.append(f"governor_level={governor_level}")
                if float(governor_state.get("live_profit_factor", 0.0) or 0.0) < min_profit_factor_for_boost:
                    cadence_boost_blockers.append(
                        f"profit_factor={float(governor_state.get('live_profit_factor', 0.0) or 0.0):.2f}"
                    )
                if pre_cycle_entry_freeze:
                    cadence_boost_blockers.append(f"entry_freeze={pre_cycle_freeze_reason or 'state_mismatch'}")
                if session_kill_switch_active:
                    cadence_boost_blockers.append("session_kill_switch")
                if trading_mode == "live":
                    cadence_balance = _get_entry_available_balance()
                    cadence_sizing = _entry_balance_sizing_context(cadence_balance)
                    if cadence_sizing["low_balance_pause"]:
                        cadence_boost_blockers.append(
                            "low_balance_pause="
                            f"balance${cadence_balance:.2f}<floor_support${cadence_sizing['min_balance_for_exchange_floor']:.2f}"
                        )
                if cadence_boost_blockers:
                    cadence_boost_active = False
                    logging.info(
                        "Trade cadence booster suppressed: last entry %.1f minutes ago, blockers=%s.",
                        last_entry_age_minutes,
                        ", ".join(cadence_boost_blockers),
                    )
                else:
                    logging.info(
                        "Trade cadence booster active: last entry %.1f minutes ago (target %.1f minutes). Relaxing top %d candidate gates this cycle.",
                        last_entry_age_minutes,
                        target_entry_interval_minutes,
                        entry_aggression_top_k,
                    )

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
                if token_id in _orderbook_unavailable_tokens:
                    _mark_dead_orderbook_token(token_id, reason="cached_dead_orderbook", persist_close=True)
                    return {
                        "status": "dead_orderbook",
                        "filled_shares": 0.0,
                        "avg_price": fallback_price,
                        "remaining_exchange_shares": 0.0,
                        "attempts": [{"attempt": 0, "price": fallback_price, "result": "dead_orderbook", "reason": "cached_dead_orderbook"}],
                        "analysis": {},
                        "elapsed_seconds": 0.0,
                        "cancel_count": 0,
                        "partial_fill_ratio": 0.0,
                        "slippage_bps": 0.0,
                    }
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
                        "elapsed_seconds": 0.0,
                        "cancel_count": 0,
                        "partial_fill_ratio": 0.0,
                        "slippage_bps": 0.0,
                    }

                per_attempt_timeout = max(2.0, float(os.getenv("LIVE_EXIT_ATTEMPT_TIMEOUT_SECONDS", "6") or 6))
                ladder_prices, analysis = _build_live_exit_price_ladder(token_id, fallback_price=fallback_price, aggressive=aggressive_exit)
                attempts = []
                filled_total_shares = 0.0
                filled_total_notional = 0.0
                available_before = initial_available_shares
                cancel_count = 0
                started_at = datetime.now(timezone.utc)

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
                        dead_orderbook_reason = (
                            (exit_row or {}).get("reason")
                            or (exit_response or {}).get("reason")
                            or (exit_row or {}).get("orderbook_error")
                            or (exit_response or {}).get("orderbook_error")
                        )
                        if _reason_indicates_dead_orderbook(dead_orderbook_reason):
                            attempt_record["result"] = "dead_orderbook"
                            attempt_record["reason"] = dead_orderbook_reason
                            attempts.append(attempt_record)
                            _mark_dead_orderbook_token(token_id, reason=dead_orderbook_reason, persist_close=True)
                            break
                        attempt_record["result"] = "rejected"
                        attempt_record["reason"] = dead_orderbook_reason
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
                            cancel_count += 1
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
                elapsed_seconds = max(0.0, (datetime.now(timezone.utc) - started_at).total_seconds())
                partial_fill_ratio = float(filled_total_shares / requested_shares) if requested_shares > 1e-9 else 0.0
                slippage_bps = 0.0
                if reference_price and avg_fill_price:
                    slippage_bps = ((float(reference_price) - float(avg_fill_price)) / max(float(reference_price), 1e-9)) * 10_000.0
                return {
                    "status": "filled" if filled_total_shares >= (requested_shares - 1e-6) else ("partial" if filled_total_shares > 0 else "unfilled"),
                    "filled_shares": float(filled_total_shares),
                    "avg_price": float(avg_fill_price),
                    "remaining_exchange_shares": float(remaining_exchange_shares),
                    "attempts": attempts,
                    "analysis": analysis,
                    "elapsed_seconds": float(elapsed_seconds),
                    "cancel_count": int(cancel_count),
                    "partial_fill_ratio": float(partial_fill_ratio),
                    "slippage_bps": float(slippage_bps),
                }

            def _apply_exit_execution_metrics(trade, exit_result, intended_reason: str, reference_price: float):
                if trade is None:
                    return
                trade.intended_exit_reason = intended_reason
                status = str((exit_result or {}).get("status", "unknown") or "unknown")
                trade.actual_execution_path = f"live_exit_{status}"
                trade.exit_fill_latency_seconds = float((exit_result or {}).get("elapsed_seconds", 0.0) or 0.0)
                trade.exit_cancel_count = int((exit_result or {}).get("cancel_count", 0) or 0)
                trade.exit_partial_fill_ratio = float((exit_result or {}).get("partial_fill_ratio", 0.0) or 0.0)
                trade.exit_realized_slippage_bps = float((exit_result or {}).get("slippage_bps", 0.0) or 0.0)

            def _apply_source_wallet_reduce(trade, signal_row: dict, reason: str) -> bool:
                if trade is None:
                    return False
                token_id = str(getattr(trade, "token_id", "") or "")
                current_price = float(getattr(trade, "current_price", 0.0) or getattr(trade, "entry_price", 0.0) or 0.0)
                pre_reduce_shares = max(float(getattr(trade, "shares", 0.0) or 0.0), 0.0)
                if not token_id or pre_reduce_shares <= 1e-6:
                    return False

                try:
                    _ob_exit = orderbook_guard.analyze_book(token_id, depth=5)
                    exit_price = _ob_exit.get("best_bid") or current_price
                except Exception:
                    exit_price = current_price
                exit_price = float(exit_price or 0.0)
                if exit_price <= 0:
                    logging.warning("Source-wallet REDUCE skipped for %s due to invalid exit price.", token_id)
                    return False

                min_reduce_notional = max(
                    float(getattr(TradingConfig, "MIN_BET_USDC", 1.0)),
                    float(getattr(TradingConfig, "MIN_REDUCE_NOTIONAL_USDC", 2.5)),
                )
                min_remainder_notional = max(
                    0.0,
                    float(getattr(TradingConfig, "MIN_POSITION_REMAINDER_USDC", min_reduce_notional)),
                )
                reduce_fraction = resolve_source_wallet_reduce_fraction(signal_row)
                if should_convert_reduce_to_exit(
                    total_shares=pre_reduce_shares,
                    reduce_fraction=reduce_fraction,
                    reference_price=exit_price,
                    min_reduce_notional=min_reduce_notional,
                    min_remainder_notional=min_remainder_notional,
                ):
                    return False

                requested_shares = min(pre_reduce_shares, pre_reduce_shares * reduce_fraction)
                if requested_shares <= 1e-6:
                    return False

                if trading_mode == "live" and order_manager is not None:
                    reduce_result = _execute_live_sell_ladder(
                        token_id=token_id,
                        requested_shares=requested_shares,
                        condition_id=trade.condition_id,
                        outcome_side=trade.outcome_side,
                        reference_price=exit_price,
                        close_reason=reason,
                    )
                    if str(reduce_result.get("status") or "").strip().lower() == "dead_orderbook":
                        logging.warning(
                            "Source-wallet REDUCE converted to local operational close for dead token %s.",
                            token_id,
                        )
                        _force_local_dead_orderbook_close(trade, reason, reduce_result)
                        return True
                    actual_fill_size = min(float(reduce_result.get("filled_shares", 0.0) or 0.0), pre_reduce_shares)
                    if actual_fill_size <= 1e-6:
                        logging.warning(
                            "Source-wallet REDUCE remained unfilled for %s after %s attempt(s).",
                            token_id,
                            len(reduce_result.get("attempts", [])),
                        )
                        return True
                    actual_fill_price = float(reduce_result.get("avg_price", exit_price) or exit_price)
                    _apply_exit_execution_metrics(trade, reduce_result, reason, exit_price)
                    log_live_fill_event(signal_row, actual_fill_price, actual_fill_size, action_type="LIVE_SOURCE_REDUCE")
                    trade.partial_exit(
                        fraction=min(1.0, actual_fill_size / max(pre_reduce_shares, 1e-9)),
                        exit_price=actual_fill_price,
                    )
                    logging.info(
                        "Source-wallet REDUCE filled %.6f/%.6f shares for %s (reason=%s).",
                        actual_fill_size,
                        pre_reduce_shares,
                        token_id,
                        reason,
                    )
                    return True

                trade.actual_execution_path = "paper_source_wallet_reduce"
                trade.intended_exit_reason = reason
                trade.partial_exit(fraction=reduce_fraction, exit_price=exit_price)
                logging.info(
                    "Paper source-wallet REDUCE for %s fraction=%.3f reason=%s",
                    token_id,
                    reduce_fraction,
                    reason,
                )
                return True

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
                runtime_identity = _runtime_identity_from_row(signal_row)
                signal_market_family = str(signal_row.get("market_family", "") or runtime_identity.get("market_family") or "btc")
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
                    "market_family": runtime_identity["market_family"],
                    "brain_id": runtime_identity["brain_id"],
                    "active_model_group": runtime_identity["active_model_group"],
                    "active_model_kind": runtime_identity["active_model_kind"],
                    "active_regime": runtime_identity["active_regime"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                detail_payload = dict(extra or {})
                detail_payload.setdefault("performance_governor_level", int(governor_state.get("governor_level", 0) or 0))
                detail_payload.setdefault("performance_governor_reason", governor_state.get("reason", ""))
                detail_payload.setdefault(
                    "entry_model_version",
                    signal_row.get("entry_model_version")
                    or (weather_active_model_version if signal_market_family.startswith("weather_temperature") else btc_active_model_version),
                )
                detail_payload.setdefault(
                    "entry_model_family",
                    signal_row.get("entry_model_family")
                    or ("weather_temperature_hybrid" if signal_market_family.startswith("weather_temperature") else "runtime_live_stack"),
                )
                # blocker / gate lineage
                detail_payload.setdefault("first_blocker", gate if reject_reason_norm else None)
                detail_payload.setdefault("all_blockers", gate if reject_reason_norm else None)
                detail_payload.setdefault("passed_gates", ",".join(extra.get("_passed_gates", [])) if extra.get("_passed_gates") else None)
                detail_payload.setdefault("active_model_family", detail_payload.get("entry_model_family"))
                detail_payload.setdefault("brain_id", runtime_identity["brain_id"])
                detail_payload.setdefault("active_model_group", runtime_identity["active_model_group"])
                detail_payload.setdefault("active_model_kind", runtime_identity["active_model_kind"])
                detail_payload.setdefault("active_regime", runtime_identity["active_regime"])
                for live_field in (
                    "btc_live_price",
                    "btc_live_index_price",
                    "btc_live_mark_price",
                    "btc_live_return_5m",
                    "btc_live_return_15m",
                    "btc_live_return_1h",
                    "btc_live_bias",
                    "btc_live_confluence",
                    "btc_live_source_quality",
                    "btc_live_source_quality_score",
                    "btc_live_source_divergence_bps",
                    "btc_live_mark_index_basis_bps",
                    "open_positions_count",
                    "open_positions_negotiated_value_total",
                    "open_positions_max_payout_total",
                    "open_positions_current_value_total",
                    "open_positions_unrealized_pnl_total",
                    "open_positions_unrealized_pnl_pct_total",
                    "open_positions_avg_to_now_price_change_pct_mean",
                    "open_positions_avg_to_now_price_change_pct_min",
                    "open_positions_avg_to_now_price_change_pct_max",
                    "open_positions_winner_count",
                    "open_positions_loser_count",
                ):
                    if live_field not in detail_payload:
                        detail_payload[live_field] = signal_row.get(live_field)
                for key, value in signal_row.items():
                    key_text = str(key or "")
                    if key_text.startswith(("weather_", "forecast_")) or key_text in SOURCE_WALLET_LOG_COLUMNS:
                        if key_text not in detail_payload:
                            detail_payload[key_text] = value
                details_json = json.dumps(detail_payload, default=str, separators=(",", ":"))
                try:
                    db.execute(
                        """
                        INSERT INTO candidate_decisions (
                            cycle_id, candidate_id, token_id, condition_id, outcome_side, market,
                            market_slug, trader_wallet, entry_intent, model_action, final_decision,
                            reject_reason, reject_category, gate, confidence, p_tp_before_sl,
                            expected_return, edge_score, calibrated_edge, calibrated_baseline,
                            proposed_size_usdc, final_size_usdc, available_balance, order_id, details_json,
                            market_family, brain_id, active_model_group, active_model_kind, active_regime, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                            payload["market_family"],
                            payload["brain_id"],
                            payload["active_model_group"],
                            payload["active_model_kind"],
                            payload["active_regime"],
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
                extra.setdefault("_passed_gates", _passed_gates)
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
                if _shutdown_requested:
                    break
                s_row = row.to_dict()
                source_wallet_exit_signal = bool(s_row.get("source_wallet_exit_signal", False))
                source_wallet_reduce_signal = bool(s_row.get("source_wallet_reduce_signal", False))
                if str(s_row.get("entry_intent", "")).upper() == "CLOSE_LONG" or source_wallet_exit_signal or source_wallet_reduce_signal:
                    m_key = _trade_key_from_signal(s_row)
                    if m_key in trade_manager.active_trades:
                        _trade = trade_manager.active_trades[m_key]
                        if not source_wallet_signal_matches_trade(s_row, _trade):
                            continue
                        close_reason = "ai_close_long"
                        if source_wallet_reduce_signal:
                            close_reason = "source_wallet_sharp_reduce"
                        elif str(s_row.get("source_wallet_position_event", "")).upper() == "REVERSAL_EXIT":
                            close_reason = "source_wallet_reversal"
                        elif source_wallet_exit_signal:
                            close_reason = "source_wallet_exit"
                        logging.warning(
                            "Source/AI exit received for %s. wallet=%s reason=%s",
                            m_key,
                            s_row.get("trader_wallet"),
                            close_reason,
                        )
                        if source_wallet_reduce_signal:
                            if _apply_source_wallet_reduce(_trade, s_row, close_reason):
                                continue
                        _px = float(getattr(_trade, "current_price", 0.0) or getattr(_trade, "entry_price", 0.0) or 0.0)
                        if _px > 0:
                            _trade.close(exit_price=_px, reason=close_reason, exit_btc_price=btc_live_price)
                        else:
                            _trade.state = TradeState.CLOSED
                            _trade.close_reason = close_reason

            # FIX 1B: Normal entry loop
            governor_top_signal_consumed_count = 0
            for candidate_rank, (_, row) in enumerate(scored_df.iterrows(), start=1):
                if _shutdown_requested:
                    break
                signal_row = row.to_dict()
                if cadence_boost_active and candidate_rank <= entry_aggression_top_k:
                    signal_row = apply_entry_cadence_boost(
                        signal_row,
                        last_entry_age_minutes,
                        target_entry_interval_minutes,
                        candidate_rank,
                    )
                _candidate_stats["candidates_seen"] += 1
                _passed_gates: list[str] = []
                token_id_norm = normalize_token_id(signal_row.get("token_id"))
                token_id = str(token_id_norm or "")
                entry_intent = str(signal_row.get("entry_intent", "OPEN_LONG") or "OPEN_LONG").upper()
                market_key = _trade_key_from_signal(signal_row)
                market_family = str(signal_row.get("market_family", "") or "").strip().lower()

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
                _passed_gates.append("freshness")

                if len(current_active_trades) >= _max_pos:
                    _log_candidate_skip(
                        signal_row,
                        "max_concurrent_positions_reached",
                        gate="capacity",
                        active_positions=len(current_active_trades),
                        max_positions=_max_pos,
                    )
                    continue
                if market_family.startswith("weather_temperature") and current_weather_active_trade_count >= int(getattr(weather_temperature_strategy, "max_concurrent_positions", 6) or 6):
                    _log_candidate_skip(
                        signal_row,
                        "weather_max_concurrent_positions_reached",
                        gate="capacity",
                        active_weather_positions=current_weather_active_trade_count,
                        max_weather_positions=int(getattr(weather_temperature_strategy, "max_concurrent_positions", 6) or 6),
                    )
                    continue
                
                _passed_gates.append("capacity")
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
                _passed_gates.append("identity")
                _passed_gates.append("market_universe")
                wallet_state_gate_pass = bool(signal_row.get("wallet_state_gate_pass", True))
                if entry_intent == "OPEN_LONG" and not wallet_state_gate_pass:
                    if should_soften_wallet_state_conflict(signal_row):
                        signal_row = signal_row.copy()
                        signal_row["wallet_state_gate_soft_override"] = True
                        signal_row["wallet_state_gate_original_pass"] = False
                        signal_row["wallet_state_gate_pass"] = True
                    else:
                        _log_candidate_skip(
                            signal_row,
                            "wallet_state_gate_failed",
                            gate="wallet_state",
                            wallet_state_gate_reason=signal_row.get("wallet_state_gate_reason"),
                            wallet_watchlist_approved=bool(signal_row.get("wallet_watchlist_approved", True)),
                            wallet_quality_score=_safe_float(signal_row.get("wallet_quality_score", 0.0), default=0.0),
                            wallet_fresh=bool(signal_row.get("source_wallet_fresh", False)),
                            wallet_conflict_with_stronger=bool(signal_row.get("wallet_conflict_with_stronger", False)),
                            wallet_agreement_score=_safe_float(signal_row.get("wallet_agreement_score", 0.0), default=0.0),
                            wallet_stronger_conflict_score=_safe_float(signal_row.get("wallet_stronger_conflict_score", 0.0), default=0.0),
                            wallet_support_strength=_safe_float(signal_row.get("wallet_support_strength", 0.0), default=0.0),
                            source_wallet_direction_confidence=_safe_float(signal_row.get("source_wallet_direction_confidence", 0.0), default=0.0),
                            source_wallet_position_event=signal_row.get("source_wallet_position_event"),
                        )
                        continue

                _passed_gates.append("wallet_state")
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
                _passed_gates.append("dedupe")
                _passed_gates.append("liquidity")
                _passed_gates.append("freeze")
                _passed_gates.append("kill_switch")

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
                    signal_market_family = str(signal_row.get("market_family", "") or "btc")
                    signal_runtime_identity = _runtime_identity_from_row(signal_row)
                    signal_row["performance_governor_level"] = int(governor_state.get("governor_level", 0) or 0)
                    signal_row["entry_model_family"] = str(
                        signal_row.get("entry_model_family")
                        or ("weather_temperature_hybrid" if signal_market_family.startswith("weather_temperature") else "runtime_live_stack")
                    )
                    signal_row["entry_model_version"] = str(
                        signal_row.get("entry_model_version")
                        or (weather_active_model_version if signal_market_family.startswith("weather_temperature") else btc_active_model_version)
                    )
                    signal_row["brain_id"] = signal_runtime_identity["brain_id"]
                    signal_row["active_model_group"] = signal_runtime_identity["active_model_group"]
                    signal_row["active_model_kind"] = signal_runtime_identity["active_model_kind"]
                    signal_row["active_regime"] = signal_runtime_identity["active_regime"]
                    signal_quality_context = build_quality_context(
                        {
                            **(signal_row.to_dict() if hasattr(signal_row, "to_dict") else dict(signal_row)),
                            "confidence_at_entry": confidence,
                            "entry_model_family": signal_row["entry_model_family"],
                            "entry_model_version": signal_row["entry_model_version"],
                        }
                    )
                    for field in (
                        "market_family",
                        "horizon_bucket",
                        "liquidity_bucket",
                        "volatility_bucket",
                        "technical_regime_bucket",
                        "entry_context_complete",
                    ):
                        if signal_quality_context.get(field) not in [None, ""]:
                            signal_row[field] = signal_quality_context.get(field)
                    signal_row["signal_label"] = resolve_entry_signal_label(
                        {
                            **(signal_row.to_dict() if hasattr(signal_row, "to_dict") else dict(signal_row)),
                            **signal_quality_context,
                        }
                    )
                    threshold_conflict = find_conflicting_btc_price_threshold_position(
                        signal_row.to_dict() if hasattr(signal_row, "to_dict") else dict(signal_row),
                        current_active_trades,
                    )
                    if threshold_conflict:
                        existing = threshold_conflict["existing"]
                        _log_candidate_skip(
                            signal_row,
                            "btc_threshold_conflict",
                            gate="portfolio_consistency",
                            conflicting_market=existing.get("market") or existing.get("market_title"),
                            conflicting_condition_id=existing.get("condition_id"),
                            conflicting_outcome_side=existing.get("outcome_side"),
                            conflicting_threshold_price=round(float(existing.get("threshold_price", 0.0) or 0.0), 2),
                            candidate_threshold_price=round(float(threshold_conflict["candidate_threshold_price"]), 2),
                            expiry_key=threshold_conflict["expiry_key"],
                        )
                        continue
                    if market_family.startswith("weather_temperature"):
                        weather_conflict = find_conflicting_weather_temperature_position(
                            signal_row.to_dict() if hasattr(signal_row, "to_dict") else dict(signal_row),
                            current_active_trades,
                            cluster_cap=int(getattr(weather_temperature_strategy, "cluster_cap", 1) or 1),
                        )
                        if weather_conflict:
                            conflicting_trade = weather_conflict.get("trade")
                            conflicting_payload = getattr(conflicting_trade, "__dict__", {}) if conflicting_trade is not None else {}
                            _log_candidate_skip(
                                signal_row,
                                str(weather_conflict.get("reason") or "weather_temperature_conflict"),
                                gate="portfolio_consistency",
                                conflict_cluster_key=weather_conflict.get("cluster_key"),
                                conflict_cluster_size=weather_conflict.get("cluster_size"),
                                conflict_cluster_cap=weather_conflict.get("cluster_cap"),
                                conflicting_market=conflicting_payload.get("market") or conflicting_payload.get("market_title"),
                                conflicting_condition_id=conflicting_payload.get("condition_id"),
                                conflicting_outcome_side=conflicting_payload.get("outcome_side"),
                            )
                            continue
                    governor_min_conf = float(governor_state.get("min_confidence", 0.0) or 0.0)
                    if governor_min_conf > 0 and confidence < governor_min_conf:
                        _log_candidate_skip(
                            signal_row,
                            "performance_governor_min_confidence",
                            gate="performance_governor",
                            model_action=action_map.get(action_val, "UNKNOWN"),
                            governor_level=int(governor_state.get("governor_level", 0) or 0),
                            required_confidence=round(governor_min_conf, 4),
                        )
                        continue
                    governor_min_liquidity = float(governor_state.get("min_liquidity_score", 0.0) or 0.0)
                    liquidity_score = _safe_float(
                        signal_row.get("liquidity_score", signal_row.get("liquidity_depth_score", signal_row.get("market_liquidity_score", 1.0))),
                        default=1.0,
                    )
                    governor_liquidity_floor_failed = False
                    if governor_min_liquidity > 0 and liquidity_score < governor_min_liquidity:
                        governor_liquidity_floor_failed = True
                        signal_row["governor_liquidity_floor_failed"] = True
                        signal_row["governor_liquidity_score"] = round(liquidity_score, 4)
                        signal_row["governor_required_liquidity_score"] = round(governor_min_liquidity, 4)
                    allow_top_signal = performance_governor_top_signal_decision(
                        governor_state,
                        governor_top_signal_consumed_count,
                    )
                    if not allow_top_signal:
                        _log_candidate_skip(
                            signal_row,
                            "performance_governor_top_signal_only",
                            gate="performance_governor",
                            model_action=action_map.get(action_val, "UNKNOWN"),
                            governor_level=int(governor_state.get("governor_level", 0) or 0),
                        )
                        continue

                    # ── Get balance (with paper mode fallback) ──
                    if bool(governor_state.get("top_signal_only")):
                        signal_row["signal_label"] = "HIGHEST-RANKED PAPER SIGNAL"
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
                    size_usdc *= float(governor_state.get("size_multiplier", 1.0) or 1.0)
                    if governor_liquidity_floor_failed:
                        size_usdc *= float(governor_state.get("liquidity_size_multiplier", 1.0) or 1.0)
                    sizing_context = _entry_balance_sizing_context(_available_bal)
                    min_bet_usdc = sizing_context["min_bet_usdc"]
                    configured_min_entry = sizing_context["configured_min_entry"]
                    tradable_balance = sizing_context["tradable_balance"]
                    risk_capped_max_entry = sizing_context["risk_capped_max_entry"]
                    low_balance_pause = sizing_context["low_balance_pause"]
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
                    if bool(governor_state.get("force_min_size")):
                        size_usdc = min_bet_usdc
                        effective_min_entry = min_bet_usdc
                    if size_usdc <= 0:
                        logging.info(
                            "MoneyManager: skip trade (balance=$%.2f, conf=%.2f, exposure=$%.2f, low_balance_pause=%s)",
                            _available_bal, confidence, _current_exposure, low_balance_pause,
                        )
                        _log_candidate_skip(
                            signal_row,
                            "low_balance_pause" if low_balance_pause else "min_size",
                            gate="sizing",
                            model_action=action_map.get(action_val, "UNKNOWN"),
                            available_balance=round(_available_bal, 6),
                            current_exposure=round(_current_exposure, 6),
                            tradable_balance=round(tradable_balance, 6),
                            risk_capped_max_entry=round(risk_capped_max_entry, 6),
                            min_balance_for_exchange_floor=round(float(sizing_context["min_balance_for_exchange_floor"]), 6),
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
                        trade.on_signal(signal_row)
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
                        governor_top_signal_consumed_count = performance_governor_consume_top_signal_slot(
                            governor_state,
                            governor_top_signal_consumed_count,
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
                            governor_top_signal_consumed_count = performance_governor_consume_top_signal_slot(
                                governor_state,
                                governor_top_signal_consumed_count,
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
                        portfolio_pressure_penalty=rule_eval.get("portfolio_pressure_penalty") if isinstance(rule_eval, dict) else None,
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
                    except Exception as exc:
                        logging.debug("Orderbook fallback ping failed for %s: %s", _tid, exc)

            # If in live mode, reconcile with exchange before making decisions
            if trading_mode == "live" and execution_client is not None:
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
                    if _shutdown_requested:
                        break
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
                                if str(reduce_result.get("status") or "").strip().lower() == "dead_orderbook":
                                    logging.warning(
                                        "Live REDUCE converted to local operational close for dead token %s.",
                                        token_id,
                                    )
                                    _force_local_dead_orderbook_close(trade, "rl_reduce", reduce_result)
                                    continue
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
                                if str(exit_result.get("status") or "").strip().lower() == "dead_orderbook":
                                    logging.warning(
                                        "Live EXIT converted to local operational close for dead token %s.",
                                        token_id,
                                    )
                                    _force_local_dead_orderbook_close(trade, "rl_exit", exit_result)
                                    continue
                                actual_fill_size = min(float(exit_result.get("filled_shares", 0.0) or 0.0), pre_exit_shares)
                                if actual_fill_size > 1e-6:
                                    actual_fill_price = float(exit_result.get("avg_price", exit_price) or exit_price)
                                    _apply_exit_execution_metrics(trade, exit_result, "rl_exit", exit_price)
                                    log_live_fill_event(pos_dict, actual_fill_price, actual_fill_size, action_type="LIVE_EXIT")
                                    if actual_fill_size >= pre_exit_shares - 1e-6:
                                        trade.close(exit_price=actual_fill_price, reason="rl_exit", exit_btc_price=btc_live_price) # Update TradeLifecycle
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
                            trade.actual_execution_path = "paper_rl_exit"
                            trade.intended_exit_reason = "rl_exit"
                            trade.close(exit_price=trade.current_price, reason="rl_exit", exit_btc_price=btc_live_price) # FIX M2: real reason
                            logging.info("Paper EXIT for %s. Realized PnL: %.2f", token_id, trade.realized_pnl)
                            trade_manager.active_trades.pop(_make_position_key(token_id=trade.token_id, condition_id=trade.condition_id, outcome_side=trade.outcome_side, market=trade.market), None) # Remove from active trades

            # Process any pending exits (e.g., from CLOSE_LONG signals or internal rules)
            try:
                weather_exit_events = weather_temperature_strategy.apply_active_exit_rules(trade_manager, markets_df)
                if weather_exit_events:
                    logging.info(
                        "Weather active-exit rules closed %d trade(s) before generic exit processing.",
                        len(weather_exit_events),
                    )
                    autonomous_monitor.write_heartbeat(
                        "weather_strategy",
                        status="ok",
                        message="weather_active_exit_rules_triggered",
                        extra={"closed_count": len(weather_exit_events)},
                    )
            except Exception as exc:
                logging.warning("Weather active exit rules failed: %s", exc)
                autonomous_monitor.write_heartbeat(
                    "weather_strategy",
                    status="warn",
                    message="weather_active_exit_rules_failed",
                    extra={"error": str(exc)},
                )
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
                technical_context=ta_context,
            )
            if closed_trades:
                logging.info("[%s] Processed %s closed trades.", trading_mode.upper(), len(closed_trades))

                # CLEANED LIVE EXIT & MONEY MANAGER BLOCK
            if trading_mode == "live" and order_manager is not None:
                for ct in closed_trades:
                    if _shutdown_requested:
                        break
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
                            if str(_exit_result.get("status") or "").strip().lower() == "dead_orderbook":
                                logging.warning(
                                    "Rule exit converted to local operational close for dead token %s.",
                                    _ct_token[:16],
                                )
                                _force_local_dead_orderbook_close(
                                    ct,
                                    getattr(ct, "close_reason", None) or "policy_exit",
                                    _exit_result,
                                )
                                continue
                            _filled_ct_shares = min(float(_exit_result.get("filled_shares", 0.0) or 0.0), _pre_ct_shares)
                            if _filled_ct_shares > 1e-6:
                                _actual_exit_price = float(_exit_result.get("avg_price", _exit_p) or _exit_p)
                                _apply_exit_execution_metrics(ct, _exit_result, getattr(ct, "close_reason", None) or "policy_exit", _exit_p)
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
                if _shutdown_requested:
                    break
                # Only record if it wasn't rolled back into OPEN state
                if getattr(ct, 'state', None) == TradeState.CLOSED:
                    if ct.realized_pnl >= 0:
                        _money_mgr.record_win(ct.realized_pnl)
                    else:
                        _money_mgr.record_loss(ct.realized_pnl)
            finalized_closed_trades = [ct for ct in closed_trades if getattr(ct, "state", None) == TradeState.CLOSED]
            closed_trade_feedback_count = feedback_learner.record_closed_trades(finalized_closed_trades)
            try:
                btc_feedback_df = btc_trade_feedback.write_feedback()
                if not btc_feedback_df.empty:
                    btc_fb_stats = btc_trade_feedback.compute_feedback_weights()
                    logging.info(
                        "BTC forecast feedback: accuracy=%.1f%% win_when_correct=%.1f%% win_when_wrong=%.1f%% (n=%d)",
                        btc_fb_stats.get("btc_direction_accuracy", 0) * 100,
                        btc_fb_stats.get("win_rate_when_btc_correct", 0) * 100,
                        btc_fb_stats.get("win_rate_when_btc_wrong", 0) * 100,
                        btc_fb_stats.get("n_trades_with_btc_data", 0),
                    )
            except Exception as exc:
                logging.warning("BTC trade feedback analysis failed: %s", exc)
            try:
                lifecycle_auditor.build_reports()
            except Exception as exc:
                logging.warning("Trade lifecycle audit failed: %s", exc)

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
            latest_open_positions_snapshot_for_shutdown = (
                open_positions_df_for_status.copy()
                if open_positions_df_for_status is not None and hasattr(open_positions_df_for_status, "copy")
                else pd.DataFrame()
            )
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

            sleep_seconds, sleep_reason = choose_cycle_sleep_interval(
                open_positions_count=open_positions_count_for_sleep,
                entry_freeze_active=previous_entry_freeze_active,
                active_position_poll_seconds=active_position_poll_seconds,
                idle_poll_seconds=idle_poll_seconds,
                entry_freeze_poll_seconds=entry_freeze_poll_seconds,
            )
            logging.info(
                "Cycle complete. %s. Sleeping for %.1f seconds...",
                sleep_reason.capitalize(),
                sleep_seconds,
            )
            if _sleep_until_shutdown_or_timeout(sleep_seconds):
                break

        except KeyboardInterrupt:
            _request_shutdown(getattr(signal, "SIGINT", 2), None)
            logging.info("Supervisor halted manually by user.")
            break
        except Exception as e:
            autonomous_monitor.write_failure("supervisor", str(e))
            logging.error(
                "Critical error in main loop: %s. Relaxing for %.1f seconds to respect API limits "
                "(Memory state is completely preserved).",
                e,
                error_backoff_seconds,
            )
            if _sleep_until_shutdown_or_timeout(max(1.0, error_backoff_seconds)):
                break

    _flush_runtime_state(reason="shutdown_signal" if _shutdown_requested else "loop_exit")
    if _shutdown_requested:
        logging.info("Supervisor shutting down gracefully after signal.")


if __name__ == "__main__":
    main_loop()

