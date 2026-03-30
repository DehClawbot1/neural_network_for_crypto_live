from trade_lifecycle import TradeState
import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

from leaderboard_scraper import run_scraper_cycle
from market_monitor import fetch_btc_markets, save_market_snapshot, fetch_markets_by_slugs
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
from ops_state_sync import sync_ops_state_to_db
from stage3_hybrid import Stage3HybridScorer
from config import TradingConfig
from strategy_layers import EntryRuleLayer
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
from orderbook_guard import OrderBookGuard

# Configure logging for zero-intervention monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure logging directory exists
os.makedirs("logs", exist_ok=True)
EXECUTION_FILE = "logs/execution_log.csv"
SIGNALS_FILE = "logs/signals.csv"
RAW_CANDIDATES_FILE = "logs/raw_candidates.csv"
MARKETS_FILE = "logs/markets.csv"


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
    }
    append_csv_record(SIGNALS_FILE, record)


def choose_action(signal_row, entry_rule: EntryRuleLayer, entry_brain=None, legacy_brain=None):
    action_val = 0
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

    # FIX H7: Apply entry rule as VETO even when RL says enter
    if action_val in (1, 2) and not entry_rule.should_enter(signal_row):
        logging.info("Entry rule vetoed RL action=%d for %s", action_val,
                     signal_row.get("market_title", signal_row.get("market", "unknown")))
        return 0
    if action_val != 0:
        return action_val

    # FIX V5: If RL models are loaded and they predicted 0, respect the VETO.
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



# --- IP SAFEGUARD INTERCEPTOR ---
# Prevents Polymarket IP Bans during 5-second fast-polling
_orig_run_scraper_cycle = run_scraper_cycle
_orig_fetch_btc_markets = fetch_btc_markets
_last_research_time = 0
_cached_open_markets = pd.DataFrame()

def safe_run_scraper_cycle(*args, **kwargs):
    global _last_research_time
    import time
    import pandas as pd
    if time.time() - _last_research_time < 55:
        return pd.DataFrame()
    return _orig_run_scraper_cycle(*args, **kwargs)

def safe_fetch_btc_markets(closed=False, max_offset=0, *args, **kwargs):
    global _last_research_time, _cached_open_markets
    import time
    import pandas as pd
    if not closed and time.time() - _last_research_time < 55:
        return _cached_open_markets
    
    res = _orig_fetch_btc_markets(closed=closed, max_offset=max_offset, *args, **kwargs)
    if not closed:
        _cached_open_markets = res
        _last_research_time = time.time()
    return res

run_scraper_cycle = safe_run_scraper_cycle
fetch_btc_markets = safe_fetch_btc_markets
# --------------------------------

def main_loop():
    """The continuous autonomous loop (research + paper-trading mode)."""
    logging.info("Initializing LIVE PolyMarket Supervisor...")

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
    entry_rule = EntryRuleLayer()
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
    orderbook_guard = OrderBookGuard(max_spread=0.20, min_bid_depth=1, min_ask_depth=1)
    _money_mgr = MoneyManager()
    autonomous_monitor = AutonomousMonitor()
    retrainer = Retrainer()
    previous_markets_df = None
    previous_entry_freeze_active = False
    previous_entry_freeze_reason = None
    try:
        shadow_purgatory = ShadowPurgatory()
    except Exception as exc:
        shadow_purgatory = None
        logging.warning("ShadowPurgatory unavailable at startup: %s", exc)
    db = Database()


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


    while True:
        try:
            if trading_mode == "live" and order_manager is not None and hasattr(order_manager, "risk") and hasattr(order_manager.risk, "reset_failed_orders"):
                order_manager.risk.reset_failed_orders()
            if trading_mode == "live" and reconciliation_service is not None:
                try:
                    sync_summary = reconciliation_service.sync_orders_and_fills()
                    logging.info("Exchange reconciliation synced orders=%s fills=%s", sync_summary.get("orders", 0), sync_summary.get("fills", 0))
                except Exception as exc:
                    logging.warning("Exchange reconciliation failed: %s", exc)
            logging.info("--- Starting Research + Paper-Trading Evaluation Cycle ---")

            # 1. Gather public market context + public wallet activity
            open_markets = fetch_btc_markets(closed=False)
            closed_markets = fetch_btc_markets(closed=True, max_offset=500)
            if not open_markets.empty and not closed_markets.empty:
                markets_df = pd.concat([open_markets, closed_markets], ignore_index=True).drop_duplicates(subset=["market_id"])
            else:
                markets_df = open_markets if not open_markets.empty else closed_markets
            autonomous_monitor.write_heartbeat("market_monitor", status="ok", message="markets_fetched", extra={"rows": len(markets_df) if markets_df is not None else 0})
            save_market_snapshot(markets_df)
            signals_df = run_scraper_cycle()
            autonomous_monitor.write_heartbeat("signal_engine", status="ok", message="signals_scraped", extra={"rows": len(signals_df) if signals_df is not None else 0})

            if signals_df is not None and not signals_df.empty and "market_slug" in signals_df.columns:
                scraped_slugs = set(signals_df["market_slug"].dropna().astype(str).unique())
                scraped_slugs.discard("")
                known_slugs = set(markets_df["slug"].dropna().astype(str).unique()) if markets_df is not None and not markets_df.empty and "slug" in markets_df.columns else set()
                missing_slugs = scraped_slugs - known_slugs
                if missing_slugs:
                    logging.info("Universe Gap: %s slugs missing. Synchronizing...", len(missing_slugs))
                    missing_df = fetch_markets_by_slugs(list(missing_slugs))
                    if missing_df is not None and not missing_df.empty:
                        markets_df = pd.concat([markets_df, missing_df], ignore_index=True).drop_duplicates(subset=["slug"])
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
            features_df = feature_builder.build_features(signals_df, markets_df)
            log_raw_candidates(features_df)
            inferred_df = model_inference.run(features_df)
            inferred_df = stage1_inference.run(inferred_df)
            inferred_df = stage2_inference.run(inferred_df)
            if "temporal_p_tp_before_sl" in inferred_df.columns:
                inferred_df["p_tp_before_sl"] = inferred_df[["p_tp_before_sl", "temporal_p_tp_before_sl"]].max(axis=1)
            if "temporal_expected_return" in inferred_df.columns:
                inferred_df["expected_return"] = inferred_df[["expected_return", "temporal_expected_return"]].mean(axis=1)
                inferred_df["edge_score"] = inferred_df["p_tp_before_sl"].astype(float) * inferred_df["expected_return"].astype(float)
            inferred_df = hybrid_scorer.run(inferred_df)
            if "hybrid_edge" in inferred_df.columns:
                inferred_df["edge_score"] = inferred_df["hybrid_edge"]
            log_raw_candidates(inferred_df)
            scored_df = signal_engine.score_features(inferred_df)

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
                    f"confidence={float(ranked_row.get('confidence', 0.0)):.2f} | "
                    f"market={ranked_row.get('market_title')} | "
                    f"side={ranked_row.get('side')}"
                )
                logging.info(" - %s", summary_line)
                print(summary_line)
                print(f"   reason: {ranked_row.get('reason')}")
                log_ranked_signal(ranked_row.to_dict())
            print("======================================\n")

            # 4A. Candidate-entry path for new signals
            current_active_trades = trade_manager.get_open_positions()
            active_trade_keys = {
                _trade_key_from_signal({
                    "token_id": trade.token_id,
                    "condition_id": trade.condition_id,
                    "market_title": trade.market,
                    "outcome_side": trade.outcome_side,
                })
                for trade in current_active_trades if trade.market or trade.token_id
            }
            live_entry_freeze = False
            freeze_reason = None
            freeze_detail = {}
            if trading_mode == "live" and live_position_book is not None:
                live_position_book.rebuild_from_db()
                reconciled_positions_df = live_position_book.get_enriched_open_positions(scored_df=scored_df)
                if mismatch_detector is not None:
                    mismatch_summary = mismatch_detector.detect(current_active_trades, reconciled_positions_df)
                    if mismatch_summary.get("freeze_entries"):
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
                        logging.info("Entry freeze cleared (previous_reason=%s)", previous_entry_freeze_reason or "state_mismatch")

            previous_entry_freeze_active = live_entry_freeze
            previous_entry_freeze_reason = freeze_reason

            _max_pos = getattr(TradingConfig, 'MAX_CONCURRENT_POSITIONS', 5)
            
            # FIX 1A: Process all AI exits FIRST, ignoring max_pos restrictions
            for _, row in scored_df.iterrows():
                s_row = row.to_dict()
                if str(s_row.get("entry_intent", "")).upper() == "CLOSE_LONG":
                    m_key = _trade_key_from_signal(s_row)
                    if m_key in trade_manager.active_trades:
                        logging.warning("AI Veto! CLOSE_LONG received for %s. Forcing exit.", m_key)

                        trade_manager.active_trades[m_key].state = TradeState.CLOSED
                        trade_manager.active_trades[m_key].close_reason = "ai_close_long"

            # FIX 1B: Normal entry loop
            for _, row in scored_df.iterrows():
                if len(trade_manager.active_trades) >= _max_pos:
                    logging.info("Max concurrent positions reached (%d/%d). Skipping remaining entries.",
                                 len(trade_manager.active_trades), _max_pos)
                    break
                    
                signal_row = row.to_dict()
                token_id = str(signal_row.get("token_id", "") or "")
                entry_intent = str(signal_row.get("entry_intent", "OPEN_LONG") or "OPEN_LONG").upper()
                market_key = _trade_key_from_signal(signal_row)
                
                if not market_key or not token_id or entry_intent == "CLOSE_LONG":
                    continue

                # FIX: Check dynamic active_trades to prevent Triple-Buy duplicates in the same loop
                if market_key in active_trade_keys or market_key in trade_manager.active_trades:
                    logging.info("Prevented duplicate entry: Trade already open for %s.", market_key)
                    continue
                if trading_mode == "live" and live_entry_freeze:
                    logging.warning(
                        "Skipping new live entry for %s because entry freeze is active (reason=%s, local_count=%s, live_count=%s)",
                        token_id,
                        freeze_reason or "state_mismatch",
                        freeze_detail.get("local_count"),
                        freeze_detail.get("live_count"),
                    )
                    continue

                action_val = choose_action(
                    signal_row,
                    entry_rule,
                    entry_brain=entry_brain,
                    legacy_brain=legacy_brain,
                )
                if action_val not in [0, 1, 2]:
                    action_val = 0

                action_map = {0: "IGNORE", 1: "SMALL_BUY", 2: "LARGE_BUY"}
                try:
                    db.execute(
                        "INSERT INTO model_decisions (token_id, condition_id, outcome_side, model_name, score, action, feature_snapshot, model_artifact, normalization_artifact) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            token_id,
                            signal_row.get("condition_id"),
                            signal_row.get("outcome_side", signal_row.get("side")),
                            entry_model_name,
                            float(signal_row.get("confidence", 0.0) or 0.0),
                            action_map.get(action_val, "UNKNOWN"),
                            build_feature_snapshot(signal_row),
                            entry_model_artifact,
                            entry_norm_artifact,
                        ),
                    )
                except Exception as exc:
                    logging.warning("Model decision logging failed for %s: %s", token_id, exc)

                if action_val != 0:
                    # ── BUGFIX: Define confidence BEFORE using it ──
                    confidence = float(signal_row.get("confidence", 0.0) or 0.0)

                    # ── Get balance (with paper mode fallback) ──
                    _available_bal = 0.0
                    if order_manager is not None:
                        try:
                            _available_bal, _ = order_manager._get_available_balance(asset_type="COLLATERAL")
                        except Exception:
                            pass
                    if _available_bal <= 0 and execution_client is not None:
                        try:
                            _available_bal = execution_client.get_available_balance(asset_type="COLLATERAL")
                        except Exception:
                            pass
                    # ── BUGFIX: Paper mode needs simulated balance ──
                    if _available_bal <= 0 and trading_mode == "paper":
                        _sim_bal = float(os.getenv("SIMULATED_STARTING_BALANCE", "1000"))
                        _open_cost = sum(float(getattr(t, 'size_usdc', 0) or 0) for t in trade_manager.get_open_positions())
                        _available_bal = max(0.0, _sim_bal - _open_cost)

                    _current_exposure = sum(
                        float(getattr(t, 'size_usdc', 0) or 0)
                        for t in trade_manager.get_open_positions()
                    )
                    size_usdc = _money_mgr.calculate_bet_size(
                        available_balance=_available_bal,
                        confidence=confidence,
                        current_exposure=_current_exposure,
                    )
                    if size_usdc <= 0:
                        logging.info(
                            "MoneyManager: skip trade (balance=$%.2f, conf=%.2f, exposure=$%.2f)",
                            _available_bal, confidence, _current_exposure,
                        )
                        continue
                    # ── Order book guard: check spread/depth before entry ──
                    try:
                        ob_check = orderbook_guard.check_before_entry(
                            token_id=token_id, side="BUY", intended_size_usdc=size_usdc,
                        )
                        if not ob_check["tradable"]:
                            logging.info("OrderBookGuard BLOCKED %s: %s", token_id[:16], ob_check["reason"])
                            continue
                        fill_price = ob_check.get("recommended_entry_price") or quote_entry_price(signal_row)
                        for _w in ob_check.get("warnings", []):
                            logging.warning("OrderBookGuard %s: %s", token_id[:16], _w)
                    except Exception as _ob_exc:
                        logging.warning("OrderBookGuard failed for %s: %s (using fallback price)", token_id[:16], _ob_exc)
                        fill_price = quote_entry_price(signal_row)
                    
                    if fill_price is None or pd.isna(fill_price):
                        logging.warning("Skipping signal with missing fill price for token_id=%s", token_id)
                        continue


                    if trading_mode == "live" and order_manager is not None:
                        # For live mode: submit order first, register trade only on fill
                        from pnl_engine import PNLEngine as _PNLEngine
                        _order_shares = _PNLEngine.shares_from_capital(size_usdc, fill_price)
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
                        entry_order_id = (entry_row or {}).get("order_id") or (entry_response or {}).get("orderID") or (entry_response or {}).get("order_id") or (entry_response or {}).get("id")
                        if not entry_order_id:
                            logging.info("Live entry rejected/skipped for token_id=%s reason=%s", token_id, (entry_row or {}).get("reason"))
                            # If order is rejected/skipped, we should potentially revert trade creation in TradeManager
                            # For simplicity, we'll let process_exits handle cleanup later if trade is not filled
                            continue
                        
                        fill_result = order_manager.wait_for_fill(entry_order_id)
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
                                continue
                        
                        fill_payload = fill_result.get("response") or {}
                        actual_fill_price = float(fill_payload.get("price", fill_price) or fill_price)
                        actual_fill_size = float(fill_payload.get("size", _order_shares) or _order_shares)
                        
                        log_live_fill_event(signal_row, actual_fill_price, size_usdc, action_type="LIVE_TRADE")
                        # Register trade AFTER confirmed fill (not before)
                        trade = TradeLifecycle(
                            market=signal_row.get("market_title", signal_row.get("market", "Unknown")),
                            token_id=token_id,
                            condition_id=signal_row.get("condition_id"),
                            outcome_side=signal_row.get("outcome_side", signal_row.get("side", "YES")),
                        )
                        trade.enter(size_usdc=size_usdc, entry_price=actual_fill_price)
                        trade.shares = actual_fill_size
                        trade_manager.active_trades[market_key] = trade
                        logging.info("Live trade filled for %s at %s. Shares: %s", token_id, actual_fill_price, actual_fill_size)
                    else:
                        # Paper trade: register in TradeManager (which logs to execution_log.csv)
                        # FIX C7: Don't also call execute_paper_trade — it creates duplicate log entries
                        trade = trade_manager.handle_signal(signal_row=pd.Series(signal_row), confidence=confidence, size_usdc=size_usdc, entry_price_override=fill_price)
                        if trade is not None:
                            logging.info(
                                "Brain: FOLLOW -> Paper filled %s USDC on %s at $%.3f for '%s' | label=%s confidence=%.2f",
                                size_usdc,
                                signal_row.get("outcome_side", "?"),
                                fill_price,
                                signal_row.get("market_title", "Unknown"),
                                signal_row.get("signal_label", "UNKNOWN"),
                                confidence,
                            )

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
                    trade_manager.reconcile_live_positions(reconciled_positions_df=reconciled_positions_df)
                except Exception as exc:
                    logging.warning("Live trade reconciliation failed before management decisions: %s", exc)
            current_open_trades = trade_manager.get_open_positions()
            if current_open_trades and (position_brain is not None or legacy_brain is not None):
                for trade in current_open_trades:
                    pos_dict = {
                        "token_id": trade.token_id,
                        "condition_id": trade.condition_id,
                        "outcome_side": trade.outcome_side,
                        "entry_price": trade.entry_price,
                        "current_price": trade.current_price,
                        "size_usdc": trade.size_usdc,
                        "shares": trade.shares,
                        "market_title": trade.market,
                        "confidence": 0.5 # Placeholder or fetch from signal
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
                        trade.has_been_reduced = True
                        if trading_mode == "live" and order_manager is not None:
                            try:
                                _ob_exit = orderbook_guard.analyze_book(token_id, depth=5)
                                if _ob_exit.get("best_bid") is not None:
                                    exit_price = _ob_exit["best_bid"]
                                else:
                                    exit_price = quote_exit_price(pos_dict)
                            except Exception:
                                exit_price = quote_exit_price(pos_dict)
                            exit_shares = trade.shares * 0.5
                            if exit_price is not None and exit_shares > 0:
                                reduce_row, reduce_response = order_manager.submit_entry(
                                    token_id=token_id,
                                    price=exit_price,
                                    size=exit_shares,
                                    side="SELL",
                                    condition_id=trade.condition_id,
                                    outcome_side=trade.outcome_side,
                                )
                                reduce_order_id = (reduce_row or {}).get("order_id") or (reduce_response or {}).get("orderID") or (reduce_response or {}).get("order_id") or (reduce_response or {}).get("id")
                                if reduce_order_id:
                                    fill_result = order_manager.wait_for_fill(reduce_order_id)
                                    if fill_result.get("filled"):
                                        fill_payload = fill_result.get("response") or {}
                                        actual_fill_price = float(fill_payload.get("price", exit_price) or exit_price)
                                        actual_fill_size = float(fill_payload.get("size", exit_shares) or exit_shares)
                                        log_live_fill_event(pos_dict, actual_fill_price, actual_fill_size, action_type="LIVE_REDUCE")
                                        trade.partial_exit(fraction=actual_fill_size / trade.shares, exit_price=actual_fill_price) # Update TradeLifecycle
                                    else:
                                        logging.warning("Live REDUCE not filled for %s; attempting cancel for order_id=%s", token_id, reduce_order_id)
                                        try:
                                            order_manager.cancel_stale_order(reduce_order_id)
                                        except Exception as exc:
                                            logging.warning("Failed to cancel stale live reduce order %s: %s", reduce_order_id, exc)
                            else:
                                logging.warning("Live REDUCE skipped for %s due to invalid exit price/size", token_id)
                        else:
                            trade.partial_exit(fraction=0.5, exit_price=trade.current_price) # Paper reduce
                            logging.info("Paper REDUCE for %s. Current PnL: %.2f (reason=rl_reduce)", token_id, trade.realized_pnl)
                    elif pos_action_val == 5: # EXIT
                        logging.info("[%s] Exiting position for %s", trading_mode.upper(), token_id)
                        if trading_mode == "live" and order_manager is not None:
                            try:
                                _ob_exit = orderbook_guard.analyze_book(token_id, depth=5)
                                if _ob_exit.get("best_bid") is not None:
                                    exit_price = _ob_exit["best_bid"]
                                else:
                                    exit_price = quote_exit_price(pos_dict)
                            except Exception:
                                exit_price = quote_exit_price(pos_dict)
                            exit_shares = trade.shares
                            if exit_price is not None and exit_shares > 0:
                                exit_row, exit_response = order_manager.submit_entry(
                                    token_id=token_id,
                                    price=exit_price,
                                    size=exit_shares,
                                    side="SELL",
                                    condition_id=trade.condition_id,
                                    outcome_side=trade.outcome_side,
                                )
                                exit_order_id = (exit_row or {}).get("order_id") or (exit_response or {}).get("orderID") or (exit_response or {}).get("order_id") or (exit_response or {}).get("id")
                                if exit_order_id:
                                    fill_result = order_manager.wait_for_fill(exit_order_id)
                                    if fill_result.get("filled"):
                                        fill_payload = fill_result.get("response") or {}
                                        actual_fill_price = float(fill_payload.get("price", exit_price) or exit_price)
                                        actual_fill_size = float(fill_payload.get("size", exit_shares) or exit_shares)
                                        log_live_fill_event(pos_dict, actual_fill_price, actual_fill_size, action_type="LIVE_EXIT")
                                        trade.close(exit_price=actual_fill_price) # Update TradeLifecycle
                                        trade_manager.active_trades.pop(_make_position_key(token_id=trade.token_id, condition_id=trade.condition_id, outcome_side=trade.outcome_side, market=trade.market), None) # Remove from active trades
                                    else:
                                        logging.warning("Live EXIT not filled for %s; attempting cancel for order_id=%s", token_id, exit_order_id)
                                        try:
                                            order_manager.cancel_stale_order(exit_order_id)
                                        except Exception as exc:
                                            logging.warning("Failed to cancel stale live exit order %s: %s", exit_order_id, exc)
                            else:
                                logging.warning("Live EXIT skipped for %s due to invalid exit price/size", token_id)
                        else:
                            trade.close(exit_price=trade.current_price, reason="rl_exit") # FIX M2: real reason
                            logging.info("Paper EXIT for %s. Realized PnL: %.2f", token_id, trade.realized_pnl)
                            trade_manager.active_trades.pop(_make_position_key(token_id=trade.token_id, condition_id=trade.condition_id, outcome_side=trade.outcome_side, market=trade.market), None) # Remove from active trades

            # Process any pending exits (e.g., from CLOSE_LONG signals or internal rules)
            from datetime import timezone
            closed_trades = trade_manager.process_exits(datetime.now(timezone.utc))
            if closed_trades:
                logging.info("[%s] Processed %s closed trades.", trading_mode.upper(), len(closed_trades))

                # CLEANED LIVE EXIT & MONEY MANAGER BLOCK
            if trading_mode == "live" and order_manager is not None:
                for ct in closed_trades:
                    if ct.shares <= 0: continue
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
                            _exit_row, _exit_resp = order_manager.submit_entry(
                                token_id=_ct_token, price=_exit_p, size=ct.shares, side="SELL", 
                                condition_id=ct.condition_id, outcome_side=ct.outcome_side)
                                
                            _exit_oid = (_exit_row or {}).get("order_id")
                            if _exit_oid:
                                _fill = order_manager.wait_for_fill(_exit_oid, timeout_seconds=15)
                                if _fill.get("filled"):
                                    log_live_fill_event(
                                        {"token_id": _ct_token, "market_title": ct.market, "outcome_side": ct.outcome_side, "current_price": _exit_p},
                                        _exit_p, ct.shares, action_type=f"LIVE_EXIT_{ct.close_reason}")
                                else:
                                    try: order_manager.cancel_stale_order(_exit_oid)
                                    except Exception: pass
                                    # FIX: RESTORE ZOMBIE TRADE TO MEMORY IF SELL FAILS

                                    trade_manager.active_trades[f"{ct.market}-{ct.outcome_side}"] = ct
                                    ct.state = TradeState.OPEN
                                    ct.close_reason = None
                                    logging.warning("Live SELL failed for %s. Restored to active tracking.", _ct_token[:16])
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

            # 5. Phase 2 analytics outputs
            trades_df = safe_read_csv(EXECUTION_FILE)
            trader_signals_df = scored_df.rename(columns={"trader_wallet": "wallet_copied", "market_title": "market"})
            trader_analytics.write(trader_signals_df, trades_df)
            backtester.write(trader_signals_df)
            dataset_builder.write()

            alerts_df = safe_read_csv("logs/alerts.csv")

            if trading_mode == "live" and live_position_book is not None and live_pnl is not None:
                live_position_book.rebuild_from_db()
                open_positions_df_for_status = live_position_book.get_enriched_open_positions(scored_df=scored_df)
                open_positions_df_for_status = live_pnl.enrich_positions(open_positions_df_for_status)
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
            trade_manager.persist_open_positions()
            try:
                sync_ops_state_to_db("logs")
            except Exception as exc:
                logging.warning("Ops state sync to DB failed: %s", exc)
            retrainer.maybe_retrain()
            autonomous_monitor.write_heartbeat("retrainer", status="ok", message="retrain_checked")

            if len(trade_manager.get_open_positions()) > 0:
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
            logging.error(f"Critical error in main loop: {e}. Auto-restarting in 60 seconds...")
            time.sleep(60)


if __name__ == "__main__":
    main_loop()

