import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO

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
from position_manager import PositionManager
from model_inference import ModelInference
from stage1_inference import Stage1Inference
from stage2_temporal_inference import Stage2TemporalInference
from stage3_hybrid import Stage3HybridScorer
from strategy_layers import EntryRuleLayer

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
    """Loads the trained Reinforcement Learning model."""
    recurrent_path = "weights/recurrent_ppo_polytrader"
    try:
        if RecurrentPPO is not None and os.path.exists(recurrent_path + ".zip"):
            model = RecurrentPPO.load(recurrent_path)
            logging.info(f"[+] Successfully loaded recurrent RL brain from {recurrent_path}.zip")
            return StatefulRecurrentBrain(model)
        model = PPO.load(model_path)
        logging.info(f"[+] Successfully loaded RL brain from {model_path}.zip")
        return model
    except Exception as e:
        logging.error(f"[-] Failed to load model from {model_path}. Error: {e}")
        return None


def prepare_observation(feature_row, legacy=False):
    """
    Converts grouped feature-engine output into the observation vector for the RL Brain.
    Supports a legacy 4-feature fallback for older saved models.
    """
    if legacy:
        obs = np.array(
            [
                float(feature_row.get("trader_win_rate", 0.5)),
                float(feature_row.get("normalized_trade_size", 0.5)),
                float(feature_row.get("current_price", 0.5)),
                float(feature_row.get("time_left", 0.5)),
            ],
            dtype=np.float32,
        )
        return obs

    obs = np.array(
        [
            float(feature_row.get("trader_win_rate", 0.5)),
            float(feature_row.get("normalized_trade_size", 0.5)),
            float(feature_row.get("current_price", 0.5)),
            float(feature_row.get("time_left", 0.5)),
            float(feature_row.get("liquidity_score", 0.5)),
            float(feature_row.get("volume_score", 0.5)),
            float(feature_row.get("probability_momentum", 0.5)),
            float(feature_row.get("volatility_score", 0.5)),
            float(feature_row.get("whale_pressure", 0.5)),
            float(feature_row.get("market_structure_score", 0.5)),
        ],
        dtype=np.float32,
    )
    return obs


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
        "wallet_copied": str(signal_row.get("trader_wallet", "Unknown"))[:8],
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


def choose_action(signal_row, entry_rule: EntryRuleLayer, brain=None):
    if brain is not None:
        try:
            obs = prepare_observation(signal_row)
            action, _ = brain.predict(obs, deterministic=True)
            action_val = int(action.item() if hasattr(action, "item") else action[0])
            return action_val
        except Exception:
            pass

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


def execute_paper_trade(action, signal_row, fill_price=None):
    """Simulates a trade fill and logs the hypothetical position."""
    if action == 0:
        logging.info(f"Brain: IGNORE -> Skipping signal from {signal_row.get('trader_wallet', 'Unknown')[:8]}")
        return

    size = 10 if action == 1 else 50
    outcome_side = str(signal_row.get("outcome_side", signal_row.get("side", "UNKNOWN"))).upper()
    signal_price = float(signal_row.get("current_price", signal_row.get("price", 0.5)))
    fill_price = quote_entry_price(signal_row) if fill_price is None else fill_price

    trade_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market": signal_row.get("market_title", "Unknown Market"),
        "wallet_copied": str(signal_row.get("trader_wallet", "Unknown"))[:8],
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


def main_loop():
    """The continuous autonomous loop (research + paper-trading mode)."""
    logging.info("Initializing PAPER-TRADING PolyMarket Supervisor...")

    brain = load_brain()
    if not brain:
        logging.warning("RL brain unavailable. Continuing with supervised-first ranking path.")

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
    position_manager = PositionManager()
    trading_mode = os.getenv("TRADING_MODE", "paper").strip().lower()
    execution_client = ExecutionClient() if trading_mode == "live" else None
    autonomous_monitor = AutonomousMonitor()
    retrainer = Retrainer()
    previous_markets_df = None

    while True:
        try:
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
                logging.info("No actionable signals found on this pass.")
                logging.info("Cycle complete. Sleeping for 60 seconds...")
                time.sleep(60)
                continue

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

            if shadow_logger is not None and not scored_df.empty:
                scored_view = normalize_dataframe_columns(scored_df)
                for _, row in scored_view.head(5).iterrows():
                    try:
                        shadow_logger.log_entry(row.to_dict(), pd.DataFrame([row]))
                    except Exception as exc:
                        logging.warning("Shadow logging failed for %s: %s", row.get("market_title", row.get("market")), exc)

            if scored_df.empty:
                logging.info("No scored signals generated on this pass.")
                logging.info("Cycle complete. Sleeping for 60 seconds...")
                time.sleep(60)
                continue

            sort_cols = [c for c in ["risk_adjusted_ev", "entry_ev", "execution_quality_score", "edge_score", "p_tp_before_sl", "confidence", "normalized_trade_size"] if c in scored_df.columns]
            scored_df = scored_df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))

            # Suppress duplicate entries / repeated token spam
            if "token_id" in scored_df.columns:
                scored_df = scored_df.drop_duplicates(subset=["token_id"], keep="first")

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

            # 4A. Candidate-entry path for tokens without open positions
            open_positions_df = position_manager.get_open_positions()
            open_token_ids = set(open_positions_df.get("token_id", pd.Series(dtype=str)).dropna().astype(str).tolist()) if not open_positions_df.empty else set()
            for _, row in scored_df.iterrows():
                signal_row = row.to_dict()
                token_id = str(signal_row.get("token_id", "") or "")
                entry_intent = str(signal_row.get("entry_intent", "OPEN_LONG") or "OPEN_LONG").upper()

                if not token_id:
                    logging.warning("Skipping signal with missing token_id: %s", signal_row.get("market_title", signal_row.get("market", "unknown_market")))
                    continue

                if entry_intent == "CLOSE_LONG":
                    if token_id and token_id in open_token_ids:
                        matching = open_positions_df[open_positions_df.get("token_id", pd.Series(dtype=str)).astype(str) == token_id]
                        if not matching.empty:
                            position_manager.close_position(matching.iloc[0].to_dict(), reason="whale_sell_exit")
                    continue

                if token_id and token_id in open_token_ids:
                    continue

                action_val = choose_action(signal_row, entry_rule, brain=brain)
                if action_val not in [0, 1, 2]:
                    action_val = 0

                if action_val != 0:
                    size = 10 if action_val == 1 else 50
                    fill_price = quote_entry_price(signal_row)
                    if fill_price is None or pd.isna(fill_price):
                        logging.warning("Skipping signal with missing fill price for token_id=%s", token_id)
                        continue
                    if trading_mode == "live" and execution_client is not None:
                        execution_client.create_and_post_order(
                            token_id=signal_row.get("token_id"),
                            price=fill_price,
                            size=size,
                            side=signal_row.get("order_side", "BUY"),
                        )
                    else:
                        execute_paper_trade(action_val, signal_row, fill_price=fill_price)
                    position_manager.open_position(signal_row, size_usdc=size, fill_price=fill_price)

            # 4B. Open-position management path for hold / reduce / exit
            open_positions_df = position_manager.update_mark_to_market(scored_df)
            if not open_positions_df.empty and brain is not None:
                for _, pos_row in open_positions_df.iterrows():
                    obs = prepare_position_observation(pos_row.to_dict())
                    try:
                        pos_action, _ = brain.predict(obs, deterministic=True)
                        pos_action_val = int(pos_action.item() if hasattr(pos_action, "item") else pos_action[0])
                    except Exception:
                        pos_action_val = 3

                    if pos_action_val == 4:
                        position_manager.reduce_position(pos_row.to_dict(), fraction=0.5)
                    elif pos_action_val == 5:
                        position_manager.close_position(pos_row.to_dict(), reason="rl_exit")

            # 5. Phase 2 analytics outputs
            trades_df = safe_read_csv(EXECUTION_FILE)
            trader_signals_df = scored_df.rename(columns={"trader_wallet": "wallet_copied", "market_title": "market"})
            trader_analytics.write(trader_signals_df, trades_df)
            backtester.write(trader_signals_df)
            dataset_builder.write()

            alerts_df = safe_read_csv("logs/alerts.csv")
            position_manager.update_mark_to_market(scored_df)
            position_manager.apply_exit_rules(alerts_df)
            open_positions_df = position_manager.get_open_positions()
            autonomous_monitor.write_heartbeat("position_manager", status="ok", message="positions_updated", extra={"open_positions": len(open_positions_df) if open_positions_df is not None else 0})
            autonomous_monitor.write_status(trader_signals_df, trades_df, alerts_df, open_positions_df)
            retrainer.maybe_retrain()
            autonomous_monitor.write_heartbeat("retrainer", status="ok", message="retrain_checked")

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

