import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO

from leaderboard_scraper import run_scraper_cycle
from market_monitor import fetch_btc_markets, save_market_snapshot
from feature_builder import FeatureBuilder
from signal_engine import SignalEngine
from whale_tracker import WhaleTracker
from alerts_engine import AlertsEngine
from trader_analytics import TraderAnalytics
from historical_dataset_builder import HistoricalDatasetBuilder
from backtester import StrategyBacktester
from simulation_engine import SimulationEngine
from autonomous_monitor import AutonomousMonitor
from retrainer import Retrainer
from position_manager import PositionManager

# Configure logging for zero-intervention monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure logging directory exists
os.makedirs("logs", exist_ok=True)
SUMMARY_FILE = "logs/daily_summary.txt"
SIGNALS_FILE = "logs/signals.csv"
MARKETS_FILE = "logs/markets.csv"


def load_brain(model_path="weights/ppo_polytrader"):
    """Loads the trained Reinforcement Learning model."""
    try:
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


def append_csv_record(path, record):
    df = pd.DataFrame([record])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def log_ranked_signal(signal_row):
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market": signal_row.get("market_title", "Unknown Market"),
        "wallet_copied": str(signal_row.get("trader_wallet", "Unknown"))[:8],
        "side": signal_row.get("side", "UNKNOWN"),
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


def execute_paper_trade(action, signal_row):
    """Simulates a trade fill and logs the hypothetical position."""
    if action == 0:
        logging.info(f"Brain: IGNORE -> Skipping signal from {signal_row.get('trader_wallet', 'Unknown')[:8]}")
        return

    # Action 1 = Small Trade (10 USDC), Action 2 = Large Trade (50 USDC)
    size = 10 if action == 1 else 50
    side = str(signal_row.get("side", "BUY")).upper()

    # Simulate slippage (e.g., getting filled 1 cent worse than the signal price)
    signal_price = float(signal_row.get("current_price", signal_row.get("price", 0.5)))
    fill_price = min(0.99, signal_price + 0.01) if side == "BUY" else max(0.01, signal_price - 0.01)

    trade_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market": signal_row.get("market_title", "Unknown Market"),
        "wallet_copied": str(signal_row.get("trader_wallet", "Unknown"))[:8],
        "side": side,
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
        side,
        fill_price,
        trade_record["market"],
        trade_record["signal_label"],
        float(trade_record["confidence"]),
    )

    try:
        append_csv_record(SUMMARY_FILE, trade_record)
    except Exception as e:
        logging.error(f"[-] Failed to write to {SUMMARY_FILE}: {e}")


def main_loop():
    """The continuous autonomous loop (research + paper-trading mode)."""
    logging.info("Initializing PAPER-TRADING PolyMarket Supervisor...")

    brain = load_brain()
    if not brain:
        logging.error("Halting execution: Missing trained brain.")
        return

    feature_builder = FeatureBuilder()
    signal_engine = SignalEngine()
    whale_tracker = WhaleTracker()
    alerts_engine = AlertsEngine()
    trader_analytics = TraderAnalytics()
    dataset_builder = HistoricalDatasetBuilder()
    backtester = StrategyBacktester()
    simulation_engine = SimulationEngine()
    position_manager = PositionManager()
    autonomous_monitor = AutonomousMonitor()
    retrainer = Retrainer()
    previous_markets_df = None

    while True:
        try:
            logging.info("--- Starting Research + Paper-Trading Evaluation Cycle ---")

            # 1. Gather public market context + public wallet activity
            markets_df = fetch_btc_markets()
            save_market_snapshot(markets_df)
            signals_df = run_scraper_cycle()

            if signals_df.empty:
                logging.info("No actionable signals found on this pass.")
                logging.info("Cycle complete. Sleeping for 60 seconds...")
                time.sleep(60)
                continue

            # 2. Build whale summaries and detect alerts from public data
            whale_tracker.write_summary(signals_df)
            alerts_engine.process_alerts(markets_df, previous_markets_df, signals_df)
            previous_markets_df = markets_df.copy()

            # 3. Build features and score paper-trading opportunities
            features_df = feature_builder.build_features(signals_df, markets_df)
            scored_df = signal_engine.score_features(features_df)

            if scored_df.empty:
                logging.info("No scored signals generated on this pass.")
                logging.info("Cycle complete. Sleeping for 60 seconds...")
                time.sleep(60)
                continue

            scored_df = scored_df.sort_values(by=["confidence", "normalized_trade_size"], ascending=[False, False])

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

            # 4. Evaluate scored rows with RL model, then simulate paper execution only
            for _, row in scored_df.iterrows():
                signal_row = row.to_dict()
                obs = prepare_observation(signal_row)

                try:
                    action, _states = brain.predict(obs, deterministic=True)
                except Exception:
                    # Backward-compatible fallback for older 4-feature models
                    legacy_obs = prepare_observation(signal_row, legacy=True)
                    action, _states = brain.predict(legacy_obs, deterministic=True)

                action_val = int(action.item() if hasattr(action, "item") else action[0])

                execute_paper_trade(action_val, signal_row)

                if action_val != 0:
                    size = 10 if action_val == 1 else 50
                    fill_price = min(0.99, float(signal_row.get("current_price", 0.5)) + 0.01)
                    simulation_engine.open_position(signal_row, size_usdc=size, fill_price=fill_price)
                    position_manager.open_position(signal_row, size_usdc=size, fill_price=fill_price)

            # 5. Phase 2 analytics outputs
            trades_df = pd.read_csv(SUMMARY_FILE) if os.path.exists(SUMMARY_FILE) else pd.DataFrame()
            trader_signals_df = scored_df.rename(columns={"trader_wallet": "wallet_copied", "market_title": "market"})
            trader_analytics.write(trader_signals_df, trades_df)
            backtester.write(trader_signals_df)
            dataset_builder.write()

            alerts_df = pd.read_csv("logs/alerts.csv") if os.path.exists("logs/alerts.csv") else pd.DataFrame()
            position_manager.update_mark_to_market(scored_df)
            position_manager.apply_exit_rules(alerts_df)
            open_positions_df = position_manager.get_open_positions()
            autonomous_monitor.write_status(trader_signals_df, trades_df, alerts_df, open_positions_df)
            retrainer.maybe_retrain()

            logging.info("Cycle complete. Sleeping for 60 seconds...")
            time.sleep(60)

        except KeyboardInterrupt:
            logging.info("Supervisor halted manually by user.")
            break
        except Exception as e:
            logging.error(f"Critical error in main loop: {e}. Auto-restarting in 60 seconds...")
            time.sleep(60)


if __name__ == "__main__":
    main_loop()
