import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from leaderboard_scraper import run_scraper_cycle

# Configure logging for zero-intervention monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure logging directory exists
os.makedirs("logs", exist_ok=True)
SUMMARY_FILE = "logs/daily_summary.txt"


def load_brain(model_path="weights/ppo_polytrader"):
    """Loads the trained Reinforcement Learning model."""
    try:
        model = PPO.load(model_path)
        logging.info(f"[+] Successfully loaded RL brain from {model_path}.zip")
        return model
    except Exception as e:
        logging.error(f"[-] Failed to load model from {model_path}. Error: {e}")
        return None


def prepare_observation(signal):
    """
    Converts a raw scraped signal into the 4-value normalized array for the RL Brain.
    Expected State: [trader_win_rate, normalized_trade_size, current_price, time_left]
    """
    price = float(signal.get("price", 0.5))
    obs = np.array([0.75, 0.5, price, 0.9], dtype=np.float32)
    return obs


def execute_paper_trade(action, signal):
    """Simulates a trade fill and logs the hypothetical position."""
    if action == 0:
        logging.info(f"Brain: IGNORE -> Skipping signal from {signal.get('trader_wallet', 'Unknown')[:8]}")
        return

    # Action 1 = Small Trade (10 USDC), Action 2 = Large Trade (50 USDC)
    size = 10 if action == 1 else 50
    side = signal.get("side", "BUY").upper()

    # Simulate slippage (e.g., getting filled 1 cent worse than the signal price)
    signal_price = float(signal.get("price", 0.5))
    fill_price = min(0.99, signal_price + 0.01) if side == "BUY" else max(0.01, signal_price - 0.01)

    trade_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market": signal.get("market_title", "Unknown Market"),
        "wallet_copied": signal.get("trader_wallet", "Unknown")[:8],
        "side": side,
        "signal_price": round(signal_price, 3),
        "fill_price": round(fill_price, 3),
        "size_usdc": size,
        "action_type": "PAPER_TRADE",
    }

    logging.info(
        f"Brain: FOLLOW -> Paper filled {size} USDC on {side} at ${fill_price:.3f} for '{trade_record['market']}'"
    )

    # Write to the daily ledger
    try:
        df = pd.DataFrame([trade_record])
        # Append without headers if file exists, with headers if it doesn't
        df.to_csv(SUMMARY_FILE, mode="a", header=not os.path.exists(SUMMARY_FILE), index=False)
    except Exception as e:
        logging.error(f"[-] Failed to write to {SUMMARY_FILE}: {e}")


def main_loop():
    """The zero-intervention continuous autonomous loop (Simulation Mode)."""
    logging.info("Initializing PAPER-TRADING PolyMarket Supervisor...")

    brain = load_brain()
    if not brain:
        logging.error("Halting execution: Missing trained brain.")
        return

    while True:
        try:
            logging.info("--- Starting Paper-Trading Evaluation Cycle ---")

            # 1. Scrape Alpha Signals
            signals_df = run_scraper_cycle()

            if not signals_df.empty:
                # 2. Evaluate each signal with the RL model
                for index, row in signals_df.iterrows():
                    signal = row.to_dict()
                    obs = prepare_observation(signal)

                    action, _states = brain.predict(obs, deterministic=True)
                    action_val = int(action.item() if hasattr(action, "item") else action[0])

                    # 3. Route to mock execution
                    execute_paper_trade(action_val, signal)
            else:
                logging.info("No actionable signals found on this pass.")

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
