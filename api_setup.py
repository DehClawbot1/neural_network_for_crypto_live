import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def validate_environment():
    """
    Validates the presence and structure of the .env file for paper or live-test mode.
    """
    logging.info("Validating local environment setup...")

    env_path = ".env"

    if not os.path.exists(env_path):
        logging.warning("[-] .env file not found. Generating a starter template...")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("# PolyMarket Bot Configuration\n")
            f.write("TRADING_MODE=paper\n")
            f.write("SIMULATED_STARTING_BALANCE=1000\n")
            f.write("MAX_RISK_PER_TRADE=50\n")
            f.write("# For live-test branch only:\n")
            f.write("# PRIVATE_KEY=\n")
            f.write("# POLYMARKET_FUNDER=\n")
        logging.info("[+] Starter .env template created. Please review variables.")
        return False

    load_dotenv()

    trading_mode = os.getenv("TRADING_MODE", "paper").strip().lower()
    balance = os.getenv("SIMULATED_STARTING_BALANCE")

    if trading_mode == "paper":
        if balance:
            logging.info(f"[+] Environment validated. Running in paper mode with ${balance} simulated balance.")
            return True
        logging.error("[-] Invalid paper config. Please set SIMULATED_STARTING_BALANCE.")
        return False

    if trading_mode == "live":
        private_key = os.getenv("PRIVATE_KEY")
        funder = os.getenv("POLYMARKET_FUNDER")
        if private_key and funder:
            logging.info("[+] Environment validated for live-test mode.")
            return True
        logging.error("[-] Invalid live config. PRIVATE_KEY and POLYMARKET_FUNDER are required.")
        return False

    logging.error("[-] Invalid TRADING_MODE. Use paper or live.")
    return False


if __name__ == "__main__":
    is_valid = validate_environment()
    if is_valid:
        print("\n[+] Environment is valid. You may start run_bot.py.")
    else:
        print("\n[-] Validation failed or template generated. Please check your .env file and run again.")
