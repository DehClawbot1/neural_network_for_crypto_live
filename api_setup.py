import os
import sys
import logging
import subprocess
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def prompt_runtime_config():
    trading_mode = os.getenv("TRADING_MODE", "").strip().lower()
    interactive = sys.stdin.isatty()

    if not interactive:
        if not trading_mode:
            trading_mode = "paper"
            os.environ["TRADING_MODE"] = trading_mode
        return trading_mode

    if not trading_mode:
        trading_mode = input("TRADING_MODE [paper/live]: ").strip().lower() or "paper"
        os.environ["TRADING_MODE"] = trading_mode

    if not os.getenv("POLYMARKET_PUBLIC_ADDRESS"):
        os.environ["POLYMARKET_PUBLIC_ADDRESS"] = input("POLYMARKET_PUBLIC_ADDRESS (optional for public profile/data endpoints): ").strip()

    if trading_mode == "live":
        if not os.getenv("POLYMARKET_FUNDER"):
            os.environ["POLYMARKET_FUNDER"] = input("POLYMARKET_FUNDER: ").strip()
        if not os.getenv("PRIVATE_KEY"):
            os.environ["PRIVATE_KEY"] = input("PRIVATE_KEY (shown while typing): ").strip()
        if not os.getenv("POLYMARKET_API_KEY"):
            os.environ["POLYMARKET_API_KEY"] = input("POLYMARKET_API_KEY (shown while typing, optional): ").strip()
        if not os.getenv("POLYMARKET_API_SECRET"):
            os.environ["POLYMARKET_API_SECRET"] = input("POLYMARKET_API_SECRET (shown while typing, optional): ").strip()
        if not os.getenv("POLYMARKET_API_PASSPHRASE"):
            os.environ["POLYMARKET_API_PASSPHRASE"] = input("POLYMARKET_API_PASSPHRASE (shown while typing, optional): ").strip()
    return trading_mode


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
            f.write("POLYMARKET_PUBLIC_ADDRESS=\n")
            f.write("# For live-test branch only:\n")
            f.write("# PRIVATE_KEY=\n")
            f.write("# POLYMARKET_FUNDER=\n")
            f.write("# POLYMARKET_API_KEY=\n")
            f.write("# POLYMARKET_API_SECRET=\n")
            f.write("# POLYMARKET_API_PASSPHRASE=\n")
        logging.info("[+] Starter .env template created. Please review variables.")
        return False

    load_dotenv()

    trading_mode = prompt_runtime_config()
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
        api_key = os.getenv("POLYMARKET_API_KEY")
        api_secret = os.getenv("POLYMARKET_API_SECRET")
        api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE")
        if private_key and funder:
            if api_key and api_secret and api_passphrase:
                logging.info("[+] Environment validated for live-test mode with stored L2 API credentials.")
            else:
                logging.warning("[!] Live-test config is missing stored L2 API credentials. The client may derive them at startup as a fallback.")
            return True
        logging.error("[-] Invalid live config. PRIVATE_KEY and POLYMARKET_FUNDER are required.")
        return False

    logging.error("[-] Invalid TRADING_MODE. Use paper or live.")
    return False


if __name__ == "__main__":
    is_valid = validate_environment()
    if is_valid:
        print("\n[+] Environment is valid.")
        if not sys.stdin.isatty():
            print("[+] Non-interactive environment detected. Skipping startup prompts.")
        else:
            start_bot = input("Start run_bot.py now? [y/N]: ").strip().lower()
            start_dashboard = input("Start dashboard.py now? [y/N]: ").strip().lower()

            if start_bot in {"y", "yes"} and start_dashboard in {"y", "yes"}:
                subprocess.run([sys.executable, "run_bot_and_dashboard.py"], check=False)
            elif start_bot in {"y", "yes"}:
                subprocess.run([sys.executable, "run_bot.py"], check=False)
            elif start_dashboard in {"y", "yes"}:
                subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], check=False)
            else:
                print("[+] Setup complete. You may start run_bot.py and dashboard.py manually.")
    else:
        print("\n[-] Validation failed or template generated. Please check your .env file and run again.")

