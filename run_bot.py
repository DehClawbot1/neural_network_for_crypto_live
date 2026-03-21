import os
import sys
import logging
from pathlib import Path

from api_setup import validate_environment
from rl_trainer import train_model
from supervisor import main_loop, load_brain
from retrainer import Retrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

WEIGHTS_PATH = Path("weights/ppo_polytrader.zip")


def print_banner():
    print("\n=== NEURAL NETWORK FOR CRYPTO ===")
    print("Mode: PAPER TRADING / REAL-TIME PUBLIC DATA")
    print("This launcher validates the environment, checks model weights, and starts the supervisor.\n")


def ensure_environment():
    print("[1/3] Checking environment...")
    valid = validate_environment()
    if not valid:
        print("[!] Environment template created or configuration needs review.")
        print("    Re-run `python run_bot.py` after confirming your .env values.\n")
        return False

    print("[+] Environment OK\n")
    return True


def ensure_model():
    print("[2/3] Checking RL model weights...")

    if WEIGHTS_PATH.exists():
        print(f"[+] Found model weights: {WEIGHTS_PATH}\n")
        return True

    print("[!] No trained model found.")
    print("[!] Starting a quick bootstrap training run so the supervisor has a model to use...")
    try:
        train_model(timesteps=5000)
    except Exception as exc:
        print(f"[-] Failed to train bootstrap model: {exc}")
        return False

    if WEIGHTS_PATH.exists():
        print(f"[+] Model created successfully: {WEIGHTS_PATH}\n")
        return True

    print("[-] Training completed but weights were not found.")
    return False


def maybe_retrain_before_start():
    print("[2.5/3] Checking whether the model should retrain from accumulated data...")
    retrainer = Retrainer()
    retrained = retrainer.maybe_retrain()
    if retrained:
        print("[+] Retraining triggered before startup. Latest weights refreshed.\n")
    else:
        print("[+] No pre-start retraining needed yet.\n")


def start_supervisor():
    print("[3/3] Starting supervisor...")
    print("[+] Status: RUNNING")
    print("[+] Expected behavior:")
    print("    - fetch public BTC-related market/account activity")
    print("    - rank paper-trading opportunities")
    print("    - simulate paper trades")
    print("    - sleep 60 seconds and repeat")
    print("\n[+] Open the dashboard in another terminal with:")
    print("    streamlit run dashboard.py\n")

    main_loop()


def main():
    print_banner()

    if not ensure_environment():
        sys.exit(1)

    if not ensure_model():
        sys.exit(1)

    # Final quick sanity check before loop
    maybe_retrain_before_start()

    if load_brain() is None:
        print("[-] Model exists but could not be loaded. Aborting.")
        sys.exit(1)

    start_supervisor()


if __name__ == "__main__":
    main()
