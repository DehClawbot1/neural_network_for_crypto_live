import os
import sys
import logging
from pathlib import Path

from api_setup import validate_environment
from rl_trainer import train_model
from supervisor import main_loop, load_brain
from retrainer import Retrainer
from real_pipeline import run_research_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

WEIGHTS_PATH = Path("weights/ppo_polytrader.zip")


def print_banner():
    print("\n=== NEURAL NETWORK FOR CRYPTO ===")
    print("Mode: PAPER TRADING / REAL-TIME PUBLIC DATA")
    print("TRADING_MODE supported here: paper")
    print("This launcher validates the environment, checks model weights, and starts the supervisor.\n")


def ensure_environment():
    print("[1/3] Checking environment...")
    trading_mode = os.getenv("TRADING_MODE", "paper").strip().lower()
    if trading_mode != "paper":
        print(f"[!] Unsupported TRADING_MODE='{trading_mode}'.")
        print("[!] This repository only supports paper mode. Set TRADING_MODE=paper and retry.\n")
        return False

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
    print("[2.5/4] Checking whether the model should retrain from accumulated data...")
    retrainer = Retrainer()
    retrained = retrainer.maybe_retrain()
    if retrained:
        print("[+] Retraining triggered before startup. Latest weights refreshed.\n")
    else:
        print("[+] No pre-start retraining needed yet.\n")


def build_research_artifacts():
    print("[3/4] Building research datasets / supervised artifacts...")
    try:
        run_research_pipeline()
        print("[+] Research pipeline refreshed (historical dataset, targets, eval files).\n")
    except Exception as exc:
        print(f"[!] Research pipeline failed but supervisor can still run: {exc}\n")


def start_supervisor():
    print("[4/4] Starting supervisor...")
    print("[+] Status: RUNNING")
    print("[+] Expected behavior:")
    print("    - fetch public BTC-related market/account activity")
    print("    - rank paper-trading opportunities")
    print("    - simulate paper trades")
    print("    - write execution and episode logs")
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
    build_research_artifacts()

    if load_brain() is None:
        print("[-] Model exists but could not be loaded. Aborting.")
        sys.exit(1)

    start_supervisor()


if __name__ == "__main__":
    main()
