import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv
from api_setup import validate_environment
from rl_trainer import train_model
from supervisor import main_loop, load_brain
import supervisor as supervisor_module
from supervisor_ui_patch import apply_supervisor_ui_patch
from retrainer import Retrainer
from real_pipeline import run_research_pipeline
from execution_client import ExecutionClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

WEIGHTS_PATH = Path("weights/ppo_polytrader.zip")
LOGS_DIR = Path("logs")
RESEARCH_ARTIFACTS = [
    LOGS_DIR / "historical_dataset.csv",
    LOGS_DIR / "contract_targets.csv",
    LOGS_DIR / "wallet_alpha_history.csv",
    LOGS_DIR / "supervised_eval.csv",
    LOGS_DIR / "time_split_eval.csv",
    LOGS_DIR / "path_replay_backtest.csv",
]


def print_banner():
    print("\n=== NEURAL NETWORK FOR CRYPTO ===")
    print("Mode: LIVE-TEST / REAL-TIME DATA")
    print("TRADING_MODE required here: live")
    print("This launcher validates the environment, checks model weights, and starts the supervisor.\n")


def ensure_environment():
    print("[1/3] Checking environment...")
    trading_mode = os.getenv("TRADING_MODE", "").strip().lower()
    if trading_mode != "live":
        print(f"[!] Invalid TRADING_MODE='{trading_mode or 'missing'}'.")
        print("[!] This launcher now requires TRADING_MODE=live.\n")
        return False

    valid = validate_environment()
    if not valid:
        print("[!] Environment template created or configuration needs review.")
        print("    Re-run `python run_bot.py` after confirming your .env values.\n")
        return False

    print("[+] Environment OK\n")
    return True


def ensure_live_client_ready():
    print("[1.5/4] Verifying live client connectivity...")
    try:
        client = ExecutionClient()
        collateral = client.get_balance_allowance(asset_type="COLLATERAL")
        if not isinstance(collateral, dict):
            print("[!] Live client failed: collateral balance payload missing or invalid.\n")
            return False
        balance = collateral.get("balance", collateral.get("amount", collateral.get("available_balance")))
        source = getattr(client, 'credential_source', 'unknown')
        if source not in {"stored_env", "derived_refreshed_env"}:
            print(f"[!] Live client connected through unsupported credential source: {source}\n")
            return False
        print(f"[+] Live client connected. Collateral balance payload received: {balance}")
        print(f"[+] Credential source in use: {source}\n")
        return True
    except Exception as exc:
        print(f"[!] Live client verification failed: {exc}\n")
        return False


def ensure_optional_rl_model():
    print("[2/3] Checking optional RL model weights...")

    if WEIGHTS_PATH.exists():
        print(f"[+] Found RL weights: {WEIGHTS_PATH} (optional fallback)\n")
        return True

    print("[!] No RL weights found.")
    print("[+] Continuing anyway: supervised / event-driven pipeline is the default path.\n")
    return True


def maybe_retrain_before_start():
    print("[2.5/4] Checking whether the model should retrain from accumulated data...")
    retrainer = Retrainer()
    try:
        retrained = retrainer.maybe_retrain()
    except Exception as exc:
        print(f"[!] Pre-start retraining failed but startup will continue: {exc}\n")
        return False
    if retrained:
        print("[+] Retraining triggered before startup. Latest weights refreshed.\n")
    else:
        print("[+] No pre-start retraining needed yet.\n")
    return retrained


def should_refresh_research_artifacts(max_age_minutes=60):
    force_refresh = os.getenv("FORCE_RESEARCH_REFRESH", "").strip().lower() in {"1", "true", "yes", "on"}
    if force_refresh:
        return True, "FORCE_RESEARCH_REFRESH enabled"
    missing = [str(path) for path in RESEARCH_ARTIFACTS if not path.exists()]
    if missing:
        return True, f"missing artifacts: {', '.join(missing[:3])}"
    latest_mtime = min(path.stat().st_mtime for path in RESEARCH_ARTIFACTS)
    age_seconds = max(0, int(__import__("time").time() - latest_mtime))
    if age_seconds > max_age_minutes * 60:
        return True, f"artifacts are stale ({age_seconds}s old)"
    return False, f"artifacts are fresh ({age_seconds}s old)"


def build_research_artifacts():
    refresh, reason = should_refresh_research_artifacts()
    if not refresh:
        print(f"[3/4] Skipping research rebuild: {reason}.\n")
        return
    print(f"[3/4] Building research datasets / supervised artifacts... ({reason})")
    try:
        run_research_pipeline()
        print("[+] Research pipeline refreshed (historical dataset, targets, eval files).\n")
    except Exception as exc:
        print(f"[!] Research pipeline failed but supervisor can still run: {exc}\n")


def start_supervisor():
    apply_supervisor_ui_patch(supervisor_module)
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

    if not ensure_live_client_ready():
        sys.exit(1)

    if not ensure_optional_rl_model():
        sys.exit(1)

    # Final quick sanity check before loop
    maybe_retrain_before_start()
    build_research_artifacts()

    if load_brain() is None:
        print("[!] RL model not available. Starting with supervised-first mode only.\n")

    start_supervisor()


if __name__ == "__main__":
    main()

