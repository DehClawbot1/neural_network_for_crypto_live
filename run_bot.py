import os
import sys
import logging
from pathlib import Path

# ── Only load .env if NOT in interactive mode ──
if not os.environ.get("_INTERACTIVE_MODE"):
    from dotenv import load_dotenv
    load_dotenv()
else:
    # In interactive mode, credentials are already in os.environ
    pass

from api_setup import validate_environment

# ── BUG FIX I: Set CPU threading env vars BEFORE any numpy/sklearn import ──
try:
    from hardware_config import get_parallel_env_vars, get_sklearn_jobs, get_torch_device, PHYSICAL_CORES, LOGICAL_CORES
    get_parallel_env_vars()
    logging.info("[HW] CPU: %d cores / %d threads | sklearn n_jobs=%d | torch device=%s",
                 PHYSICAL_CORES, LOGICAL_CORES, get_sklearn_jobs(), get_torch_device())
except ImportError:
    pass

from rl_trainer import train_model
from supervisor import main_loop, load_brain
import supervisor as supervisor_module
from supervisor_ui_patch import apply_supervisor_ui_patch
from retrainer import Retrainer
from real_pipeline import run_research_pipeline
from execution_client import ExecutionClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

LEGACY_WEIGHTS_PATH = Path("weights/ppo_polytrader.zip")
ENTRY_WEIGHTS_PATH = Path("weights/ppo_entry_policy.zip")
POSITION_WEIGHTS_PATH = Path("weights/ppo_position_policy.zip")
LOGS_DIR = Path("logs")
RESEARCH_ARTIFACTS = [
    LOGS_DIR / "historical_dataset.csv",
    LOGS_DIR / "contract_targets.csv",
    LOGS_DIR / "wallet_alpha_history.csv",
    LOGS_DIR / "supervised_eval.csv",
    LOGS_DIR / "time_split_eval.csv",
    LOGS_DIR / "path_replay_backtest.csv",
]


def is_interactive():
    return os.environ.get("_INTERACTIVE_MODE") == "1"


def print_banner():
    print("\n=== NEURAL NETWORK FOR CRYPTO ===")
    print("Mode: LIVE-TEST / REAL-TIME DATA")
    if is_interactive():
        print("Credentials: FROM USER INPUT (interactive mode)")
    else:
        print("Credentials: FROM .env FILE")
    print("This launcher validates the environment, checks model weights, and starts the supervisor.\n")


def ensure_environment():
    print("[1/5] Checking environment...")
    trading_mode = os.getenv("TRADING_MODE", "").strip().lower()
    if trading_mode != "live":
        print(f"[!] Invalid TRADING_MODE='{trading_mode or 'missing'}'.")
        print("[!] This launcher now requires TRADING_MODE=live.\n")
        return False

    if is_interactive():
        # In interactive mode, skip .env file validation — creds are in memory
        private_key = os.getenv("PRIVATE_KEY")
        funder = os.getenv("POLYMARKET_FUNDER")
        if private_key and funder:
            print("[+] Environment OK (interactive mode — credentials in memory)\n")
            return True
        else:
            print("[!] Missing PRIVATE_KEY or POLYMARKET_FUNDER.\n")
            return False

    valid = validate_environment()
    if not valid:
        print("[!] Environment template created or configuration needs review.")
        print("    Re-run `python run_bot.py` after confirming your .env values.\n")
        return False

    print("[+] Environment OK\n")
    return True


def ensure_live_client_ready():
    """
    BUG FIX B: Print clear USDC balance at startup.
    """
    print("[2/5] Verifying live client connectivity and balance...")
    try:
        client = ExecutionClient()
        collateral = client.get_balance_allowance(asset_type="COLLATERAL")
        if not isinstance(collateral, dict):
            print("[!] Live client failed: collateral balance payload missing or invalid.\n")
            return False

        # ── BUG FIX B: Extract and display actual balance clearly ──
        clob_balance = 0.0
        for key in ["balance", "available", "available_balance", "amount"]:
            if collateral.get(key) is not None:
                # FIX C1: Normalize microdollars → dollars
                clob_balance = client._normalize_usdc_balance(collateral[key])
                break

        onchain_balance = 0.0
        try:
            onchain = client.get_onchain_collateral_balance()
            onchain_balance = float((onchain or {}).get("total", 0.0) or 0.0)
        except Exception:
            pass

        available = clob_balance
        source = getattr(client, 'credential_source', 'unknown')
        signature_type = str(getattr(client, 'signature_type', os.getenv("POLYMARKET_SIGNATURE_TYPE", "")))

        if source not in {"stored_env", "derived_refreshed_env"}:
            print(f"[!] Live client connected through unsupported credential source: {source}\n")
            return False

        print(f"[+] Live client connected successfully!")
        print(f"    Credential source: {source}")
        print(f"    Signature type:    {signature_type}")
        print(f"    ┌─────────────────────────────────────────┐")
        print(f"    │  CLOB/API Balance:    ${clob_balance:>12,.2f}   │")
        print(f"    │  On-chain USDC:       ${onchain_balance:>12,.2f}   │")
        print(f"    │  ─────────────────────────────────────  │")
        print(f"    │  SPENDABLE NOW:       ${available:>12,.2f}   │")
        print(f"    └─────────────────────────────────────────┘")
        print()

        if available <= 0:
            print("[!] WARNING: CLOB/API spendable balance is zero. The bot will not be able to place bets.\n")
        if signature_type == "2" and onchain_balance > 0 and clob_balance <= 0:
            print("[!] MODE-2 WARNING: wallet USDC exists on-chain, but CLOB/API balance is zero.")
            print("    This usually means one of these is wrong:")
            print("    - POLYMARKET_FUNDER is not the Polymarket profile wallet")
            print("    - stored L2 API creds belong to a different signature mode / wallet")
            print("    - collateral is visible on-chain but not usable by the CLOB account yet")
            print()

        return True
    except Exception as exc:
        print(f"[!] Live client verification failed: {exc}\n")
        return False


def ensure_optional_rl_model():
    """
    BUG FIX A: If no RL weights exist at all, train initial bootstrap weights
    so the model doesn't start from scratch every time.
    """
    print("[3/5] Checking RL model weights...")

    legacy_exists = LEGACY_WEIGHTS_PATH.exists()
    entry_exists = ENTRY_WEIGHTS_PATH.exists()
    position_exists = POSITION_WEIGHTS_PATH.exists()

    if entry_exists or position_exists:
        if entry_exists:
            print(f"[+] Found entry RL weights: {ENTRY_WEIGHTS_PATH}")
        else:
            print("[!] Entry RL weights missing: weights/ppo_entry_policy.zip")
        if position_exists:
            print(f"[+] Found position RL weights: {POSITION_WEIGHTS_PATH}")
        else:
            print("[!] Position RL weights missing: weights/ppo_position_policy.zip")
        print("")
        return True

    if legacy_exists:
        print(f"[+] Found existing RL weights: {LEGACY_WEIGHTS_PATH}")
        print(f"    (Will resume training from these weights on retrain)\n")
        return True

    # ── BUG FIX A: Bootstrap initial RL weights so they persist ──
    print("[!] No RL weights found. Training initial bootstrap weights (1000 steps)...")
    print("    This is a one-time operation. Future runs will resume from saved weights.")
    try:
        os.makedirs("weights", exist_ok=True)
        train_model(timesteps=1000)
        if LEGACY_WEIGHTS_PATH.exists():
            print(f"[+] Initial RL weights saved to {LEGACY_WEIGHTS_PATH}")
            print("    Future retrains will resume from these weights (not start from scratch).\n")
            return True
        else:
            print("[!] RL training completed but weights file not found. Continuing with supervised mode.\n")
            return True
    except Exception as exc:
        print(f"[!] Initial RL training failed: {exc}")
        print("[+] Continuing with supervised-first mode (RL is optional).\n")
        return True


def maybe_retrain_before_start():
    startup_retrain_enabled = os.getenv("ENABLE_STARTUP_RETRAIN", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not startup_retrain_enabled:
        print("[3.5/5] Skipping pre-start retraining (ENABLE_STARTUP_RETRAIN is off).\n")
        return False

    print("[3.5/5] Checking whether the model should retrain from accumulated data...")
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


def should_refresh_research_artifacts(max_age_minutes=240):
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
        print(f"[4/5] Skipping research rebuild: {reason}.\n")
        return
    print(f"[4/5] Building research datasets / supervised artifacts... ({reason})")
    try:
        run_research_pipeline()
        print("[+] Research pipeline refreshed (historical dataset, targets, eval files).\n")
    except Exception as exc:
        print(f"[!] Research pipeline failed but supervisor can still run: {exc}\n")


def start_supervisor():
    apply_supervisor_ui_patch(supervisor_module)
    print("[5/5] Starting supervisor...")
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




def ensure_signature_type():
    """Always prompt in interactive sessions so the launcher never silently assumes the signature type."""
    labels = {"0": "EOA (direct wallet)", "1": "Email/Magic/Google", "2": "MetaMask/Rabby"}
    current = os.getenv("POLYMARKET_SIGNATURE_TYPE", "").strip()

    if not sys.stdin.isatty():
        if current not in labels:
            current = "1"
            os.environ["POLYMARKET_SIGNATURE_TYPE"] = current
            print("[+] Non-interactive mode: defaulting to signature_type=1 (Email/Magic/Google)")
        else:
            print(f"[+] Signature type: {current} ({labels[current]})")
        return

    print()
    print("--- POLYMARKET LOGIN METHOD ---")
    print("Choose the signature type before execution:")
    print("  1 = Email / Magic Link / Google login  -> trade through your Polymarket proxy wallet")
    print("      Best for normal bot users. This is the safest default for live trading in this repo.")
    print("  2 = MetaMask / Rabby / browser extension wallet -> trade through the browser-wallet-linked proxy wallet")
    print("      Use this when your Polymarket account is tied to an extension wallet and your CLOB balance lives there.")
    print("  0 = Direct EOA wallet -> sign directly from the wallet itself")
    print("      Only choose this if you intentionally trade as a direct wallet and already set all required on-chain approvals.")
    print("      In this repo, type 0 is treated as advanced/unsupported and often leads to order failures or $0 usable balance.")
    print()
    if current in labels:
        print(f"Current setting: {current} ({labels[current]})")
    print("If you are unsure, start with 1. A wrong choice commonly shows $0 balance or ghost positions.")
    print()
    choice = input("Your choice [1/2/0] (Enter = keep current, default: 1): ").strip()
    if choice not in ("0", "1", "2"):
        choice = current if current in labels else "1"
    os.environ["POLYMARKET_SIGNATURE_TYPE"] = choice
    print(f"[+] Using signature_type={choice} ({labels[choice]})")
    print()

def main():
    print_banner()

    if not ensure_environment():
        sys.exit(1)

    ensure_signature_type()

    if not ensure_live_client_ready():
        sys.exit(1)

    if not ensure_optional_rl_model():
        sys.exit(1)

    maybe_retrain_before_start()
    build_research_artifacts()

    if load_brain() is None:
        print("[!] RL model not available. Starting with supervised-first mode only.\n")

    start_supervisor()


if __name__ == "__main__":
    main()
