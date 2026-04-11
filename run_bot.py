import os
import sys
import logging
from pathlib import Path
import pandas as pd

# â"€â"€ Only load .env if NOT in interactive mode â"€â"€
if not os.environ.get("_INTERACTIVE_MODE"):
    from dotenv import load_dotenv
    load_dotenv()
else:
    # In interactive mode, credentials are already in os.environ
    pass

from api_setup import validate_environment

# â"€â"€ BUG FIX I: Set CPU threading env vars BEFORE any numpy/sklearn import â"€â"€
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
from brain_paths import list_brain_contexts
from brain_coverage_report import build_btc_brain_coverage_report, format_btc_brain_coverage_line
from leaderboard_service import PolymarketLeaderboardService
from model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

LEGACY_WEIGHTS_PATH = Path("weights/ppo_polytrader.zip")
ENTRY_WEIGHTS_PATH = Path("weights/ppo_entry_policy.zip")
POSITION_WEIGHTS_PATH = Path("weights/ppo_position_policy.zip")
LOGS_DIR = Path("logs")
DEFAULT_RESEARCH_REFRESH_MAX_AGE_MINUTES = 30


def _research_artifacts():
    contexts = list_brain_contexts(shared_logs_dir=LOGS_DIR, shared_weights_dir=Path("weights"))
    artifacts = [LOGS_DIR / "wallet_alpha_history.csv"]
    for context in contexts:
        artifacts.extend(
            [
                context.logs_dir / "historical_dataset.csv",
                context.logs_dir / "contract_targets.csv",
                context.logs_dir / "sequence_dataset.csv",
                context.logs_dir / "baseline_eval.csv",
                context.logs_dir / "model_registry_comparison.csv",
                context.logs_dir / "regime_model_comparison.csv",
                context.logs_dir / "decision_profit_audit.md",
            ]
        )
    btc_logs_dir = contexts[0].logs_dir if contexts else LOGS_DIR / "btc"
    artifacts.extend(
        [
            btc_logs_dir / "supervised_eval.csv",
            btc_logs_dir / "time_split_eval.csv",
            btc_logs_dir / "stage2_temporal_eval.csv",
            btc_logs_dir / "walk_forward_eval.csv",
            btc_logs_dir / "path_replay_backtest.csv",
        ]
    )
    return artifacts


def is_interactive():
    return os.environ.get("_INTERACTIVE_MODE") == "1"


def _supports_unicode_stdout() -> bool:
    """
    Return True when stdout encoding can reliably print Unicode box-drawing chars.
    """
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in encoding

def print_banner():
    print("\n=== NEURAL NETWORK FOR CRYPTO ===")
    print("Mode: LIVE-TEST / REAL-TIME DATA")
    if is_interactive():
        print("Credentials: FROM USER INPUT (interactive mode)")
    else:
        print("Credentials: FROM .env FILE")
    print("This launcher validates the environment, checks model weights, and starts the supervisor.\n")


def _preflight_env_check():
    """Validate all required and recommended env vars at startup with clear diagnostics."""
    errors = []
    warnings = []

    trading_mode = os.getenv("TRADING_MODE", "").strip().lower()
    if not trading_mode:
        errors.append("TRADING_MODE is not set (must be 'live')")
    elif trading_mode != "live":
        errors.append(f"TRADING_MODE='{trading_mode}' — this launcher requires 'live'")

    if trading_mode == "live":
        for var in ("PRIVATE_KEY", "POLYMARKET_FUNDER"):
            val = os.getenv(var, "").strip()
            if not val and not (var == "POLYMARKET_FUNDER" and is_interactive()):
                errors.append(f"{var} is not set")

        sig_type = os.getenv("POLYMARKET_SIGNATURE_TYPE", "").strip()
        if sig_type and sig_type not in ("0", "1", "2"):
            errors.append(f"POLYMARKET_SIGNATURE_TYPE='{sig_type}' — must be 0, 1, or 2")

    # Recommended env vars
    for var, desc in [
        ("POLYMARKET_HOST", "Polymarket CLOB host URL"),
        ("POLYMARKET_CHAIN_ID", "Polygon chain ID"),
    ]:
        if not os.getenv(var, "").strip():
            warnings.append(f"{var} not set ({desc}) — using default")

    return errors, warnings


def ensure_environment():
    print("[1/5] Checking environment...")

    errors, warnings = _preflight_env_check()
    for w in warnings:
        print(f"[~] {w}")

    if errors:
        for e in errors:
            print(f"[!] {e}")
        if not is_interactive():
            valid = validate_environment()
            if not valid:
                print("[!] Environment template created or configuration needs review.")
                print("    Re-run `python run_bot.py` after confirming your .env values.\n")
                return False
        else:
            print("[!] Missing required credentials.\n")
            return False

    if is_interactive():
        print("[+] Environment OK (interactive mode — credentials in memory)\n")
        return True

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

        # BUG FIX B: Extract and display actual balance clearly
        clob_balance = 0.0
        for key in ["balance", "available", "available_balance", "amount"]:
            if collateral.get(key) is not None:
                # FIX C1: Normalize microdollars -> dollars
                clob_balance = client._normalize_usdc_balance(collateral[key])
                break

        onchain_balance = 0.0
        try:
            onchain = client.get_onchain_collateral_balance()
            onchain_balance = float((onchain or {}).get("total", 0.0) or 0.0)
        except Exception:
            pass

        available = clob_balance
        source = getattr(client, "credential_source", "unknown")
        signature_type = str(getattr(client, "signature_type", os.getenv("POLYMARKET_SIGNATURE_TYPE", "")))

        if source not in {"stored_env", "derived_refreshed_env"}:
            print(f"[!] Live client connected through unsupported credential source: {source}\n")
            return False

        print("[+] Live client connected successfully!")
        print(f"    Credential source: {source}")
        print(f"    Signature type:    {signature_type}")

        if _supports_unicode_stdout():
            print("    ┌─────────────────────────────────────────┐")
            print(f"    │  CLOB/API Balance:    ${clob_balance:>12,.2f}   │")
            print(f"    │  On-chain USDC:       ${onchain_balance:>12,.2f}   │")
            print("    │  ─────────────────────────────────────  │")
            print(f"    │  SPENDABLE NOW:       ${available:>12,.2f}   │")
            print("    └─────────────────────────────────────────┘")
        else:
            print("    +-----------------------------------------+")
            print(f"    |  CLOB/API Balance:    ${clob_balance:>12,.2f}   |")
            print(f"    |  On-chain USDC:       ${onchain_balance:>12,.2f}   |")
            print("    |  -------------------------------------  |")
            print(f"    |  SPENDABLE NOW:       ${available:>12,.2f}   |")
            print("    +-----------------------------------------+")
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

    # â"€â"€ BUG FIX A: Bootstrap initial RL weights so they persist â"€â"€
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
    print("[3.5/5] Checking whether the model should retrain from accumulated data...")
    try:
        coverage = build_btc_brain_coverage_report(shared_logs_dir=LOGS_DIR, shared_weights_dir=Path("weights"))
        print(f"[~] {format_btc_brain_coverage_line(coverage)}")
    except Exception as exc:
        print(f"[!] BTC brain coverage report failed but startup will continue: {exc}")
    retrainer = Retrainer()
    try:
        retrained = retrainer.maybe_retrain()
    except Exception as exc:
        print(f"[!] Pre-start retraining failed but startup will continue: {exc}\n")
        return False
    if retrained:
        print("[+] Retraining triggered before startup. Latest weights refreshed.\n")
    else:
        print("[+] No pre-start retraining needed yet (cooldown / thresholds / quality gates).\n")
    return retrained


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def should_refresh_research_artifacts(max_age_minutes=None):
    if max_age_minutes is None:
        max_age_minutes = _env_int("RESEARCH_REFRESH_MAX_AGE_MINUTES", DEFAULT_RESEARCH_REFRESH_MAX_AGE_MINUTES)
    max_age_minutes = max(1, int(max_age_minutes))
    force_refresh = os.getenv("FORCE_RESEARCH_REFRESH", "").strip().lower() in {"1", "true", "yes", "on"}
    if force_refresh:
        return True, "FORCE_RESEARCH_REFRESH enabled", max_age_minutes
    research_artifacts = _research_artifacts()
    missing = [str(path) for path in research_artifacts if not path.exists()]
    if missing:
        return True, f"missing artifacts: {', '.join(missing[:3])}", max_age_minutes
    latest_mtime = min(path.stat().st_mtime for path in research_artifacts)
    age_seconds = max(0, int(__import__("time").time() - latest_mtime))
    if age_seconds > max_age_minutes * 60:
        return True, f"artifacts are stale ({age_seconds}s old; threshold={max_age_minutes}m)", max_age_minutes
    return False, f"artifacts are fresh ({age_seconds}s old; threshold={max_age_minutes}m)", max_age_minutes


def build_research_artifacts():
    refresh, reason, _ = should_refresh_research_artifacts()
    if not refresh:
        print(f"[4/5] Skipping research rebuild: {reason}.\n")
        return
    print(f"[4/5] Building research datasets / supervised artifacts... ({reason})")
    try:
        run_research_pipeline()
        print("[+] Research pipeline refreshed (historical dataset, targets, eval files).\n")
        try:
            coverage = build_btc_brain_coverage_report(shared_logs_dir=LOGS_DIR, shared_weights_dir=Path("weights"))
            print(f"[+] {format_btc_brain_coverage_line(coverage)}\n")
        except Exception as coverage_exc:
            print(f"[!] BTC brain coverage refresh failed but startup will continue: {coverage_exc}\n")
    except Exception as exc:
        print(f"[!] Research pipeline failed but supervisor can still run: {exc}\n")


def log_live_leaderboard_status():
    print("[4.5/5] Refreshing leaderboard source status...")
    service = PolymarketLeaderboardService(logs_dir="logs")
    try:
        btc_status = service.snapshot_status(category="CRYPTO", limit=int(os.getenv("LEADERBOARD_TOP_TRADERS_LIMIT", "100") or 100))
        print(
            f"[+] BTC leaderboard: {btc_status['wallet_count']} wallets | "
            f"last_refresh={btc_status['fetched_at'] or 'n/a'}"
        )
    except Exception as exc:
        print(f"[!] BTC leaderboard status unavailable: {exc}")
    try:
        weather_status = service.snapshot_status(category="WEATHER", limit=int(os.getenv("WEATHER_LEADERBOARD_LIMIT", "100") or 100))
        print(
            f"[+] Weather leaderboard: {weather_status['wallet_count']} wallets | "
            f"last_refresh={weather_status['fetched_at'] or 'n/a'}"
        )
    except Exception as exc:
        print(f"[!] Weather leaderboard status unavailable: {exc}")
    print("")


def log_active_model_champions():
    print("[4.6/5] Reading active model champions...")
    contexts = list_brain_contexts(shared_logs_dir=LOGS_DIR, shared_weights_dir=Path("weights"))
    printed_any = False
    for context in contexts:
        registry = ModelRegistry(brain_context=context)
        table = registry.comparison_table()
        if table.empty:
            continue
        champions = table[table.get("is_champion", pd.Series(dtype=bool)) == True].copy()
        if champions.empty:
            champions = table[table.get("promotion_status", pd.Series(dtype=str)).fillna("").astype(str).str.lower() == "promoted"].copy()
        if champions.empty:
            continue
        champions = champions.drop_duplicates(subset=["artifact_group", "market_family", "regime_slice"], keep="last")
        if not printed_any:
            print("[+] Active champions:")
            printed_any = True
        print(f"    [{context.brain_id}]")
        for _, row in champions.iterrows():
            accuracy = pd.to_numeric(pd.Series([row.get("accuracy")]), errors="coerce").iloc[0]
            rmse = pd.to_numeric(pd.Series([row.get("rmse")]), errors="coerce").iloc[0]
            metric_bits = []
            if pd.notna(accuracy):
                metric_bits.append(f"accuracy={float(accuracy):.4f}")
            if pd.notna(rmse):
                metric_bits.append(f"rmse={float(rmse):.4f}")
            print(
                "      - {group} | {kind} | family={family} | regime={regime}{metrics}".format(
                    group=row.get("artifact_group", ""),
                    kind=row.get("model_kind", ""),
                    family=row.get("market_family", ""),
                    regime=row.get("regime_slice", ""),
                    metrics=(f" | {' '.join(metric_bits)}" if metric_bits else ""),
                )
            )
    if not printed_any:
        print("[+] No promoted model champions recorded yet.\n")
        return
    print("")


def start_supervisor():
    apply_supervisor_ui_patch(supervisor_module)
    print("[5/5] Starting supervisor...")
    print("[+] Status: RUNNING")
    print("[+] Press Ctrl+C to stop the bot gracefully.\n")
    print("[+] Expected behavior (LIVE MODE):")
    print("    - fetch BTC-related market / account activity")
    print("    - score opportunities through the signal + inference pipeline")
    print("    - submit REAL LIVE orders via Polymarket CLOB API")
    print("    - wait for fill confirmation before registering a position")
    print("    - manage open positions and exit via RL / rule-based decisions")
    print("    - retrain RL model from closed trade outcomes")
    print("    - write execution and episode logs")
    print("    - recheck on the configured supervisor cadence and repeat")
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
    print("  1 = Email / Magic Link / Google login  -> CLOB proxy-wallet trading (off-chain order signing)")
    print("      Best for normal bot users. This is the safest default for live trading in this repo.")
    print("  2 = MetaMask / Rabby / browser extension wallet -> CLOB proxy-wallet trading linked to extension login")
    print("      Use this when your Polymarket account is tied to an extension wallet and your CLOB balance lives there.")
    print("  0 = Direct EOA wallet -> direct wallet signing path (requires on-chain approvals/allowances)")
    print("      Only choose this if you intentionally trade as a direct wallet and already set all required on-chain approvals.")
    print("      In this repo, type 0 is treated as advanced/unsupported and often leads to order failures or $0 usable balance.")
    print()
    if current in labels:
        print(f"Current setting: {current} ({labels[current]})")
    print("If you are unsure, start with 1. A wrong choice commonly shows $0 balance or ghost positions.")
    print()
    try:
        choice = input("Your choice [1/2/0] (Enter = keep current, default: 1): ").strip()
    except EOFError:
        choice = current if current in labels else "1"
        print(f"[+] No interactive stdin available. Using signature_type={choice} ({labels[choice]}).")
    if choice not in ("0", "1", "2"):
        choice = current if current in labels else "1"
    os.environ["POLYMARKET_SIGNATURE_TYPE"] = choice
    print(f"[+] Using signature_type={choice} ({labels[choice]})")
    print()

def main():
    try:
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
        log_live_leaderboard_status()
        log_active_model_champions()

        if load_brain() is None:
            print("[!] RL model not available. Starting with supervised-first mode only.\n")

        start_supervisor()
    except KeyboardInterrupt:
        print("\n[+] Ctrl+C received. Shutting down gracefully...")
        raise SystemExit(130)


if __name__ == "__main__":
    main()

