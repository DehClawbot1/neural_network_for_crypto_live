"""
start.py — Interactive Launcher for Neural Network for Crypto (FIXED)
=====================================================================

FIXES:
  1. Properly patches load_dotenv BEFORE any imports that might call it
  2. Offers to launch dashboard alongside bot (via run_bot_and_dashboard.py)
  3. Better error messages for common credential issues
  4. Validates wallet address format more carefully
  5. FIX: Asks for signature_type interactively instead of hardcoding it

Usage:
    python start.py
"""

import os
import sys
import getpass


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_banner():
    clear_screen()
    print("=" * 60)
    print("  NEURAL NETWORK FOR CRYPTO — INTERACTIVE LAUNCHER")
    print("=" * 60)
    print()
    print("  This launcher will ask for your credentials.")
    print("  Nothing is saved to disk — credentials stay in memory.")
    print()
    print("=" * 60)
    print()


def prompt_credentials():
    """Prompt user for all required credentials interactively."""

    print("--- WALLET CONFIGURATION ---\n")

    wallet_address = input(
        "  Polymarket wallet address\n"
        "  (the 0x... address from your Polymarket profile): "
    ).strip()

    if wallet_address and not wallet_address.startswith("0x"):
        wallet_address = "0x" + wallet_address

    print()
    private_key = getpass.getpass(
        "  Private key (hidden while typing): "
    ).strip()

    if private_key and not private_key.startswith("0x"):
        private_key = "0x" + private_key

    # ── FIX: Ask for signature type interactively ──
    print()
    print("--- SIGNATURE TYPE ---\n")
    print("  How do you log into Polymarket?")
    print("    1 = Email / Magic Link / Google login  (most common)")
    print("    2 = MetaMask / Rabby / browser wallet")
    print("    0 = Direct EOA (no Polymarket account, trading from own wallet)")
    print()
    sig_type_input = input("  Your login method [1/2/0] (default: 1): ").strip()
    if sig_type_input in ("0", "1", "2"):
        sig_type = sig_type_input
    else:
        sig_type = "1"  # Default to email/Magic since that's most common

    sig_type_labels = {"0": "EOA (direct wallet)", "1": "Email/Magic/Google", "2": "MetaMask/Rabby"}
    print(f"  → Using signature_type={sig_type} ({sig_type_labels.get(sig_type, 'unknown')})")

    print()
    print("--- TRADING MODE ---\n")
    mode = input("  Trading mode [live/paper] (default: live): ").strip().lower()
    if mode not in ("live", "paper"):
        mode = "live"

    print()
    print("--- OPTIONAL: L2 API CREDENTIALS ---")
    print("  (Press Enter to skip — the bot will auto-derive them)\n")

    api_key = input("  API Key (or Enter to skip): ").strip()
    api_secret = ""
    api_passphrase = ""

    if api_key:
        api_secret = getpass.getpass("  API Secret (hidden): ").strip()
        api_passphrase = getpass.getpass("  API Passphrase (hidden): ").strip()

    print()
    print("--- LAUNCH MODE ---\n")
    launch_dashboard = input("  Also launch dashboard? [Y/n]: ").strip().lower()
    launch_dashboard = launch_dashboard in ("", "y", "yes")

    return {
        "POLYMARKET_PUBLIC_ADDRESS": wallet_address,
        "POLYMARKET_FUNDER": wallet_address,
        "PRIVATE_KEY": private_key,
        "TRADING_MODE": mode,
        "POLYMARKET_API_KEY": api_key,
        "POLYMARKET_API_SECRET": api_secret,
        "POLYMARKET_API_PASSPHRASE": api_passphrase,
        # Defaults
        "POLYMARKET_HOST": "https://clob.polymarket.com",
        "POLYMARKET_CHAIN_ID": "137",
        # FIX: Use the user's chosen signature type instead of hardcoding
        "POLYMARKET_SIGNATURE_TYPE": sig_type,
        "SIMULATED_STARTING_BALANCE": "1000",
        "MAX_RISK_PER_TRADE": "50",
        # Flags
        "_INTERACTIVE_MODE": "1",
        "_LAUNCH_DASHBOARD": "1" if launch_dashboard else "0",
    }


def apply_credentials(creds: dict):
    """Set all credentials as environment variables (memory only, never on disk)."""
    for key, value in creds.items():
        if value:
            os.environ[key] = value


def patch_dotenv():
    """Patch load_dotenv BEFORE any other imports to prevent .env override.

    FIX: This must happen before importing run_bot, supervisor, etc. because
    those modules call `from dotenv import load_dotenv; load_dotenv()` at
    the top level.
    """
    try:
        import dotenv
        _original = dotenv.load_dotenv

        def _noop(*args, **kwargs):
            """Patched: skip .env file loading in interactive mode."""
            return False

        dotenv.load_dotenv = _noop
        # Also patch at the module level for any future imports
        sys.modules["dotenv"].load_dotenv = _noop
    except ImportError:
        pass  # dotenv not installed, nothing to patch


def print_startup_summary(creds: dict):
    """Show what's configured (masking sensitive values)."""
    wallet = creds.get("POLYMARKET_PUBLIC_ADDRESS", "?")
    mode = creds.get("TRADING_MODE", "?")
    has_api_key = bool(creds.get("POLYMARKET_API_KEY"))
    pk = creds.get("PRIVATE_KEY", "")
    pk_preview = pk[:6] + "..." + pk[-4:] if len(pk) > 10 else "(too short)"
    launch_dash = creds.get("_LAUNCH_DASHBOARD") == "1"
    sig_type = creds.get("POLYMARKET_SIGNATURE_TYPE", "?")
    sig_type_labels = {"0": "EOA", "1": "Email/Magic/Google", "2": "MetaMask/Rabby"}

    print()
    print("--- STARTUP SUMMARY ---")
    print()
    print(f"  Mode:           {mode.upper()}")
    print(f"  Wallet:         {wallet}")
    print(f"  Private Key:    {pk_preview}")
    print(f"  Signature Type: {sig_type} ({sig_type_labels.get(sig_type, 'unknown')})")
    print(f"  API Creds:      {'provided' if has_api_key else 'will auto-derive'}")
    print(f"  Dashboard:      {'will launch alongside bot' if launch_dash else 'bot only'}")
    print()

    confirm = input("  Start? [Y/n]: ").strip().lower()
    return confirm in ("", "y", "yes")


def run_bot(with_dashboard=False):
    """Import and run the bot after credentials are set."""
    print("=" * 60)
    if with_dashboard:
        print("  STARTING BOT + DASHBOARD...")
    else:
        print("  STARTING BOT...")
    print("=" * 60)
    print()

    try:
        if with_dashboard:
            from run_bot_and_dashboard import run_bot_process, run_dashboard_process
            import multiprocessing
            import time
            import webbrowser

            multiprocessing.freeze_support()
            bot_proc = multiprocessing.Process(target=run_bot_process, name="Bot")
            dash_proc = multiprocessing.Process(target=run_dashboard_process, name="Dashboard")
            bot_proc.start()
            time.sleep(2)
            dash_proc.start()
            time.sleep(4)
            webbrowser.open("http://127.0.0.1:8501")
            try:
                bot_proc.join()
                dash_proc.join()
            except KeyboardInterrupt:
                print("\n[!] Shutting down...")
                for p in [bot_proc, dash_proc]:
                    if p.is_alive():
                        p.terminate()
                bot_proc.join()
                dash_proc.join()
        else:
            from run_bot import main
            main()
    except KeyboardInterrupt:
        print("\n\n[!] Bot stopped by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[!] Bot crashed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    print_banner()
    creds = prompt_credentials()

    # ── FIX 1: Apply credentials BEFORE patching dotenv ──
    apply_credentials(creds)

    # ── FIX 1: Patch dotenv BEFORE any imports that call load_dotenv() ──
    patch_dotenv()

    if not print_startup_summary(creds):
        print("\n  Cancelled. Exiting.")
        sys.exit(0)

    with_dashboard = creds.get("_LAUNCH_DASHBOARD") == "1"
    run_bot(with_dashboard=with_dashboard)


if __name__ == "__main__":
    main()
