"""
start.py — Interactive Launcher for Neural Network for Crypto
=============================================================

This is your new main entry point. It prompts for credentials at startup,
keeps them in memory only (never written to disk), then runs the bot.

Usage:
    python start.py

No .env file is needed or read.
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

    print("─── WALLET CONFIGURATION ───\n")

    wallet_address = input(
        "  Polymarket wallet address\n"
        "  (the 0x... address from your Polymarket profile): "
    ).strip()

    if not wallet_address.startswith("0x"):
        wallet_address = "0x" + wallet_address

    print()
    private_key = getpass.getpass(
        "  Private key (hidden while typing): "
    ).strip()

    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    print()
    print("─── TRADING MODE ───\n")
    mode = input("  Trading mode [live/paper] (default: live): ").strip().lower()
    if mode not in ("live", "paper"):
        mode = "live"

    print()
    print("─── OPTIONAL: L2 API CREDENTIALS ───")
    print("  (Press Enter to skip — the bot will auto-derive them)\n")

    api_key = input("  API Key (or Enter to skip): ").strip()
    api_secret = ""
    api_passphrase = ""

    if api_key:
        api_secret = getpass.getpass("  API Secret (hidden): ").strip()
        api_passphrase = getpass.getpass("  API Passphrase (hidden): ").strip()

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
        "POLYMARKET_SIGNATURE_TYPE": "1",
        "SIMULATED_STARTING_BALANCE": "1000",
        "MAX_RISK_PER_TRADE": "50",
        # Flag so other modules know we're in interactive mode
        "_INTERACTIVE_MODE": "1",
    }


def apply_credentials(creds: dict):
    """Set all credentials as environment variables (memory only, never on disk)."""
    for key, value in creds.items():
        if value:  # Only set non-empty values
            os.environ[key] = value

    # Prevent load_dotenv from overriding our values
    # by setting this before any imports that call load_dotenv
    os.environ["_SKIP_DOTENV"] = "1"


def print_startup_summary(creds: dict):
    """Show what's configured (masking sensitive values)."""
    wallet = creds.get("POLYMARKET_PUBLIC_ADDRESS", "?")
    mode = creds.get("TRADING_MODE", "?")
    has_api_key = bool(creds.get("POLYMARKET_API_KEY"))
    pk_preview = creds.get("PRIVATE_KEY", "")[:6] + "..." + creds.get("PRIVATE_KEY", "")[-4:]

    print()
    print("─── STARTUP SUMMARY ───")
    print()
    print(f"  Mode:           {mode.upper()}")
    print(f"  Wallet:         {wallet}")
    print(f"  Private Key:    {pk_preview}")
    print(f"  API Creds:      {'provided' if has_api_key else 'will auto-derive'}")
    print()

    confirm = input("  Start the bot? [Y/n]: ").strip().lower()
    return confirm in ("", "y", "yes")


def run_bot():
    """Import and run the bot after credentials are set."""
    # Now safe to import — env vars are already set
    # Patch load_dotenv to be a no-op so .env files are never read
    import dotenv
    _original_load = dotenv.load_dotenv

    def _noop_load_dotenv(*args, **kwargs):
        """Patched: skip .env file loading in interactive mode."""
        return False

    dotenv.load_dotenv = _noop_load_dotenv

    # Also patch it at the module level in common import locations
    sys.modules.setdefault("dotenv", dotenv)

    print("=" * 60)
    print("  STARTING BOT...")
    print("=" * 60)
    print()

    try:
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
    apply_credentials(creds)

    if not print_startup_summary(creds):
        print("\n  Cancelled. Exiting.")
        sys.exit(0)

    run_bot()


if __name__ == "__main__":
    main()
