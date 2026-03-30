"""
launch.py — ONE SCRIPT TO RULE THEM ALL
=========================================
Run this. Type your credentials. It validates everything and starts the bot.
No .env file needed. No PowerShell variables. No config files.

Usage:
    python launch.py

That's it.
"""

import os
import sys
import getpass
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def banner():
    clear()
    print()
    print("  ╔════════════════════════════════════════════════╗")
    print("  ║   NEURAL NETWORK FOR CRYPTO — QUICK LAUNCH    ║")
    print("  ║                                                ║")
    print("  ║   Answer the questions below. Nothing is       ║")
    print("  ║   saved to disk. Credentials stay in memory.   ║")
    print("  ╚════════════════════════════════════════════════╝")
    print()


def ask_credentials():
    """Ask for every credential interactively."""

    # ── 1. Wallet address ──
    print("─── STEP 1: WALLET ADDRESS ───")
    print("  Go to polymarket.com > click your profile icon > copy your wallet address")
    print("  It starts with 0x...")
    print()
    wallet = input("  Your Polymarket wallet address: ").strip()
    if not wallet:
        print("\n  [!] Wallet address is required. Exiting.")
        sys.exit(1)
    if not wallet.startswith("0x"):
        wallet = "0x" + wallet
    if len(wallet) != 42:
        print(f"\n  [!] Wallet address should be 42 characters (0x + 40 hex). Got {len(wallet)}.")
        print(f"       You entered: {wallet}")
        cont = input("  Continue anyway? [y/N]: ").strip().lower()
        if cont not in ("y", "yes"):
            sys.exit(1)

    # ── 2. Private key ──
    print()
    print("─── STEP 2: PRIVATE KEY ───")
    print("  This is your signer key. NOT visible while typing.")
    print("  It must be the key associated with your Polymarket email login.")
    print()
    pk = getpass.getpass("  Private key (hidden): ").strip()
    if not pk:
        print("\n  [!] Private key is required. Exiting.")
        sys.exit(1)
    if not pk.startswith("0x"):
        pk = "0x" + pk
    # Validate it's hex
    try:
        hex_part = pk[2:]
        int(hex_part, 16)
        if len(hex_part) != 64:
            print(f"\n  [!] Private key should be 66 chars (0x + 64 hex). Got 0x + {len(hex_part)}.")
            cont = input("  Continue anyway? [y/N]: ").strip().lower()
            if cont not in ("y", "yes"):
                sys.exit(1)
    except ValueError:
        print("\n  [!] Private key contains non-hex characters!")
        print("       Only 0-9 and a-f are allowed after 0x.")
        print("       Check for spaces, quotes, or copy-paste errors.")
        sys.exit(1)

    # ── 3. Signature type ──
    print()
    print("─── STEP 3: LOGIN METHOD ───")
    print("  How do you log into Polymarket?")
    print()
    print("    1 = Email / Magic Link / Google  (most common)")
    print("    2 = MetaMask / Rabby browser wallet")
    print("    0 = Direct wallet (no Polymarket account)")
    print()
    sig = input("  Your choice [1/2/0] (default: 1): ").strip()
    if sig not in ("0", "1", "2"):
        sig = "1"
    sig_label = {"0": "EOA (direct wallet)", "1": "Email/Magic/Google", "2": "MetaMask/Rabby"}[sig]

    # ── 4. API credentials (optional) ──
    print()
    print("─── STEP 4: API CREDENTIALS (optional) ───")
    print("  If you have L2 API credentials, enter them.")
    print("  If not, press Enter to skip — the bot will auto-derive them.")
    print()
    api_key = input("  API Key (or Enter to skip): ").strip()
    api_secret = ""
    api_passphrase = ""
    if api_key:
        api_secret = getpass.getpass("  API Secret (hidden): ").strip()
        api_passphrase = getpass.getpass("  API Passphrase (hidden): ").strip()

    # ── 5. Dashboard? ──
    print()
    print("─── STEP 5: LAUNCH OPTIONS ───")
    launch_dash = input("  Also launch dashboard? [Y/n]: ").strip().lower()
    launch_dash = launch_dash in ("", "y", "yes")

    return {
        "wallet": wallet,
        "private_key": pk,
        "signature_type": sig,
        "sig_label": sig_label,
        "api_key": api_key,
        "api_secret": api_secret,
        "api_passphrase": api_passphrase,
        "launch_dashboard": launch_dash,
    }


def set_env(creds):
    """Put everything into os.environ so the bot can find it."""
    os.environ["POLYMARKET_PUBLIC_ADDRESS"] = creds["wallet"]
    os.environ["POLYMARKET_FUNDER"] = creds["wallet"]
    os.environ["PRIVATE_KEY"] = creds["private_key"]
    os.environ["POLYMARKET_SIGNATURE_TYPE"] = creds["signature_type"]
    os.environ["POLYMARKET_HOST"] = "https://clob.polymarket.com"
    os.environ["POLYMARKET_CHAIN_ID"] = "137"
    os.environ["TRADING_MODE"] = "live"
    os.environ["SIMULATED_STARTING_BALANCE"] = "1000"
    os.environ["MAX_RISK_PER_TRADE"] = "50"
    os.environ["_INTERACTIVE_MODE"] = "1"

    if creds["api_key"]:
        os.environ["POLYMARKET_API_KEY"] = creds["api_key"]
    if creds["api_secret"]:
        os.environ["POLYMARKET_API_SECRET"] = creds["api_secret"]
    if creds["api_passphrase"]:
        os.environ["POLYMARKET_API_PASSPHRASE"] = creds["api_passphrase"]


def patch_dotenv():
    """Prevent .env file from overriding our in-memory credentials."""
    try:
        import dotenv
        dotenv.load_dotenv = lambda *a, **kw: False
        sys.modules["dotenv"].load_dotenv = lambda *a, **kw: False
    except ImportError:
        pass


def show_summary(creds):
    """Show what we're about to do."""
    pk = creds["private_key"]
    pk_show = pk[:6] + "..." + pk[-4:] if len(pk) > 10 else "???"

    print()
    print("  ╔════════════════════════════════════════════════╗")
    print("  ║              READY TO LAUNCH                   ║")
    print("  ╠════════════════════════════════════════════════╣")
    print(f"  ║  Wallet:     {creds['wallet'][:16]}...        ║")
    print(f"  ║  Key:        {pk_show:<35}║")
    print(f"  ║  Login:      {creds['sig_label']:<35}║")
    print(f"  ║  Sig Type:   {creds['signature_type']:<35}║")
    print(f"  ║  API Creds:  {'provided' if creds['api_key'] else 'will auto-derive':<35}║")
    print(f"  ║  Dashboard:  {'yes' if creds['launch_dashboard'] else 'no':<35}║")
    print("  ╚════════════════════════════════════════════════╝")
    print()


def test_connection(creds):
    """Test that we can connect and see the balance."""
    print("─── TESTING CONNECTION ───")
    print()

    from execution_client import ExecutionClient

    try:
        client = ExecutionClient()
        print(f"  [+] Client connected (source: {client.credential_source})")
        print(f"  [+] Signature type: {client.signature_type} ({creds['sig_label']})")
        print(f"  [+] Funder: {client.funder}")
    except Exception as exc:
        print(f"  [!] Client creation FAILED: {exc}")
        print()
        print("  Common causes:")
        print("    - Private key has non-hex characters")
        print("    - Wrong signature type for your login method")
        print("    - Network/firewall blocking polymarket.com")
        return False

    # CLOB balance
    clob_balance = 0.0
    try:
        raw = client.get_balance_allowance(asset_type="COLLATERAL")
        if isinstance(raw, dict):
            raw_val = raw.get("balance", raw.get("amount", 0))
            clob_balance = client._normalize_usdc_balance(raw_val)
    except Exception:
        pass

    # On-chain balance
    onchain_total = 0.0
    try:
        onchain = client.get_onchain_collateral_balance()
        onchain_total = float((onchain or {}).get("total", 0.0) or 0.0)
    except Exception:
        pass

    available = max(clob_balance, onchain_total)

    print()
    print(f"  +-------------------------------------------+")
    print(f"  |  CLOB/API Balance:    ${clob_balance:>12,.2f}   |")
    print(f"  |  On-chain USDC:       ${onchain_total:>12,.2f}   |")
    print(f"  |  -----------------------------------------|")
    print(f"  |  AVAILABLE TO TRADE:  ${available:>12,.2f}   |")
    print(f"  +-------------------------------------------+")
    print()

    if clob_balance > 0:
        print(f"  [+] CLOB sees your funds! Ready to trade.")
    elif onchain_total > 0:
        print(f"  [!] ${onchain_total:.2f} is on-chain but CLOB shows $0.")
        print(f"      The bot will use the on-chain fallback.")
        print(f"      For best results, deposit through polymarket.com first.")
    else:
        print(f"  [!] No funds found. Deposit USDC to your Polymarket account.")

    if available <= 0:
        cont = input("\n  Continue with $0 balance? [y/N]: ").strip().lower()
        if cont not in ("y", "yes"):
            return False

    # Test order book
    print()
    print("─── TESTING ORDER BOOK ───")
    try:
        from orderbook_guard import OrderBookGuard
        guard = OrderBookGuard()
        test_token = None
        try:
            import pandas as pd
            if os.path.exists("logs/markets.csv"):
                markets = pd.read_csv("logs/markets.csv", engine="python", on_bad_lines="skip")
                if not markets.empty and "yes_token_id" in markets.columns:
                    tokens = markets["yes_token_id"].dropna().astype(str).tolist()
                    if tokens:
                        test_token = tokens[0]
        except Exception:
            pass

        if test_token:
            analysis = guard.analyze_book(test_token, depth=5)
            if analysis["book_available"]:
                print(f"  [+] Order book working!")
                print(f"      Bid: {analysis['best_bid']} | Ask: {analysis['best_ask']} | Spread: {analysis['spread']}")
            else:
                print(f"  [~] Could not fetch test order book (token may be expired)")
                print(f"      This is OK — bot will fetch live books when trading.")
        else:
            print(f"  [~] No markets discovered yet. Bot will find them on first cycle.")
    except ImportError:
        print(f"  [~] orderbook_guard.py not found — order book checks won't run.")
    except Exception as exc:
        print(f"  [~] Order book test: {exc}")

    return True


def start_bot(creds):
    """Start the actual bot."""
    print()
    print("  ╔════════════════════════════════════════════════╗")
    print("  ║              STARTING BOT...                   ║")
    print("  ╚════════════════════════════════════════════════╝")
    print()

    if creds["launch_dashboard"]:
        try:
            from run_bot_and_dashboard import run_bot_process, run_dashboard_process
            import multiprocessing
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
        except Exception as exc:
            print(f"[!] Dashboard launch failed: {exc}")
            print("[+] Starting bot only...")
            from run_bot import main
            main()
    else:
        from run_bot import main
        main()


def main():
    banner()

    # 1. Ask for credentials
    creds = ask_credentials()

    # 2. Set env vars (memory only)
    set_env(creds)

    # 3. Patch dotenv so .env file doesn't overwrite
    patch_dotenv()

    # 4. Show summary
    show_summary(creds)

    # 5. Test connection + balance + order book
    ok = test_connection(creds)
    if not ok:
        print("\n  [!] Connection test failed. Fix the issue above and re-run.")
        sys.exit(1)

    # 6. Confirm and launch
    print()
    go = input("  Everything looks good. Start the bot? [Y/n]: ").strip().lower()
    if go not in ("", "y", "yes"):
        print("\n  Cancelled.")
        sys.exit(0)

    start_bot(creds)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  [!] Stopped by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n  [!] Error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
