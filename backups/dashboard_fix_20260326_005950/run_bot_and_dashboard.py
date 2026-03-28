"""
run_bot_and_dashboard.py — FIXED

FIX: When launched from start.py (interactive mode), env vars like PRIVATE_KEY,
TRADING_MODE, and _INTERACTIVE_MODE are set in os.environ. subprocess.run()
inherits these automatically, BUT we need to ensure the Streamlit subprocess
doesn't override them with load_dotenv().

The fix is two-fold:
  1. Pass env vars explicitly to child processes (belt)
  2. dashboard.py uses safe_load_dotenv() which checks _INTERACTIVE_MODE (suspenders)
"""

import multiprocessing
import os
import subprocess
import sys
import time
import webbrowser


def _get_env():
    """Return the current environment dict.

    In interactive mode, os.environ already has PRIVATE_KEY, TRADING_MODE, etc.
    subprocess.run inherits os.environ by default, so this just confirms
    the inheritance is explicit.
    """
    return dict(os.environ)


def run_bot_process():
    subprocess.run([sys.executable, "run_bot.py"], env=_get_env(), check=False)


def run_dashboard_process():
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "dashboard.py"],
        env=_get_env(),
        check=False,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()

    mode = os.getenv("TRADING_MODE", "paper")
    interactive = os.getenv("_INTERACTIVE_MODE", "0") == "1"

    print("[+] Starting bot + dashboard in one launcher...")
    print(f"    Mode: {mode}")
    print(f"    Interactive: {interactive}")
    if interactive:
        print("    Credentials: in-memory from start.py (will be inherited by subprocesses)")

    bot_process = multiprocessing.Process(target=run_bot_process, name="Bot")
    dashboard_process = multiprocessing.Process(target=run_dashboard_process, name="Dashboard")

    bot_process.start()
    time.sleep(2)
    dashboard_process.start()
    time.sleep(4)
    webbrowser.open("http://127.0.0.1:8501")

    try:
        bot_process.join()
        dashboard_process.join()
    except KeyboardInterrupt:
        print("\n[!] Shutting down bot + dashboard...")
        for proc in [bot_process, dashboard_process]:
            if proc.is_alive():
                proc.terminate()
        bot_process.join()
        dashboard_process.join()
        print("[+] Shutdown complete.")
