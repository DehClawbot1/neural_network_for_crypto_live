import multiprocessing
import os
import subprocess
import sys

from polymarket_profile_client import prompt_polymarket_runtime


def run_browser_api():
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "polymarket_browser_api:app", "--reload", "--port", "8001"],
        check=False,
    )


def run_browser_dashboard():
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "polymarket_browser_dashboard.py", "--server.port", "8502"],
        check=False,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    cfg = prompt_polymarket_runtime()
    if cfg.get("wallet"):
        os.environ["POLYMARKET_PUBLIC_ADDRESS"] = cfg["wallet"]
    print("[+] Starting Polymarket Browser API and Browser Dashboard...")
    print("    API:       http://127.0.0.1:8001/docs")
    print("    Dashboard: http://127.0.0.1:8502")

    api_process = multiprocessing.Process(target=run_browser_api, name="BrowserAPI")
    dashboard_process = multiprocessing.Process(target=run_browser_dashboard, name="BrowserDashboard")

    api_process.start()
    dashboard_process.start()

    try:
        api_process.join()
        dashboard_process.join()
    except KeyboardInterrupt:
        print("\n[!] Shutting down browser services...")
        for proc in [api_process, dashboard_process]:
            if proc.is_alive():
                proc.terminate()
        api_process.join()
        dashboard_process.join()
        print("[+] Browser services stopped.")

