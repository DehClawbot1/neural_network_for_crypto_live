import multiprocessing
import subprocess
import sys
import time
import webbrowser


def run_bot_process():
    subprocess.run([sys.executable, "run_bot.py"], check=False)


def run_dashboard_process():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], check=False)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("[+] Starting bot + dashboard in one launcher...")
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

