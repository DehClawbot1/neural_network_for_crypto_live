import logging
import multiprocessing
import os

from rl_trainer import fine_tune_from_live_buffer
from run_bot import main as run_main

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(message)s')


def run_trading_process():
    logging.info("Starting live trading execution process...")
    os.environ["TRADING_MODE"] = os.getenv("TRADING_MODE", "live")
    run_main()


def run_training_process():
    logging.info("Starting continuous RL training process...")
    fine_tune_from_live_buffer(min_rows=100, batch_rows=1000, timesteps=256, sleep_seconds=60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("[+] Initializing unified trading and learning system...")

    trading_process = multiprocessing.Process(target=run_trading_process, name="Executor")
    training_process = multiprocessing.Process(target=run_training_process, name="Learner")

    trading_process.start()
    training_process.start()

    try:
        trading_process.join()
        training_process.join()
    except KeyboardInterrupt:
        print("\n[!] Shutting down all processes safely...")
        for process in [trading_process, training_process]:
            if process.is_alive():
                process.terminate()
        trading_process.join()
        training_process.join()
        print("[+] Shutdown complete.")

