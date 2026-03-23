import os

from run_bot import main as run_main


if __name__ == "__main__":
    os.environ["TRADING_MODE"] = os.getenv("TRADING_MODE", "live")
    run_main()

