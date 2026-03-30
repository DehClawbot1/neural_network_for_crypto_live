import os
import sys

from run_bot import main as run_main


if __name__ == "__main__":
    os.environ["TRADING_MODE"] = "paper"
    run_main()

