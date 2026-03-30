import os

if __name__ == "__main__":
    os.environ["TRADING_MODE"] = os.getenv("TRADING_MODE", "live")
    from run_bot import main as run_main # BUG FIX 3: Import AFTER env var is set
    run_main()
