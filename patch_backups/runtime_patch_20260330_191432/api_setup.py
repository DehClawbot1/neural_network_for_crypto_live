import os
import logging
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ENV_TEMPLATE = """# Neural Network for Crypto - runtime configuration
# Choose one mode: live or paper
TRADING_MODE=live

# Live mode
PRIVATE_KEY=
POLYMARKET_FUNDER=
POLYMARKET_SIGNATURE_TYPE=1
POLYMARKET_HOST=https://clob.polymarket.com
POLYMARKET_CHAIN_ID=137

# Paper fallback
SIMULATED_STARTING_BALANCE=1000

# Optional
ALLOW_ONCHAIN_BALANCE_FALLBACK=false
"""


def _is_interactive_mode() -> bool:
    return os.environ.get("_INTERACTIVE_MODE") == "1"


def _write_template(env_path: Path) -> None:
    env_path.write_text(ENV_TEMPLATE, encoding="utf-8")


def validate_environment() -> bool:
    """
    Validate the repo environment for either live trading or paper mode.

    This replaces the older paper-only validation so startup, dashboard, and
    execution all agree on the same env contract.
    """
    env_path = Path(".env")
    if not env_path.exists() and not _is_interactive_mode():
        logging.warning(".env file not found. Writing a safe template to %s", env_path)
        _write_template(env_path)
        logging.info("Review the template, choose TRADING_MODE, and rerun.")
        return False

    load_dotenv(override=False)

    trading_mode = (os.getenv("TRADING_MODE") or "paper").strip().lower()
    if trading_mode not in {"live", "paper"}:
        logging.error("Invalid TRADING_MODE=%r. Use 'live' or 'paper'.", trading_mode)
        return False

    if trading_mode == "paper":
        balance = os.getenv("SIMULATED_STARTING_BALANCE", "").strip()
        try:
            ok = float(balance) > 0
        except Exception:
            ok = False
        if not ok:
            logging.error("Paper mode requires SIMULATED_STARTING_BALANCE > 0.")
            return False
        logging.info("Environment validated for PAPER mode. Simulated balance=%s", balance)
        return True

    # live mode
    private_key = (os.getenv("PRIVATE_KEY") or "").strip()
    funder = (os.getenv("POLYMARKET_FUNDER") or "").strip()
    signature_type = (os.getenv("POLYMARKET_SIGNATURE_TYPE") or "").strip()

    missing = []
    if not private_key:
        missing.append("PRIVATE_KEY")
    if not funder and not _is_interactive_mode():
        missing.append("POLYMARKET_FUNDER")
    if signature_type not in {"0", "1", "2"}:
        missing.append("POLYMARKET_SIGNATURE_TYPE (must be 0, 1, or 2)")

    if missing:
        logging.error("Live mode is missing: %s", ", ".join(missing))
        return False

    if signature_type == "0":
        logging.warning(
            "signature_type=0 is a direct EOA path and often fails unless all on-chain approvals already exist. "
            "Prefer 1 or 2 for normal Polymarket account trading."
        )

    logging.info(
        "Environment validated for LIVE mode. signature_type=%s funder=%s",
        signature_type,
        f"{funder[:10]}..." if funder else "interactive-memory",
    )
    return True


if __name__ == "__main__":
    sys_ok = validate_environment()
    print("[+] Environment OK" if sys_ok else "[-] Environment invalid")
