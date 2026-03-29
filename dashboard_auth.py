"""
dashboard_auth.py
=================
Shared authentication and client management for all dashboard pages.

Fixes:
  - Conditional load_dotenv() that respects _INTERACTIVE_MODE
  - Cached ExecutionClient to avoid re-creation on every Streamlit rerun
  - Graceful fallback when credentials are unavailable
"""

import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def safe_load_dotenv():
    """Load .env file ONLY if not in interactive mode.

    When the user runs `python start.py`, credentials are injected into
    os.environ in-memory.  If the dashboard subprocess inherits those
    env vars, calling load_dotenv() would overwrite them with stale
    (or empty) values from the .env file on disk.
    """
    if os.environ.get("_INTERACTIVE_MODE") == "1":
        logging.info("Dashboard: interactive mode detected — skipping .env file load.")
        return

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def get_trading_mode() -> str:
    """Return the effective trading mode, with fallback detection."""
    mode = os.getenv("TRADING_MODE", "").strip().lower()
    if mode in ("live", "paper"):
        return mode

    # Fallback: if PRIVATE_KEY is set, assume live-capable
    if os.getenv("PRIVATE_KEY"):
        return "live"

    return "paper"


def is_live_mode() -> bool:
    return get_trading_mode() == "live"


def is_interactive_mode() -> bool:
    return os.environ.get("_INTERACTIVE_MODE") == "1"


def get_execution_client_cached():
    """Create or return a cached ExecutionClient.

    Uses Streamlit's cache_resource if available, otherwise falls back
    to a module-level singleton.  This prevents re-creating the client
    (and re-deriving credentials) on every Streamlit rerun.
    """
    try:
        import streamlit as st

        @st.cache_resource(show_spinner=False)
        def _build_client():
            return _create_execution_client()

        return _build_client()
    except Exception:
        return _get_singleton_client()


_singleton_client = None


def _get_singleton_client():
    global _singleton_client
    if _singleton_client is None:
        _singleton_client = _create_execution_client()
    return _singleton_client


def _create_execution_client():
    """Actually construct the ExecutionClient with proper error handling."""
    try:
        from execution_client import ExecutionClient
        client = ExecutionClient()
        logging.info("Dashboard: ExecutionClient created (source=%s)", getattr(client, "credential_source", "unknown"))
        return client
    except ImportError:
        logging.warning("Dashboard: execution_client module not available.")
        return None
    except ValueError as exc:
        logging.warning("Dashboard: ExecutionClient init failed (missing creds?): %s", exc)
        return None
    except Exception as exc:
        logging.error("Dashboard: ExecutionClient init failed: %s", exc)
        return None


def get_wallet_address() -> str:
    """Return the best available wallet address from env or client."""
    # Try client first
    client = get_execution_client_cached()
    if client is not None:
        funder = getattr(client, "funder", None)
        if funder:
            return funder

    # Fall back to env vars
    for key in ["POLYMARKET_PUBLIC_ADDRESS", "POLYMARKET_FUNDER"]:
        addr = os.getenv(key, "").strip()
        if addr:
            return addr

    return ""


def get_balance_info() -> dict:
    """Get balance information with proper error handling."""
    result = {
        "clob_balance": 0.0,
        "onchain_balance": 0.0,
        "available": 0.0,
        "source": "none",
        "error": None,
    }

    client = get_execution_client_cached()
    if client is None:
        result["error"] = "No execution client available"
        return result

    try:
        # Refresh balance if possible
        if hasattr(client, "update_balance_allowance"):
            try:
                client.update_balance_allowance(asset_type="COLLATERAL")
            except Exception:
                pass

        # Get CLOB/API balance
        collateral = client.get_balance_allowance(asset_type="COLLATERAL")
        if isinstance(collateral, dict):
            for key in ["balance", "available", "available_balance", "amount"]:
                if collateral.get(key) is not None:
                    # H1: Normalize microdollars
                    if hasattr(client, '_normalize_usdc_balance'):
                        result["clob_balance"] = client._normalize_usdc_balance(collateral[key])
                    else:
                        raw = float(collateral[key])
                        result["clob_balance"] = raw / 1e6 if raw >= 100 and raw == int(raw) else raw
                    break

        # Get on-chain balance
        addr = get_wallet_address()
        if addr:
            try:
                onchain = client.get_onchain_collateral_balance(wallet_address=addr)
                result["onchain_balance"] = float((onchain or {}).get("total", 0.0) or 0.0)
            except Exception:
                pass

        result["available"] = max(result["clob_balance"], result["onchain_balance"])
        result["source"] = getattr(client, "credential_source", "unknown")

    except Exception as exc:
        result["error"] = str(exc)

    return result
