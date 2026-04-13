"""
label_builder.py

Centralises all learning-target computation for a single signal row.

Replaces the scattered row.update({...}) block inside ContractTargetBuilder.

Usage:
    lb = LabelBuilder(
        signal_row=signal_row.to_dict(),
        entry_price=entry_price,
        forward_window=forward_window,   # DataFrame, 15m slice
        full_future_window=full_fw,      # DataFrame, 60m slice
        tp_move=0.04,
        sl_move=0.03,
        market_family="btc",
        resolution_store=weather_resolution_store,  # optional
    )
    targets = lb.build()

Returns a flat dict with:
  - Group A (Alpha Truth) columns
  - Group B (Path Truth) columns
  - Group C (Execution Truth) placeholder columns (None — filled by ExecutionEngine)
  - Missing-data flag columns (explicit True/None, never silent 0.5 defaults)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
        return float(default) if not np.isfinite(num) else num
    except Exception:
        return float(default)


class LabelBuilder:
    """
    Compute all learning targets for a single row.

    Design rules:
      - Alpha targets: probability space [0,1] only
      - Path targets: return/direction space only
      - Execution targets: always None here (filled by ExecutionEngine)
      - Missing data: explicit flag column, never silent 0.5 substitution
    """

    def __init__(
        self,
        *,
        signal_row: dict[str, Any],
        entry_price: float,
        forward_window: pd.DataFrame,     # CLOB ticks for [0, 15m]
        full_future_window: pd.DataFrame,  # CLOB ticks for [0, 60m]
        tp_move: float = 0.04,
        sl_move: float = 0.03,
        market_family: str = "",
        resolution_store=None,             # WeatherResolutionStore instance or None
    ) -> None:
        self.row = signal_row
        self.entry_price = float(entry_price)
        self.fwd_window = forward_window
        self.full_window = full_future_window
        self.tp_move = tp_move
        self.sl_move = sl_move
        self.family = str(market_family or signal_row.get("market_family", "")).lower()
        self.resolution_store = resolution_store

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(self) -> dict[str, Any]:
        targets: dict[str, Any] = {}
        targets.update(self._build_path_truth())
        targets.update(self._build_alpha_truth(targets))
        targets.update(self._build_execution_placeholders())
        targets.update(self._build_backward_compat_aliases(targets))
        return targets

    # ------------------------------------------------------------------
    # GROUP B — Path Truth
    # ------------------------------------------------------------------

    def _build_path_truth(self) -> dict[str, Any]:
        ep = self.entry_price
        out: dict[str, Any] = {
            # Missing-data flags
            "entry_price_from_fallback": None,
            "path_horizon_short": None,
        }

        # ── 15-minute forward return ──────────────────────────────────
        fwd = self.fwd_window
        if fwd.empty:
            # Fall back to first tick of 60m window — but flag it
            fwd = self.full_window.head(1)
            out["path_horizon_short"] = True

        if fwd.empty or ep <= 0:
            out["path_return_15m"] = None
        else:
            price_15m = float(fwd["price"].iloc[-1])
            out["path_return_15m"] = (price_15m - ep) / ep

        # ── 60-minute path ────────────────────────────────────────────
        full = self.full_window
        if full.empty or ep <= 0:
            out["path_return_60m_end"] = None
            out["path_tp_hit_60m"] = None
            out["path_mfe_60m"] = None
            out["path_mae_60m"] = None
            out["_target_up_internal"] = None
        else:
            moves = [(float(p) - ep) / ep for p in full["price"].astype(float)]
            last_price = float(full["price"].iloc[-1])

            out["path_return_60m_end"] = (last_price - ep) / ep
            out["_target_up_internal"] = int(last_price > ep)

            mfe = max(moves) if moves else None
            mae = min(moves) if moves else None
            out["path_mfe_60m"] = mfe
            out["path_mae_60m"] = mae

            # TP/SL: BTC only
            if self.family.startswith("btc"):
                tp_idx = next((i for i, m in enumerate(moves) if m >= self.tp_move), None)
                sl_idx = next((i for i, m in enumerate(moves) if m <= -self.sl_move), None)
                out["path_tp_hit_60m"] = int(
                    tp_idx is not None and (sl_idx is None or tp_idx < sl_idx)
                )
            else:
                out["path_tp_hit_60m"] = None  # weather: TP/SL thresholds are invalid

        return out

    # ------------------------------------------------------------------
    # GROUP A — Alpha Truth
    # ------------------------------------------------------------------

    def _build_alpha_truth(self, path: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}

        path_return_15m = path.get("path_return_15m")
        target_up_internal = path.get("_target_up_internal")

        # --- Market implied probability (shared) ----------------------
        raw_mip = self.row.get("market_implied_prob")
        try:
            mip = float(raw_mip) if raw_mip is not None else None
            if mip is not None and (not np.isfinite(mip) or mip < 0 or mip > 1):
                mip = None
        except Exception:
            mip = None
        mip_missing = mip is None

        # ── BTC alpha targets ─────────────────────────────────────────
        if self.family.startswith("btc"):
            # A1 — binary direction at 15m (probability proxy)
            if path_return_15m is not None:
                btc_prob_up_15m = int(path_return_15m > 0)
            else:
                btc_prob_up_15m = None
            out["btc_prob_up_15m"] = btc_prob_up_15m

            # A2 — binary direction at 60m (honest name for target_up)
            out["btc_prob_up_60m"] = target_up_internal

            # A3 — market edge in pure probability space
            if btc_prob_up_15m is not None and not mip_missing:
                out["btc_market_edge"] = round(btc_prob_up_15m - mip, 6)
                out["btc_implied_prob_missing"] = None
            else:
                out["btc_market_edge"] = None
                out["btc_implied_prob_missing"] = True

            # Weather stubs
            out["weather_contract_resolved_yes"] = None
            out["weather_market_edge"] = None
            out["weather_resolution_unavailable"] = None
            out["weather_market_edge_unavailable"] = None

        # ── Weather alpha targets ─────────────────────────────────────
        elif self.family.startswith("weather"):
            # A4 — real contract resolution (not CLOB proxy)
            cid = str(self.row.get("condition_id") or "").strip()
            resolved_yes: Optional[int] = None
            if cid and self.resolution_store is not None:
                try:
                    resolved_yes = self.resolution_store.get(cid)
                except Exception as exc:
                    logger.debug("WeatherResolutionStore.get failed: %s", exc)

            out["weather_contract_resolved_yes"] = resolved_yes
            out["weather_resolution_unavailable"] = True if resolved_yes is None else None

            # A5 — weather market edge in pure probability space
            if resolved_yes is not None and not mip_missing:
                out["weather_market_edge"] = round(resolved_yes - mip, 6)
                out["weather_market_edge_unavailable"] = None
            else:
                out["weather_market_edge"] = None
                out["weather_market_edge_unavailable"] = True

            # Make implied_prob_missing flag weather-specific
            out["weather_implied_prob_missing"] = True if mip_missing else None

            # BTC stubs
            out["btc_prob_up_15m"] = None
            out["btc_prob_up_60m"] = None
            out["btc_market_edge"] = None
            out["btc_implied_prob_missing"] = None

        else:
            # Unknown family — null everything
            for col in [
                "btc_prob_up_15m", "btc_prob_up_60m", "btc_market_edge", "btc_implied_prob_missing",
                "weather_contract_resolved_yes", "weather_market_edge",
                "weather_resolution_unavailable", "weather_market_edge_unavailable",
                "weather_implied_prob_missing",
            ]:
                out[col] = None

        return out

    # ------------------------------------------------------------------
    # GROUP C — Execution placeholders (filled by ExecutionEngine)
    # ------------------------------------------------------------------

    def _build_execution_placeholders(self) -> dict[str, Any]:
        return {
            "exec_fill_prob_5s": None,
            "exec_fill_prob_30s": None,
            "exec_expected_pnl_after_cost": None,
            "exec_slippage": None,
            "exec_cancel_risk": None,
            "exec_liquidity_failure_risk": None,
            "exec_cost_unavailable": True,  # overwritten by ExecutionEngine when real data joins
        }

    # ------------------------------------------------------------------
    # Backward-compat aliases
    # (kept so existing trainers that hard-code column names don't break)
    # ------------------------------------------------------------------

    def _build_backward_compat_aliases(self, targets: dict[str, Any]) -> dict[str, Any]:
        """
        Write legacy column names as direct aliases to the new columns.
        These will be phased out once all trainers read from label_schema.
        """
        return {
            # Legacy name → new group B name
            "forward_return_15m": targets.get("path_return_15m"),
            "tp_before_sl_60m": targets.get("path_tp_hit_60m"),   # None for weather
            "target_up": targets.get("_target_up_internal"),
            "mfe_60m": targets.get("path_mfe_60m"),
            "mae_60m": targets.get("path_mae_60m"),
        }
