from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_float(value, default=0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(number):
        return float(default)
    return float(number)


class PositionTelemetry:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_file = self.logs_dir / "position_snapshots.csv"
        self.portfolio_file = self.logs_dir / "portfolio_equity_curve.csv"

    def _position_key(self, row: pd.Series | dict) -> str:
        getter = row.get if hasattr(row, "get") else lambda key, default=None: default
        position_key = str(getter("position_key", "") or getter("position_id", "") or "").strip()
        if position_key:
            return position_key
        token_id = str(getter("token_id", "") or "").strip()
        condition_id = str(getter("condition_id", "") or "").strip()
        outcome_side = str(getter("outcome_side", "") or getter("side", "") or "").strip()
        market = str(getter("market", "") or getter("market_title", "") or "").strip()
        if token_id or condition_id:
            return f"{token_id}|{condition_id}|{outcome_side}"
        return f"{market}|{outcome_side}"

    def enrich_with_live_marks(self, positions_df: pd.DataFrame | None, orderbook_guard=None, fallback_price_map: dict | None = None):
        if positions_df is None or positions_df.empty:
            return pd.DataFrame() if positions_df is None else positions_df

        fallback_price_map = fallback_price_map or {}
        df = positions_df.copy()
        for col in ["token_id", "condition_id", "outcome_side", "market", "market_title", "position_key"]:
            if col not in df.columns:
                df[col] = ""
        for col in ["current_price", "entry_price", "avg_entry_price", "shares", "realized_pnl", "unrealized_pnl", "best_bid", "best_ask", "spread"]:
            if col not in df.columns:
                df[col] = 0.0

        df["token_id"] = df["token_id"].astype(str)
        df["position_key"] = df.apply(self._position_key, axis=1)

        depth = max(1, int(os.getenv("OPEN_POSITION_MARK_DEPTH", "3") or 3))
        for idx, row in df.iterrows():
            token_id = str(row.get("token_id", "") or "").strip()
            entry_price = _safe_float(row.get("entry_price", row.get("avg_entry_price", 0.0)), 0.0)
            current_price = _safe_float(row.get("current_price", entry_price), entry_price)
            if current_price <= 0 and entry_price > 0:
                current_price = entry_price
            best_bid = _safe_float(row.get("best_bid", 0.0), 0.0)
            best_ask = _safe_float(row.get("best_ask", 0.0), 0.0)
            spread = _safe_float(row.get("spread", 0.0), 0.0)
            mark_source = "reconciled_current_price"

            fallback_price = _safe_float(
                fallback_price_map.get(token_id, fallback_price_map.get(str(row.get("market", "") or row.get("market_title", "") or ""), current_price)),
                current_price,
            )

            if orderbook_guard is not None and token_id:
                try:
                    analysis = orderbook_guard.analyze_book(token_id, depth=depth)
                except Exception:
                    analysis = {}
                if analysis.get("book_available"):
                    best_bid = _safe_float(analysis.get("best_bid", best_bid), best_bid)
                    best_ask = _safe_float(analysis.get("best_ask", best_ask), best_ask)
                    if best_bid > 0:
                        current_price = best_bid
                        mark_source = "orderbook_best_bid"
                    elif fallback_price > 0:
                        current_price = fallback_price
                        mark_source = "scored_fallback_price"
                    spread = _safe_float(analysis.get("spread", spread), spread)
                elif fallback_price > 0:
                    current_price = fallback_price
                    mark_source = "scored_fallback_price"
            elif fallback_price > 0:
                current_price = fallback_price
                mark_source = "scored_fallback_price"

            shares = _safe_float(row.get("shares", 0.0), 0.0)
            market_value = shares * current_price
            unrealized_pnl = shares * (current_price - entry_price)
            mid_price = ((best_bid + best_ask) / 2.0) if best_bid > 0 and best_ask > 0 else current_price
            spread_pct = (spread / mid_price) if mid_price > 0 and spread > 0 else 0.0

            df.at[idx, "entry_price"] = entry_price
            df.at[idx, "current_price"] = current_price
            df.at[idx, "mark_price"] = current_price
            df.at[idx, "best_bid"] = best_bid if best_bid > 0 else np.nan
            df.at[idx, "best_ask"] = best_ask if best_ask > 0 else np.nan
            df.at[idx, "spread"] = spread if spread > 0 else max(best_ask - best_bid, 0.0)
            df.at[idx, "mid_price"] = mid_price if mid_price > 0 else np.nan
            df.at[idx, "spread_pct"] = spread_pct if spread_pct > 0 else 0.0
            df.at[idx, "market_value"] = market_value
            df.at[idx, "unrealized_pnl"] = unrealized_pnl
            df.at[idx, "mark_source"] = mark_source

        return df

    def _snapshot_columns(self):
        return [
            "timestamp",
            "position_key",
            "token_id",
            "condition_id",
            "outcome_side",
            "market",
            "market_title",
            "entry_price",
            "current_price",
            "mark_price",
            "best_bid",
            "best_ask",
            "spread",
            "mid_price",
            "spread_pct",
            "shares",
            "market_value",
            "realized_pnl",
            "unrealized_pnl",
            "confidence",
            "status",
            "mark_source",
            "trajectory_state",
            "drawdown_from_peak",
            "recent_return_3",
            "runup_from_entry",
            "volatility_short",
            "fallback_ratio",
        ]

    def _project_snapshot_frame(self, positions_df: pd.DataFrame | None, timestamp: str | None = None):
        if positions_df is None or positions_df.empty:
            return pd.DataFrame(columns=self._snapshot_columns())
        now = timestamp or datetime.now(timezone.utc).isoformat()
        df = positions_df.copy()
        df["timestamp"] = now
        df["position_key"] = df.apply(self._position_key, axis=1)
        if "market_title" not in df.columns and "market" in df.columns:
            df["market_title"] = df["market"]
        if "market" not in df.columns and "market_title" in df.columns:
            df["market"] = df["market_title"]
        if "mark_price" not in df.columns and "current_price" in df.columns:
            df["mark_price"] = pd.to_numeric(df["current_price"], errors="coerce").fillna(0.0)
        if "current_price" not in df.columns and "mark_price" in df.columns:
            df["current_price"] = pd.to_numeric(df["mark_price"], errors="coerce").fillna(0.0)

        keep = self._snapshot_columns()
        for col in keep:
            if col not in df.columns:
                df[col] = np.nan
        return df[keep].copy()

    def capture_positions(self, positions_df: pd.DataFrame | None):
        if positions_df is None or positions_df.empty:
            return

        now = datetime.now(timezone.utc).isoformat()
        df = self._project_snapshot_frame(positions_df, timestamp=now)
        df.to_csv(self.snapshots_file, mode="a", header=not self.snapshots_file.exists(), index=False)

        portfolio_row = pd.DataFrame(
            [
                {
                    "timestamp": now,
                    "open_positions": int(len(df)),
                    "gross_market_value": float(pd.to_numeric(df["market_value"], errors="coerce").fillna(0.0).sum()),
                    "realized_pnl": float(pd.to_numeric(df.get("realized_pnl", 0.0), errors="coerce").fillna(0.0).sum()),
                    "unrealized_pnl": float(pd.to_numeric(df.get("unrealized_pnl", 0.0), errors="coerce").fillna(0.0).sum()),
                    "entry_notional": float(
                        (
                            pd.to_numeric(df.get("entry_price", 0.0), errors="coerce").fillna(0.0)
                            * pd.to_numeric(df.get("shares", 0.0), errors="coerce").fillna(0.0)
                        ).sum()
                    ),
                }
            ]
        )
        portfolio_row["total_pnl"] = portfolio_row["realized_pnl"] + portfolio_row["unrealized_pnl"]
        portfolio_row.to_csv(self.portfolio_file, mode="a", header=not self.portfolio_file.exists(), index=False)

    def load_snapshots(self, hours: int = 24) -> pd.DataFrame:
        if not self.snapshots_file.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.snapshots_file, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=max(1, int(hours)))
        df = df[df["timestamp"] >= cutoff].copy()
        return df.sort_values(["position_key", "timestamp"])

    def build_trajectory_metrics(self, positions_df: pd.DataFrame | None = None, hours: int = 24) -> dict:
        snapshots = self.load_snapshots(hours=hours)
        if positions_df is not None and not positions_df.empty:
            current = self._project_snapshot_frame(positions_df)
            snapshots = pd.concat([snapshots, current], ignore_index=True, sort=False) if not snapshots.empty else current
            keys = current["position_key"].astype(str).tolist()
            snapshots = snapshots[snapshots["position_key"].astype(str).isin(keys)].copy()
        if snapshots.empty:
            return {}
        snapshots["timestamp"] = pd.to_datetime(snapshots["timestamp"], errors="coerce", utc=True)
        snapshots = snapshots[snapshots["timestamp"].notna()].copy()
        if snapshots.empty:
            return {}

        lookback_points = max(4, int(os.getenv("POSITION_TRAJECTORY_LOOKBACK_POINTS", "12") or 12))
        profit_guard_drawdown = _safe_float(os.getenv("POSITION_PROFIT_GUARD_DRAWDOWN", "0.010"), 0.010)
        reversal_drawdown = _safe_float(os.getenv("POSITION_REVERSAL_DRAWDOWN", "0.012"), 0.012)
        reversal_recent_drop = _safe_float(os.getenv("POSITION_REVERSAL_RECENT_DROP", "-0.006"), -0.006)
        panic_drop_1 = _safe_float(os.getenv("POSITION_PANIC_DROP_1", "-0.018"), -0.018)
        panic_drop_3 = _safe_float(os.getenv("POSITION_PANIC_DROP_3", "-0.025"), -0.025)
        runup_activation = _safe_float(os.getenv("POSITION_RUNUP_ACTIVATION", "0.015"), 0.015)
        spread_stress_pct = _safe_float(os.getenv("POSITION_SPREAD_STRESS_PCT", "0.040"), 0.040)
        fallback_stress_ratio = _safe_float(os.getenv("POSITION_FALLBACK_STRESS_RATIO", "0.50"), 0.50)

        metrics = {}
        for position_key, group in snapshots.groupby("position_key"):
            work = group.sort_values("timestamp").tail(lookback_points).copy()
            prices = pd.to_numeric(work["mark_price"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if prices.empty:
                continue
            entry_price = _safe_float(work["entry_price"].iloc[-1], prices.iloc[0] if len(prices) else 0.0)
            current_price = float(prices.iloc[-1])
            peak_price = float(prices.max())
            runup_from_entry = ((peak_price - entry_price) / entry_price) if entry_price > 0 else 0.0
            drawdown_from_peak = ((peak_price - current_price) / peak_price) if peak_price > 0 else 0.0
            recent_return_1 = ((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2]) if len(prices) >= 2 and prices.iloc[-2] > 0 else 0.0
            recent_return_3 = ((prices.iloc[-1] - prices.iloc[-4]) / prices.iloc[-4]) if len(prices) >= 4 and prices.iloc[-4] > 0 else recent_return_1
            previous_window_return = ((prices.iloc[-2] - prices.iloc[-5]) / prices.iloc[-5]) if len(prices) >= 5 and prices.iloc[-5] > 0 else 0.0

            sample = prices.tail(min(6, len(prices)))
            if len(sample) >= 2 and entry_price > 0:
                slope = np.polyfit(np.arange(len(sample)), sample.to_numpy(dtype=float), 1)[0]
                slope_norm = slope / max(entry_price, 1e-9)
            else:
                slope_norm = 0.0

            if "spread_pct" in work.columns:
                spread_series = pd.to_numeric(work["spread_pct"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            else:
                bid_source = work["best_bid"] if "best_bid" in work.columns else pd.Series(0.0, index=work.index)
                ask_source = work["best_ask"] if "best_ask" in work.columns else pd.Series(0.0, index=work.index)
                bid_series = pd.to_numeric(bid_source, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
                ask_series = pd.to_numeric(ask_source, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
                mid_series = ((bid_series + ask_series) / 2.0).replace(0.0, np.nan)
                spread_series = ((ask_series - bid_series) / mid_series).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            latest_spread_pct = float(spread_series.iloc[-1]) if len(spread_series) else 0.0
            volatility_short = float(prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna().std() or 0.0)
            mark_sources = work.get("mark_source", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
            fallback_ratio = float((mark_sources != "orderbook_best_bid").mean()) if len(mark_sources) else 0.0

            profit_lock_signal = bool(
                current_price > entry_price
                and drawdown_from_peak >= profit_guard_drawdown
                and recent_return_1 < 0
                and slope_norm < 0
            )
            reversal_exit_signal = bool(
                runup_from_entry >= runup_activation
                and drawdown_from_peak >= reversal_drawdown
                and recent_return_3 <= reversal_recent_drop
                and (recent_return_1 < 0 or slope_norm < 0)
                and previous_window_return > 0
            )
            panic_exit_signal = bool(recent_return_1 <= panic_drop_1 or recent_return_3 <= panic_drop_3)
            liquidity_stress_signal = bool(
                latest_spread_pct >= spread_stress_pct
                or fallback_ratio >= fallback_stress_ratio
            )

            trajectory_state = "stable"
            if panic_exit_signal:
                trajectory_state = "panic_exit"
            elif reversal_exit_signal:
                trajectory_state = "reversal_exit"
            elif liquidity_stress_signal:
                trajectory_state = "liquidity_stress"
            elif profit_lock_signal:
                trajectory_state = "profit_lock"
            elif recent_return_1 > 0 and slope_norm > 0:
                trajectory_state = "trend_up"
            elif recent_return_1 < 0 and slope_norm < 0:
                trajectory_state = "trend_down"

            trade_key = f"{str(work['token_id'].iloc[-1] or '').strip()}|{str(work['condition_id'].iloc[-1] or '').strip()}|{str(work['outcome_side'].iloc[-1] or '').strip()}"
            metrics[trade_key] = {
                "position_key": position_key,
                "trajectory_state": trajectory_state,
                "current_price": current_price,
                "entry_price": entry_price,
                "runup_from_entry": round(runup_from_entry, 6),
                "drawdown_from_peak": round(drawdown_from_peak, 6),
                "recent_return_1": round(recent_return_1, 6),
                "recent_return_3": round(recent_return_3, 6),
                "previous_window_return": round(previous_window_return, 6),
                "slope_short": round(float(slope_norm), 6),
                "volatility_short": round(volatility_short, 6),
                "spread_pct": round(latest_spread_pct, 6),
                "fallback_ratio": round(fallback_ratio, 6),
                "profit_lock_signal": profit_lock_signal,
                "reversal_exit_signal": reversal_exit_signal,
                "panic_exit_signal": panic_exit_signal,
                "liquidity_stress_signal": liquidity_stress_signal,
            }
        return metrics

    def apply_trajectory_metrics(self, positions_df: pd.DataFrame | None, metrics: dict | None = None):
        if positions_df is None or positions_df.empty:
            return pd.DataFrame() if positions_df is None else positions_df
        metrics = metrics or {}
        if not metrics:
            return positions_df

        df = positions_df.copy()
        df["position_key"] = df.apply(self._position_key, axis=1)
        for col in [
            "trajectory_state",
            "drawdown_from_peak",
            "recent_return_3",
            "runup_from_entry",
            "volatility_short",
            "fallback_ratio",
            "spread_pct",
        ]:
            if col not in df.columns:
                if col == "trajectory_state":
                    df[col] = pd.Series([None] * len(df), dtype="object")
                else:
                    df[col] = np.nan

        for idx, row in df.iterrows():
            trade_key = f"{str(row.get('token_id', '') or '').strip()}|{str(row.get('condition_id', '') or '').strip()}|{str(row.get('outcome_side', '') or '').strip()}"
            telemetry_row = metrics.get(trade_key, {})
            if not telemetry_row:
                continue
            for col in [
                "trajectory_state",
                "drawdown_from_peak",
                "recent_return_3",
                "runup_from_entry",
                "volatility_short",
                "fallback_ratio",
                "spread_pct",
            ]:
                if telemetry_row.get(col) is not None:
                    df.at[idx, col] = telemetry_row.get(col)
        return df
