from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


def _safe_float(value, default=0.0) -> float:
    try:
        num = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(num):
        return float(default)
    return float(num)


def _clip01(value) -> float:
    return float(min(1.0, max(0.0, _safe_float(value, 0.0))))


def _norm_side(value: object) -> str:
    side = str(value or "").strip().upper()
    if side in {"YES", "UP", "LONG", "BULLISH"}:
        return "YES"
    if side in {"NO", "DOWN", "SHORT", "BEARISH"}:
        return "NO"
    return side or "UNKNOWN"


def _norm_trade_side(value: object) -> str:
    side = str(value or "").strip().upper()
    return side if side in {"BUY", "SELL"} else "BUY"


def _market_key(row: dict) -> str:
    for key in ("condition_id", "market_slug", "market_title", "market"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return "unknown_market"


def _other_side(side: str) -> str:
    return "NO" if side == "YES" else "YES"


def _env_float(name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        value = float(os.getenv(name, str(default)) or default)
    except Exception:
        value = float(default)
    return max(float(minimum), min(float(maximum), value))


def _env_int(name: str, default: int, minimum: int = 0, maximum: int = 1_000_000) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except Exception:
        value = int(default)
    return max(int(minimum), min(int(maximum), value))


def source_wallet_signal_matches_trade(signal_row: dict, trade) -> bool:
    signal_wallet = str(signal_row.get("trader_wallet") or "").strip().lower()
    trade_wallet = str(getattr(trade, "source_wallet", "") or "").strip().lower()
    if trade_wallet and signal_wallet and trade_wallet != signal_wallet:
        return False

    signal_token = str(signal_row.get("token_id") or "").strip()
    trade_token = str(getattr(trade, "token_id", "") or "").strip()
    if signal_token and trade_token and signal_token != trade_token:
        return False

    signal_condition = str(signal_row.get("condition_id") or "").strip()
    trade_condition = str(getattr(trade, "condition_id", "") or "").strip()
    if signal_condition and trade_condition and signal_condition != trade_condition:
        return False

    signal_side = _norm_side(signal_row.get("outcome_side", signal_row.get("side")))
    trade_side = _norm_side(getattr(trade, "outcome_side", ""))
    if signal_side and trade_side and signal_side != trade_side:
        return False
    return True


def resolve_source_wallet_reduce_fraction(signal_row: dict, default_fraction: float = 0.5) -> float:
    raw_fraction = _safe_float(signal_row.get("source_wallet_reduce_fraction"), default_fraction)
    if raw_fraction <= 0:
        raw_fraction = float(default_fraction)
    return max(0.05, min(1.0, float(raw_fraction)))


def should_convert_reduce_to_exit(
    *,
    total_shares: float,
    reduce_fraction: float,
    reference_price: float,
    min_reduce_notional: float,
    min_remainder_notional: float,
) -> bool:
    total_shares = max(0.0, _safe_float(total_shares, 0.0))
    reduce_fraction = max(0.0, min(1.0, _safe_float(reduce_fraction, 0.0)))
    reference_price = max(0.0, _safe_float(reference_price, 0.0))
    if total_shares <= 0 or reduce_fraction <= 0 or reference_price <= 0:
        return True
    reduce_shares = total_shares * reduce_fraction
    remaining_shares = max(0.0, total_shares - reduce_shares)
    reduce_notional = reduce_shares * reference_price
    remainder_notional = remaining_shares * reference_price
    if reduce_fraction >= 0.999:
        return True
    if reduce_notional < max(0.0, min_reduce_notional):
        return True
    if remaining_shares > 1e-9 and remainder_notional < max(0.0, min_remainder_notional):
        return True
    return False


@dataclass
class _MarketWalletState:
    exposure_by_side: Dict[str, float] = field(default_factory=lambda: {"YES": 0.0, "NO": 0.0})
    avg_entry_by_side: Dict[str, float] = field(default_factory=dict)
    last_add_by_side: Dict[str, Optional[str]] = field(default_factory=dict)
    last_reduce_by_side: Dict[str, Optional[str]] = field(default_factory=dict)
    last_close_by_side: Dict[str, Optional[str]] = field(default_factory=dict)

    def dominant_side(self) -> str:
        yes = _safe_float(self.exposure_by_side.get("YES"), 0.0)
        no = _safe_float(self.exposure_by_side.get("NO"), 0.0)
        if yes <= 1e-9 and no <= 1e-9:
            return "FLAT"
        return "YES" if yes >= no else "NO"

    def net_exposure(self) -> float:
        yes = _safe_float(self.exposure_by_side.get("YES"), 0.0)
        no = _safe_float(self.exposure_by_side.get("NO"), 0.0)
        return abs(yes - no)


class WalletStateEngine:
    def __init__(self):
        self.fresh_minutes = _env_int("SOURCE_WALLET_SIGNAL_FRESH_MINUTES", 90, minimum=1, maximum=24 * 60)
        self.sharp_reduce_threshold = _env_float("SOURCE_WALLET_SHARP_REDUCE_THRESHOLD", 0.55, minimum=0.05, maximum=1.0)
        self.min_wallet_quality = _env_float("SOURCE_WALLET_MIN_QUALITY", 0.45, minimum=0.0, maximum=1.0)
        self.conflict_margin = _env_float("SOURCE_WALLET_CONFLICT_MARGIN", 0.08, minimum=0.0, maximum=1.0)

    def build_state_signals(
        self,
        raw_trades_df: pd.DataFrame,
        wallet_meta: Optional[Dict[str, dict]] = None,
    ) -> pd.DataFrame:
        if raw_trades_df is None or raw_trades_df.empty:
            return pd.DataFrame()

        work_df = raw_trades_df.copy()
        wallet_meta = wallet_meta or {}
        work_df["timestamp"] = pd.to_datetime(work_df.get("timestamp"), utc=True, errors="coerce")
        work_df = work_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        states: Dict[tuple, _MarketWalletState] = {}
        emitted_rows: List[dict] = []
        now_utc = pd.Timestamp.now(tz="UTC")

        for _, raw_row in work_df.iterrows():
            trade = raw_row.to_dict()
            wallet = str(trade.get("trader_wallet") or "").strip().lower()
            market_key = _market_key(trade)
            outcome_side = _norm_side(trade.get("outcome_side", trade.get("side")))
            order_side = _norm_trade_side(trade.get("order_side", trade.get("trade_side", trade.get("side"))))
            if not wallet or outcome_side not in {"YES", "NO"}:
                continue

            state_key = (wallet, market_key)
            state = states.setdefault(state_key, _MarketWalletState())
            current_ts = pd.to_datetime(trade.get("timestamp"), utc=True, errors="coerce")
            if pd.isna(current_ts):
                continue

            wallet_detail = wallet_meta.get(wallet, {})
            wallet_quality_score = _clip01(wallet_detail.get("quality_score", trade.get("wallet_quality_score", 0.50)))
            wallet_watchlist_approved = bool(wallet_detail.get("approved", trade.get("wallet_watchlist_approved", True)))

            same_before = _safe_float(state.exposure_by_side.get(outcome_side), 0.0)
            opposite_side = _other_side(outcome_side)
            opposite_before = _safe_float(state.exposure_by_side.get(opposite_side), 0.0)
            size = max(_safe_float(trade.get("size"), 0.0), 0.0)
            price = _safe_float(trade.get("price"), 0.0)
            if size <= 0:
                continue

            if order_side == "BUY":
                emitted_rows.extend(
                    self._handle_buy(
                        trade=trade,
                        state=state,
                        wallet=wallet,
                        outcome_side=outcome_side,
                        opposite_side=opposite_side,
                        same_before=same_before,
                        opposite_before=opposite_before,
                        size=size,
                        price=price,
                        current_ts=current_ts,
                        now_utc=now_utc,
                        wallet_quality_score=wallet_quality_score,
                        wallet_watchlist_approved=wallet_watchlist_approved,
                    )
                )
            else:
                emitted_rows.extend(
                    self._handle_sell(
                        trade=trade,
                        state=state,
                        wallet=wallet,
                        outcome_side=outcome_side,
                        same_before=same_before,
                        size=size,
                        current_ts=current_ts,
                        now_utc=now_utc,
                        wallet_quality_score=wallet_quality_score,
                        wallet_watchlist_approved=wallet_watchlist_approved,
                    )
                )

        out_df = pd.DataFrame(emitted_rows)
        if out_df.empty:
            return out_df
        out_df = self._apply_market_consensus(out_df)
        out_df = out_df.sort_values("timestamp", ascending=False).reset_index(drop=True)
        return out_df

    def _handle_buy(
        self,
        *,
        trade: dict,
        state: _MarketWalletState,
        wallet: str,
        outcome_side: str,
        opposite_side: str,
        same_before: float,
        opposite_before: float,
        size: float,
        price: float,
        current_ts,
        now_utc,
        wallet_quality_score: float,
        wallet_watchlist_approved: bool,
    ) -> List[dict]:
        rows = []

        if opposite_before > 1e-9 and same_before <= 1e-9:
            rows.append(
                self._build_signal(
                    trade=trade,
                    wallet=wallet,
                    outcome_side=opposite_side,
                    entry_intent="CLOSE_LONG",
                    position_event="REVERSAL_EXIT",
                    net_increase=False,
                    size_delta=opposite_before,
                    size_delta_ratio=1.0,
                    current_net_exposure=0.0,
                    avg_entry=state.avg_entry_by_side.get(opposite_side),
                    current_direction=outcome_side,
                    confidence=self._event_confidence(
                        wallet_quality_score=wallet_quality_score,
                        dominance=1.0,
                        size_change_ratio=1.0,
                        is_exit=True,
                    ),
                    timestamp=current_ts,
                    now_utc=now_utc,
                    wallet_quality_score=wallet_quality_score,
                    wallet_watchlist_approved=wallet_watchlist_approved,
                    reduce_fraction=1.0,
                    exit_signal=True,
                    reduce_signal=False,
                    reversal_signal=True,
                )
            )
            state.exposure_by_side[opposite_side] = 0.0
            state.last_close_by_side[opposite_side] = current_ts.isoformat()

        same_after = same_before + size
        prior_avg = state.avg_entry_by_side.get(outcome_side)
        if prior_avg is None or same_before <= 1e-9:
            new_avg = price
            event = "NEW_ENTRY"
        else:
            new_avg = ((prior_avg * same_before) + (price * size)) / max(same_after, 1e-9)
            event = "SCALE_IN"

        state.exposure_by_side[outcome_side] = same_after
        state.avg_entry_by_side[outcome_side] = new_avg
        state.last_add_by_side[outcome_side] = current_ts.isoformat()

        total_after = max(1e-9, same_after + _safe_float(state.exposure_by_side.get(opposite_side), 0.0))
        dominance = same_after / total_after
        size_change_ratio = size / max(same_after, 1e-9)

        rows.append(
            self._build_signal(
                trade=trade,
                wallet=wallet,
                outcome_side=outcome_side,
                entry_intent="OPEN_LONG",
                position_event=event if event != "NEW_ENTRY" or opposite_before <= 1e-9 else "REVERSAL_ENTRY",
                net_increase=True,
                size_delta=size,
                size_delta_ratio=size_change_ratio,
                current_net_exposure=state.net_exposure(),
                avg_entry=new_avg,
                current_direction=state.dominant_side(),
                confidence=self._event_confidence(
                    wallet_quality_score=wallet_quality_score,
                    dominance=dominance,
                    size_change_ratio=size_change_ratio,
                    is_exit=False,
                ),
                timestamp=current_ts,
                now_utc=now_utc,
                wallet_quality_score=wallet_quality_score,
                wallet_watchlist_approved=wallet_watchlist_approved,
                reduce_fraction=0.0,
                exit_signal=False,
                reduce_signal=False,
                reversal_signal=opposite_before > 1e-9,
            )
        )
        return rows

    def _handle_sell(
        self,
        *,
        trade: dict,
        state: _MarketWalletState,
        wallet: str,
        outcome_side: str,
        same_before: float,
        size: float,
        current_ts,
        now_utc,
        wallet_quality_score: float,
        wallet_watchlist_approved: bool,
    ) -> List[dict]:
        if same_before <= 1e-9:
            return []

        sold_size = min(size, same_before)
        same_after = max(0.0, same_before - sold_size)
        reduce_fraction = sold_size / max(same_before, 1e-9)
        state.exposure_by_side[outcome_side] = same_after
        state.last_reduce_by_side[outcome_side] = current_ts.isoformat()
        if same_after <= 1e-9:
            state.last_close_by_side[outcome_side] = current_ts.isoformat()

        if same_after <= 1e-9:
            event = "FULL_EXIT"
            exit_signal = True
            reduce_signal = False
        elif reduce_fraction >= self.sharp_reduce_threshold:
            event = "SHARP_REDUCE"
            exit_signal = True
            reduce_signal = True
        else:
            event = "PARTIAL_EXIT"
            exit_signal = False
            reduce_signal = False

        if not exit_signal:
            return []

        return [
            self._build_signal(
                trade=trade,
                wallet=wallet,
                outcome_side=outcome_side,
                entry_intent="CLOSE_LONG",
                position_event=event,
                net_increase=False,
                size_delta=sold_size,
                size_delta_ratio=reduce_fraction,
                current_net_exposure=state.net_exposure(),
                avg_entry=state.avg_entry_by_side.get(outcome_side),
                current_direction=state.dominant_side(),
                confidence=self._event_confidence(
                    wallet_quality_score=wallet_quality_score,
                    dominance=1.0 if same_after <= 1e-9 else reduce_fraction,
                    size_change_ratio=reduce_fraction,
                    is_exit=True,
                ),
                timestamp=current_ts,
                now_utc=now_utc,
                wallet_quality_score=wallet_quality_score,
                wallet_watchlist_approved=wallet_watchlist_approved,
                reduce_fraction=reduce_fraction,
                exit_signal=True,
                reduce_signal=reduce_signal,
                reversal_signal=False,
            )
        ]

    def _build_signal(
        self,
        *,
        trade: dict,
        wallet: str,
        outcome_side: str,
        entry_intent: str,
        position_event: str,
        net_increase: bool,
        size_delta: float,
        size_delta_ratio: float,
        current_net_exposure: float,
        avg_entry: Optional[float],
        current_direction: str,
        confidence: float,
        timestamp,
        now_utc,
        wallet_quality_score: float,
        wallet_watchlist_approved: bool,
        reduce_fraction: float,
        exit_signal: bool,
        reduce_signal: bool,
        reversal_signal: bool,
    ) -> dict:
        freshness_seconds = max(0.0, (now_utc - timestamp).total_seconds())
        freshness_score = _clip01(1.0 - (freshness_seconds / max(self.fresh_minutes * 60.0, 1.0)))
        signal = dict(trade)
        signal["trader_wallet"] = wallet
        signal["outcome_side"] = outcome_side
        signal["side"] = outcome_side
        signal["entry_intent"] = entry_intent
        signal["source_wallet_position_event"] = position_event
        signal["source_wallet_net_position_increased"] = bool(net_increase)
        signal["source_wallet_current_net_exposure"] = float(current_net_exposure)
        signal["source_wallet_average_entry"] = _safe_float(avg_entry, 0.0) if avg_entry is not None else None
        signal["source_wallet_last_add"] = timestamp.isoformat() if entry_intent == "OPEN_LONG" else None
        signal["source_wallet_last_reduce"] = timestamp.isoformat() if exit_signal else None
        signal["source_wallet_last_close"] = timestamp.isoformat() if position_event in {"FULL_EXIT", "REVERSAL_EXIT"} else None
        signal["source_wallet_current_direction"] = current_direction
        signal["source_wallet_direction_confidence"] = round(_clip01(confidence), 4)
        signal["source_wallet_size_delta"] = float(size_delta)
        signal["source_wallet_size_delta_ratio"] = float(_clip01(size_delta_ratio))
        signal["source_wallet_reduce_fraction"] = float(_clip01(reduce_fraction))
        signal["source_wallet_state_freshness_seconds"] = float(freshness_seconds)
        signal["source_wallet_freshness_score"] = float(freshness_score)
        signal["source_wallet_fresh"] = bool(freshness_seconds <= (self.fresh_minutes * 60.0))
        signal["wallet_watchlist_approved"] = bool(wallet_watchlist_approved)
        signal["wallet_quality_score"] = float(_clip01(wallet_quality_score))
        signal["source_wallet_exit_signal"] = bool(exit_signal)
        signal["source_wallet_reduce_signal"] = bool(reduce_signal)
        signal["source_wallet_reversal_signal"] = bool(reversal_signal)
        gate_reasons = []
        if not wallet_watchlist_approved:
            gate_reasons.append("wallet_not_approved")
        if wallet_quality_score < self.min_wallet_quality:
            gate_reasons.append("wallet_quality_below_min")
        if not signal["source_wallet_fresh"]:
            gate_reasons.append("wallet_state_stale")
        if not (net_increase or exit_signal):
            gate_reasons.append("wallet_net_position_not_increased")
        signal["wallet_state_gate_pass"] = bool(not gate_reasons)
        signal["wallet_state_gate_reason"] = ",".join(gate_reasons)
        return signal

    def _event_confidence(
        self,
        *,
        wallet_quality_score: float,
        dominance: float,
        size_change_ratio: float,
        is_exit: bool,
    ) -> float:
        base = 0.35 if not is_exit else 0.45
        score = base + (wallet_quality_score * 0.30) + (_clip01(dominance) * 0.20) + (_clip01(size_change_ratio) * 0.15)
        return _clip01(score)

    def _apply_market_consensus(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        if signals_df.empty:
            return signals_df
        out = signals_df.copy()
        out["wallet_agreement_score"] = 0.5
        out["wallet_conflict_with_stronger"] = False
        out["wallet_stronger_conflict_score"] = 0.0
        out["wallet_support_strength"] = 0.0

        market_cols = ["condition_id", "market_slug", "market_title"]
        key_col = next((c for c in market_cols if c in out.columns), None)
        if key_col is None:
            return out

        entry_mask = out["entry_intent"].astype(str).str.upper() == "OPEN_LONG"
        if not entry_mask.any():
            return out

        for market_key, group in out[entry_mask].groupby(out[key_col].astype(str)):
            group = group.copy()
            if group.empty:
                continue
            eligible_group = group[group["wallet_state_gate_pass"].fillna(False).astype(bool)].copy()
            for idx, row in group.iterrows():
                row_side = str(row.get("outcome_side", "")).upper()
                same_side = eligible_group[eligible_group["outcome_side"].astype(str).str.upper() == row_side]
                opp_side = eligible_group[eligible_group["outcome_side"].astype(str).str.upper() != row_side]
                support_strength = (
                    same_side["wallet_quality_score"].astype(float).fillna(0.0)
                    * same_side["source_wallet_direction_confidence"].astype(float).fillna(0.0)
                ).sum() if not same_side.empty else 0.0
                oppose_strength = (
                    opp_side["wallet_quality_score"].astype(float).fillna(0.0)
                    * opp_side["source_wallet_direction_confidence"].astype(float).fillna(0.0)
                ).max() if not opp_side.empty else 0.0
                total_strength = max(support_strength + oppose_strength, 1e-9)
                agreement_score = support_strength / total_strength
                conflict = oppose_strength > (support_strength + self.conflict_margin)
                out.at[idx, "wallet_agreement_score"] = round(_clip01(agreement_score), 4)
                out.at[idx, "wallet_conflict_with_stronger"] = bool(conflict)
                out.at[idx, "wallet_stronger_conflict_score"] = round(float(oppose_strength), 4)
                out.at[idx, "wallet_support_strength"] = round(float(support_strength), 4)
                if conflict:
                    existing_reason = str(out.at[idx, "wallet_state_gate_reason"] or "").strip()
                    reason_parts = [part for part in existing_reason.split(",") if part]
                    if "conflict_with_stronger_wallet" not in reason_parts:
                        reason_parts.append("conflict_with_stronger_wallet")
                    out.at[idx, "wallet_state_gate_reason"] = ",".join(reason_parts)
                    out.at[idx, "wallet_state_gate_pass"] = False
        return out
