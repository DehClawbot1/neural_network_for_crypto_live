import logging
from datetime import datetime


def apply_supervisor_ui_patch(supervisor_module):
    if getattr(supervisor_module, "_ui_patch_applied", False):
        return supervisor_module

    supervisor_module.shadow_logger = None

    def log_ranked_signal(signal_row):
        wallet_full = str(signal_row.get("trader_wallet", signal_row.get("wallet_copied", "Unknown")) or "Unknown")
        current_price = signal_row.get("current_price", signal_row.get("market_last_trade_price", signal_row.get("price")))
        recommended_action = signal_row.get(
            "recommended_action",
            signal_row.get("action", signal_row.get("entry_intent", "OPEN_LONG")),
        )
        reason = signal_row.get("reason", "")
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "market": signal_row.get("market_title", signal_row.get("market", "Unknown Market")),
            "market_title": signal_row.get("market_title", signal_row.get("market", "Unknown Market")),
            "market_slug": signal_row.get("market_slug"),
            "wallet_copied": wallet_full,
            "wallet_short": wallet_full[:8],
            "trader_wallet": wallet_full,
            "token_id": signal_row.get("token_id"),
            "condition_id": signal_row.get("condition_id"),
            "order_side": signal_row.get("order_side", signal_row.get("trade_side", "BUY")),
            "trade_side": signal_row.get("trade_side", signal_row.get("order_side", "BUY")),
            "outcome_side": signal_row.get("outcome_side", signal_row.get("side", "UNKNOWN")),
            "entry_intent": signal_row.get("entry_intent", "OPEN_LONG"),
            "side": signal_row.get("outcome_side", signal_row.get("side", "UNKNOWN")),
            "signal_label": signal_row.get("signal_label", "UNKNOWN"),
            "confidence": signal_row.get("confidence", 0.0),
            "reason": reason,
            "reason_summary": signal_row.get("reason_summary", reason),
            "recommended_action": recommended_action,
            "action": recommended_action,
            "market_url": signal_row.get("market_url"),
            "trader_win_rate": signal_row.get("trader_win_rate"),
            "normalized_trade_size": signal_row.get("normalized_trade_size"),
            "current_price": current_price,
            "market_last_trade_price": signal_row.get("market_last_trade_price", current_price),
            "price": signal_row.get("price", current_price),
            "best_bid": signal_row.get("best_bid"),
            "best_ask": signal_row.get("best_ask"),
            "time_left": signal_row.get("time_left"),
            "liquidity_score": signal_row.get("liquidity_score"),
            "volume_score": signal_row.get("volume_score"),
            "probability_momentum": signal_row.get("probability_momentum"),
            "volatility_score": signal_row.get("volatility_score"),
            "whale_pressure": signal_row.get("whale_pressure"),
            "market_structure_score": signal_row.get("market_structure_score"),
            "volatility_risk": signal_row.get("volatility_risk"),
            "time_decay_score": signal_row.get("time_decay_score"),
            "edge_score": signal_row.get("edge_score"),
            "expected_return": signal_row.get("expected_return"),
            "p_tp_before_sl": signal_row.get("p_tp_before_sl"),
            "risk_adjusted_ev": signal_row.get("risk_adjusted_ev"),
            "entry_ev": signal_row.get("entry_ev"),
            "execution_quality_score": signal_row.get("execution_quality_score"),
        }
        supervisor_module.append_csv_record(supervisor_module.SIGNALS_FILE, record)

    def execute_paper_trade(action, signal_row, fill_price=None):
        wallet_full = str(signal_row.get("trader_wallet", signal_row.get("wallet_copied", "Unknown")) or "Unknown")
        if action == 0:
            logging.info("Brain: IGNORE -> Skipping signal from %s", wallet_full[:8])
            return

        size = 10 if action == 1 else 50
        outcome_side = str(signal_row.get("outcome_side", signal_row.get("side", "UNKNOWN"))).upper()
        signal_price = float(signal_row.get("current_price", signal_row.get("price", 0.5)))
        fill = supervisor_module.quote_entry_price(signal_row) if fill_price is None else fill_price

        trade_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "market": signal_row.get("market_title", signal_row.get("market", "Unknown Market")),
            "market_title": signal_row.get("market_title", signal_row.get("market", "Unknown Market")),
            "wallet_copied": wallet_full,
            "wallet_short": wallet_full[:8],
            "trader_wallet": wallet_full,
            "token_id": signal_row.get("token_id"),
            "condition_id": signal_row.get("condition_id"),
            "outcome_side": outcome_side,
            "order_side": signal_row.get("order_side", "BUY"),
            "signal_price": round(signal_price, 3),
            "fill_price": round(fill, 3),
            "size_usdc": size,
            "signal_label": signal_row.get("signal_label", "UNKNOWN"),
            "confidence": signal_row.get("confidence", 0.0),
            "edge_score": signal_row.get("edge_score"),
            "expected_return": signal_row.get("expected_return"),
            "p_tp_before_sl": signal_row.get("p_tp_before_sl"),
            "action_type": "PAPER_TRADE",
        }

        logging.info(
            "Brain: FOLLOW -> Paper filled %s USDC on %s at $%.3f for '%s' | label=%s confidence=%.2f",
            size,
            outcome_side,
            fill,
            trade_record["market"],
            trade_record["signal_label"],
            float(trade_record["confidence"]),
        )

        try:
            supervisor_module.append_csv_record(supervisor_module.EXECUTION_FILE, trade_record)
        except Exception as exc:
            logging.error("[-] Failed to write to %s: %s", supervisor_module.EXECUTION_FILE, exc)

    supervisor_module.log_ranked_signal = log_ranked_signal
    supervisor_module.execute_paper_trade = execute_paper_trade
    supervisor_module._ui_patch_applied = True
    logging.info("[+] Applied supervisor UI patch for richer dashboard signal rows.")
    return supervisor_module
