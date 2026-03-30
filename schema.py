CANONICAL = {
    "wallet_col": "wallet_copied",
    "market_col": "market_title",
    "token_col": "token_id",
    "condition_col": "condition_id",
    "order_side_col": "order_side",
    "outcome_side_col": "outcome_side",
    "intent_col": "entry_intent",
    "timestamp_col": "timestamp",
    "price_col": "entry_price",
    "return_target_col": "forward_return_15m",
    "btc_return_target_col": "future_return",
    "classification_target_col": "tp_before_sl_60m",
}

ALIASES = {
    "wallet_copied": ["wallet_copied", "wallet", "trader_wallet"],
    "market_title": ["market_title", "market", "question", "title"],
    "market_id": ["market_id", "id"],
    "token_id": ["token_id"],
    "condition_id": ["condition_id"],
    "order_side": ["order_side", "trade_side"],
    "outcome_side": ["outcome_side", "side"],
    "entry_intent": ["entry_intent"],
    "timestamp": ["timestamp", "updated_at", "created_at"],
    "entry_price": ["entry_price", "price"],
    "forward_return_15m": ["forward_return_15m"],
    "future_return": ["future_return"],
    "tp_before_sl_60m": ["tp_before_sl_60m", "tp_hit_before_sl"],
}


def first_present(row_or_columns, canonical_key):
    aliases = ALIASES.get(canonical_key, [canonical_key])
    for key in aliases:
        if isinstance(row_or_columns, dict) and key in row_or_columns:
            return row_or_columns.get(key)
        if hasattr(row_or_columns, "__contains__") and key in row_or_columns:
            return key
    return None


def normalize_dataframe_columns(df):
    if df is None or getattr(df, "empty", False):
        return df
    out = df.copy()
    for canonical, aliases in ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            if alias in out.columns:
                out = out.rename(columns={alias: canonical})
                break
    return out

