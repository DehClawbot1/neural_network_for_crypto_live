# Trading system audit – end-to-end findings and fixes

Repository reviewed: `DehClawbot1/neural_network_for_crypto_live`

This audit traced the path from data -> feature generation -> model scoring -> signal gating -> order placement -> fill capture -> position tracking -> dashboard state.

## Patched by the Python patch file

The attached patch script updates these files:

- `config.py`
- `execution_client.py`
- `money_manager.py`
- `strategy_layers.py`
- `trade_manager.py`
- `order_manager.py`
- `run_bot.py`
- `dashboard.py`
- `supervisor.py`

## Critical issues

### 1) Paper positions can be wiped by live reconciliation logic
- **Where**: `trade_manager.py`, `_maybe_load_reconciled_positions()`
- **Why it happens**: `get_open_positions()` always tries to reconcile from `live_positions`, even in paper mode. If `live_positions` is empty, it clears `active_trades`.
- **Impact**: paper trades disappear, PnL resets, max-position checks become wrong, dashboard goes out of sync.
- **Fix**: only reconcile from `live_positions` when `TRADING_MODE=live`.
- **Patched**: yes.

### 2) Live positions can be wiped every cycle
- **Where**: `supervisor.py` and `trade_manager.py`, `reconcile_live_positions()`
- **Why it happens**: supervisor calls `trade_manager.reconcile_live_positions(execution_client)` without passing a reconciled DataFrame. The method then clears `active_trades` when no DataFrame is supplied.
- **Impact**: bot buys, then loses in-memory tracking of the open position, so later exit logic may fail or act on stale state.
- **Fix**: fetch reconciled positions from `LivePositionBook` and pass them into `reconcile_live_positions()`. Also make `reconcile_live_positions()` rebuild from DB instead of clearing state when no DataFrame is passed.
- **Patched**: yes.

### 3) Rule-based live exits are submitted three times
- **Where**: `supervisor.py`, closed-trade processing block
- **Why it happens**: there are three duplicated SELL-submission loops after `trade_manager.process_exits()`.
- **Impact**: duplicate SELL orders, race conditions, over-selling attempts, extra cancellations, and loss of sync with exchange state.
- **Fix**: collapse that logic into one SELL submission loop.
- **Patched**: yes.

### 4) Fill records can lose token/position identity
- **Where**: `order_manager.py`, `wait_for_fill()` and `record_fill()`
- **Why it happens**:
  - `wait_for_fill()` trusts the exchange order-status payload to include `token_id`, `condition_id`, `outcome_side`, and `side`.
  - `record_fill()` writes only `fill_id`, `order_id`, `token_id`, `price`, `size`, `filled_at` into SQLite, dropping condition/outcome/side.
- **Impact**: `LivePositionBook` may not be able to rebuild open positions immediately after a fill. This is especially dangerous right after entry.
- **Fix**: recover missing order context from DB/CSV, pass fallback context from supervisor, and persist full fill context into SQLite.
- **Patched**: yes.

### 5) Paper entry price is inconsistent between logs and trade state
- **Where**: `supervisor.py` paper-entry path, `trade_manager.py`
- **Why it happens**: supervisor computes `fill_price = quote_entry_price(...)` and logs that as the paper fill, but `TradeManager.handle_signal()` opens the trade using `current_price` from the signal row.
- **Impact**: dashboard PnL and stored state disagree with logged execution price.
- **Fix**: inject the quoted fill price into `signal_row["current_price"]` and `signal_row["entry_price"]` before creating the paper trade.
- **Patched**: yes.

### 6) Trade keys are not stable enough for multi-position state
- **Where**: `supervisor.py` and `trade_manager.py`
- **Why it happens**: the runtime key uses `market_title-outcome_side`. That is not unique enough across contracts and can break removal logic.
- **Impact**: duplicate suppression fails, exits can pop the wrong trade, and multiple positions are not managed independently.
- **Fix**: use `TOKEN::{token_id}::{condition_id}::{outcome_side}` whenever token data exists.
- **Patched**: yes.

## High issues

### 7) Max positions is 4, not 5
- **Where**: `config.py`
- **Impact**: bot cannot meet your requested 5-position limit.
- **Fix**: set `MAX_CONCURRENT_POSITIONS = 5`.
- **Patched**: yes.

### 8) Duplicate entries can still slip through in one cycle
- **Where**: `supervisor.py`
- **Why it happens**: `active_trade_keys` is computed once before scanning signals and is not refreshed after a trade is opened.
- **Impact**: duplicate market entries in the same cycle.
- **Fix**: add the new key to `active_trade_keys` immediately after trade creation.
- **Patched**: yes.

### 9) `USE_MARKET_ORDERS` config is ignored
- **Where**: `supervisor.py`
- **Why it happens**: live entry always uses `submit_entry()` even though `config.py` has `USE_MARKET_ORDERS = True`.
- **Impact**: config says one thing, execution path does another.
- **Fix**: honor `TradingConfig.USE_MARKET_ORDERS` and route to `submit_market_entry()` when enabled.
- **Patched**: yes.

### 10) Entry gating is too optimistic
- **Where**: `strategy_layers.py`, `signal_engine.py`, `supervisor.py`
- **Why it happens**:
  - entry filtering relies heavily on blended confidence, not actual expected return quality.
  - `stage2_temporal_inference` is merged using `max()` for `p_tp_before_sl`, which biases the final probability upward.
- **Impact**: system can enter trades with weak or negative edge.
- **Fix**:
  - harden `EntryRuleLayer` to require positive `expected_return` and positive `edge_score`.
  - replace optimistic `max()` merge with `mean()`.
- **Patched**: yes.
- **Note**: I cannot prove the model is strong enough to justify live trading without the actual trained artifacts and eval outputs. From code review alone, the runtime gating was too permissive.

### 11) Wrong default balance type when `asset_type` is omitted
- **Where**: `execution_client.py`, `_build_balance_params()`
- **Why it happens**: `asset_type=None` falls into the CONDITIONAL branch instead of COLLATERAL.
- **Impact**: callers that omit the argument can read the wrong balance and make bad decisions.
- **Fix**: default `None` to `"COLLATERAL"`.
- **Patched**: yes.

### 12) Exposure logic double-counts open exposure
- **Where**: `money_manager.py`
- **Why it happens**: max total exposure is based on `available_balance`, which is already post-entry free cash in many call paths, and then `current_exposure` is subtracted again.
- **Impact**: the bot understates remaining capacity and can size trades inconsistently.
- **Fix**: compute exposure ceiling from effective equity (`available_balance + current_exposure`).
- **Patched**: yes.

### 13) Dashboard does not show active live orders/fills
- **Where**: `dashboard.py`
- **Impact**: dashboard misses a key part of real bot state.
- **Fix**: load `live_orders.csv` and `live_fills.csv`, and render a live order/fill panel.
- **Patched**: yes.

### 14) Live execution log mixes shares and USDC
- **Where**: `supervisor.py`, `log_live_fill_event()` call sites
- **Why it happens**: reduce/exit paths pass `actual_fill_size` into a function that stores `size_usdc`.
- **Impact**: analytics, history, and dashboard views become numerically wrong.
- **Fix**: store both `shares` and correct `size_usdc = shares * fill_price`.
- **Patched**: yes.

## Medium issues

### 15) MoneyManager accounting is duplicated and inconsistent
- **Where**: `trade_manager.py` and `supervisor.py`
- **Why it happens**: closed trades are recorded in multiple places, including a fresh `MoneyManager()` instance inside supervisor.
- **Impact**: win/loss streak state becomes unreliable.
- **Fix**: make supervisor’s shared `_money_mgr` the single runtime source of truth.
- **Patched**: partially. The TradeManager-local duplicate accounting block is removed, and supervisor keeps one shared updater.

### 16) Reconciliation only syncs open orders
- **Where**: `reconciliation_service.py`
- **Impact**: canceled or stale local orders may linger as open locally.
- **Fix proposal**: add explicit local closeout logic for orders that vanish remotely and are older than a safety threshold.
- **Patched**: no.

### 17) CSV append logs can drift from DB truth
- **Where**: `order_manager.py`, `log_loader.py`
- **Impact**: dashboard views based on CSV can show duplicate or stale rows even when SQLite is correct.
- **Fix proposal**: eventually make SQLite the main state source for dashboard status panels.
- **Patched**: no.

## Position handling review

After patching, the system is materially safer for 5 simultaneous positions because:

- keys are token-based instead of title-based
- state is no longer cleared in paper mode
- live reconciliation no longer nukes open trades
- new entries refresh `active_trade_keys` immediately
- exit SELLs are submitted once instead of three times
- fills preserve full position identity in SQLite

That still does **not** make it production-safe by itself. The remaining major risk is model quality. The runtime logic is now less optimistic, but actual live readiness still depends on the trained artifacts and evaluation files.

## Best next checks after patching

Run these after applying the patch:

```powershell
python -m py_compile supervisor.py trade_manager.py order_manager.py run_bot.py dashboard.py
python run_bot.py
streamlit run dashboard.py
```

Then verify these manually on a tiny balance / paper setup first:

1. Start bot and confirm it asks for `POLYMARKET_SIGNATURE_TYPE`.
2. Open two or more paper trades in the same run and confirm they persist.
3. In live mode, confirm a fill creates a position with correct `token_id`, `entry_price`, `shares`, and `unrealized_pnl`.
4. Trigger one exit and confirm only one SELL is submitted.
5. Confirm dashboard shows live orders and live fills.

