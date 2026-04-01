# Neural Network For Crypto (Live Trading)

This project runs a live Polymarket trading loop with:
- model-based entries and exits,
- strict reconciliation against exchange state,
- position tracking for up to 5 concurrent live positions,
- anti-ghost safeguards for open/closed trades.

## Live Execution Flow

Per cycle, the bot now syncs in this order:
1. Sync open orders + trades from Polymarket into local DB (`orders`, `fills`)
2. Rebuild `live_positions` from reconciled fills
3. Reconcile runtime in-memory trades against live DB positions
4. Run mismatch checks (missing local/remote orders/trades)
5. If mismatch is detected, freeze new entries for that cycle
6. Only then evaluate fresh signals and place new entries

This prevents opening new trades while state is out-of-sync and reduces ghost trades.

## Intelligence & Market Context Engine
The bot evaluates trades not just by copying wallets, but by measuring deep macroeconomic contexts across 5 pillars before entering a trade.

1. **Macro & Liquidity (Pillar 1):** Tracks traditional finance proxies (DXY, 10Y Yields, S&P 500) via `yfinance` to gauge capital liquidity.
2. **On-Chain Fundamentals (Pillar 2):** Tracks Bitcoin Hash Rate via CoinMetrics.
3. **Technical Analysis (Pillar 3):** Fetches live Binance Klines to compute structural trend biases (200 SMA, 21 EMA).
4. **Sentiment & Derivatives (Pillar 4):** Reads Fear & Greed index and tracks Binance Perpetual Funding Rates to detect overheated leverage (vulnerable to squeezes).
5. **Order Flow & Taker Imbalance (Pillar 5):** Synthesizes completely autonomous trading signals simply by spotting massive asymmetric volume flows in Polymarket orderbooks (`order_flow_analyzer.py`).

**Active Market Hunting:**
The `EntryRuleLayer` dynamically shifts its AI confidence requirements based on the macro trend score. If the market is completely bullish, it lowers the entry threshold to "gobble up" longs. If the market is flashing *Overheated Long*, it triggers a hard veto block to protect capital.

## Position Management

- Max concurrent live positions: `TradingConfig.MAX_CONCURRENT_POSITIONS` (default `5`)
- Each position is tracked independently (`token_id | condition_id | outcome_side`)
- Exit paths include:
  - take-profit / stop-loss / trailing stop / time stop
  - model-target take-profit (`take_profit_model_target`)
  - exchange/manual-close detection via conditional token balance checks

## Balance and Risk Controls

Live sizing uses **API/CLOB spendable balance** (not on-chain fallback) for:
- bet-size calculation,
- pre-order funding checks.

Risk controls are dynamic:
- reserve capital (`CAPITAL_RESERVE_PCT`) to avoid full-wallet deployment
- per-trade cap by risk percent (`MAX_RISK_PER_TRADE_PCT`)
- hard absolute cap (`HARD_MAX_BET_USDC`)
- exchange minimum notional enforcement (`MIN_BET_USDC`, plus $1 floor on market BUY)

## Runtime State and Corruption Recovery

The system includes DB integrity checks and optional runtime reset:
- integrity check via SQLite `PRAGMA quick_check`
- optional auto-reset trigger:
  - `RESET_RUNTIME_STATE_ON_DB_CORRUPTION=true`
- reset archives runtime DB/CSVs and rebuilds schema
- model weights in `weights/` are preserved

## Useful Commands

Run live bot:
```bash
python run_bot.py
```

Audit DB/CSV sync and ghost-trade risk:
```bash
python audit_runtime_state.py --logs-dir logs
```

Audit + reset only if corruption is detected:
```bash
python audit_runtime_state.py --logs-dir logs --reset-if-corrupt
```

## Key Files

- `supervisor.py`: main live loop, pre-cycle reconciliation gate, entry/exit orchestration
- `reconciliation_service.py`: exchange order/fill sync + mismatch report
- `live_position_book.py`: rebuild and verify live positions from fills + exchange balances
- `trade_manager.py`: position lifecycle + policy/model exit handling
- `order_manager.py`: order placement, fill tracking, API-balance-based risk checks
- `money_manager.py`: dynamic position sizing and risk caps
- `audit_runtime_state.py`: operational audit for drift/ghost/corruption checks
