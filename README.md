# PolyTrader: Neural Network For Crypto (Live Trading)

This repository runs an advanced, multi-family Polymarket trading system utilizing autonomous data loops alongside quantitative risk modeling. 

The project has recently undergone an extensive defense hardening overhaul, evolving far beyond a simple execution loop. It now contains four tightly interlocked operational layers:
1. `execution & runtime`: Live/Paper decision scaling, reconciliation checks, dead orderbook guards.
2. `defensive telemetry`: Real-time Continuous Drift Governance, Multi-Family Performance Governors.
3. `learning pipelines`: Feedback lifecycle attribution, retrainer verdicts, deployment pipeline (Shadow/Canary/Promotion logic).
4. `research & offline ablation`: Feature dataset building, multi-timeframe forecasting, order-book and on-chain microstructure processing.

---

## Architecture Overview

### 1. Risk Governance Layer (Hardened Defense)
The bot no longer relies on a single generalized capital risk check. It incorporates true multi-tier operational defense:
- **Family Performance Governors**: `btc` and `weather_temperature` possess distinct, independent alpha metrics. A bad streak in Weather will strangle Weather allocations without punishing BTC executions.
- **Continuous Drift Monitor**: Beyond lagging PnL metrics, the system inherently surveils the exact operational data flow natively.
    - Tracks calibration breakdown (Brier/PSI splits).
    - Tracks feature evaporation (`schema_health`) isolating API breakdown from market anomalies.
    - Tracks theoretical vs exact `edge_after_cost` mapping structural alpha decay against realized volume slippage limits.
    - Implements *streak tracking* requiring repeated regime breakdowns before tightening execution constraints directly without waiting for a candidate loop.
- **Level Chokes**: The system applies strict `[Level 1, Level 2A, Level 2B]` states inherently modifying `size_multiplier`, `force_min_size`, and liquidity gates autonomously.

### 2. Runtime Execution Layer
Core deterministic execution engine:
- [supervisor.py](supervisor.py): Main continuous cycle orchestration gracefully parsing drift state into candidate evaluations natively.
- [trade_manager.py](trade_manager.py): Trade lifecycle persistences, partial exit routing, learning classification definitions.
- [order_manager.py](order_manager.py): Real-time CLOB interactions resolving explicit execution fills against local synthetic limits.
- [live_position_book.py](live_position_book.py): Fills memory rebuilding explicit runtime gaps in states vs exchange APIs.
- [reconciliation_service.py](reconciliation_service.py): Local state vs Exchange truth. Identifies local desyncs immediately freezing new position entries until reconciliation succeeds.

### 3. Market Intelligence & Model Pipeline
The core intelligence stacks driving alpha decisions:
- [btc_live_price_tracker.py](btc_live_price_tracker.py): Tracks spot, futures indexing, basis divergences across venues.
- [orderbook_depth_features.py](orderbook_depth_features.py): Generates microstructure depth profiles natively analyzing whale imbalances, local slope constraints.
- [weather_temperature_strategy.py](weather_temperature_strategy.py): Fully independent forecast validation models isolating specific prediction sources against binary Polymarket outcomes.
- **Predictive Ensembles**: `[btc_multitimeframe.py]` merges 15m, 1h, 4h horizons via continuous exponential recency voting pipelines. 

### 4. Promotion, Shadow & Validation 
The platform scales updates rigorously via automated and metric-based workflows:
- **Replay / Paper Layer**: Live signals are consumed natively evaluating fill likelihood and queue mechanics using `ShadowPurgatory`.
- **Shadow Leaderboards**: `retrainer.py` routinely scores offline candidates applying parallel metrics vs the active champion output without real capital risk.
- **Promotion Constraints**: Live evaluation windows gate models requiring extreme threshold approvals (`Entry Context > 80%`, `Reconciliation < 30%`, `Rolling Win Rate / PnL constraints`) ensuring unproven updates never immediately control live allocations.

---

## Live Execution Flow (Per Cycle)

Each continuous cycle iterates perfectly down the execution ladder:
1. **Reconciliation Sync**: `reconciliation_service` pulls true limits; rebuilds the local memory state and checks for dead pairs.
2. **Governor & Drift Evaluation**: The DriftMonitor parses Brier/Schema/Edge variables assigning `low -> critical` severity streaks. The Governors ingest these states assigning family-specific execution bounds.
3. **Context Refresh**: Update Spot/Macro/On-Chain/Order Flow indicators.
4. **Candidate Pricing**: Target available assets, filter through required confidence models / weather-forecast approvals.
5. **Execution Loop**: Intersect candidate signals against Governor limits, `MoneyManager` volumetric caps, and live Orderbook spread liquidity limits. Place or execute entries natively returning execution states.
6. **Graceful Sleep**: Sleep using small iterative waits mapped against active poll bounds. Catch SIGINT commands via immediate safe exit overrides.

## Quick Start Configuration

### Running the Live Platform
The system expects `TRADING_MODE=live` via your `.env` configured exactly with API and Wallet references.
```bash
python run_bot.py
```

### Sandbox Execution (Paper Trading)
Evaluates execution decisions actively tracking pending/shadow states internally.
```bash
python run_paper.py
```

### Auditing & State Recovery
To rebuild state or unfreeze stuck instances if exchange APIs diverge significantly:
```bash
python audit_runtime_state.py --logs-dir logs --reset-if-corrupt
```

## Logs and Directory Structures 
Monitor all live analytics inside the localized `logs/` directory.
- `logs/btc_forecast_eval.csv` — Roll forward predictions against observed outcomes.
- `logs/drift_monitor_state.json` — Continuous severity streaks actively bounding performance.
- `logs/closed_positions.csv` — Trade resolutions mapped structurally against execution costs and true slippage vectors.
- `logs/performance_governor.csv` — Tracks rolling win-rates and active `Level_X` designations.

## Notes for Contributors
When updating the agent environment:
1. **Never circumvent Governor limitations.** Do not decouple PnL checks or schema validations from execution sizing logic. 
2. **Prioritize Drift.** The system is structurally protective by design — missing inputs must aggressively decay size output. 
3. **Graceful Terminations.** Polling sub-threads must cleanly die alongside `_shutdown_requested` handlers to avoid process hangs.
