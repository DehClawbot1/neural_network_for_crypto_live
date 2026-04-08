# Neural Network For Crypto (Live Trading)

This repository runs a BTC-focused Polymarket trading system with:
- live execution and paper/shadow trading flows,
- strict exchange reconciliation and runtime-state repair,
- model-based signal scoring with technical, wallet-copy, on-chain, and BTC market-context features,
- guarded retraining and promotion workflows,
- benchmark and ablation tooling for research.

The project is no longer just a simple live loop. It now has three tightly connected layers:
1. `runtime trading`: live/paper decision-making, execution, exits, and reconciliation
2. `data + learning`: lifecycle truth, feedback learning, retraining gates, and promotion control
3. `research`: dataset building, benchmark comparison, feature ablation, and offline model evaluation

## Current Repository Status

The repository has moved well beyond the original baseline described in earlier docs.

### Implemented and active
- Performance governor with degradation levels and rolling live metrics
- Stronger trade lifecycle truth and learning-eligibility classification
- Runtime reconciliation hardening for dead orderbooks and stale local state
- Low-balance pause logic instead of misleading minimum-size failures
- Live BTC spot/index/mark tracking for decision support
- Feature-family catalog and ablation harness for offline evaluation
- Benchmark strategy and benchmark comparison outputs
- Structured retrainer verdicts with promotion-block context

### Implemented but still maturing in data
- BTC live/index fields are wired into training and ablation code paths
- The live BTC snapshots are being recorded correctly
- The current labeled training corpus still has very limited mature post-patch rows, so the newest BTC live/index feature family is present in code but not yet materially active in the fitted training sample

### Main remaining focus
- continue accumulating clean post-patch labeled rows
- keep reducing lifecycle ambiguity and legacy repair dependency
- validate that the BTC live/index family improves benchmarked model quality once enough mature rows exist

## Architecture Overview

### 1. Runtime Trading Layer
Core live runtime files:
- [supervisor.py](supervisor.py): main cycle orchestration, candidate evaluation, entry/exit flow, telemetry
- [trade_manager.py](trade_manager.py): trade lifecycle, closes, persistence, audit refresh
- [order_manager.py](order_manager.py): order placement, fill handling, buying-power checks
- [money_manager.py](money_manager.py): dynamic sizing, exchange-floor handling, low-balance pauses
- [live_position_book.py](live_position_book.py): reconstructed live position ledger from fills and exchange balances
- [reconciliation_service.py](reconciliation_service.py): exchange sync and runtime drift detection
- [performance_governor.py](performance_governor.py): rolling live performance controls
- [market_monitor.py](market_monitor.py): BTC market discovery including rotating btc-updown markets (5m/15m/4h) via Gamma Events API

### 2. Market Intelligence Layer
Context and feature generation files:
- [technical_analyzer.py](technical_analyzer.py): BTC technical regime context
- [btc_live_price_tracker.py](btc_live_price_tracker.py): live BTC spot/index/mark tracking, live returns, basis, source-quality diagnostics
- [macro_analyzer.py](macro_analyzer.py): macro/liquidity context
- [onchain_analyzer.py](onchain_analyzer.py): on-chain Bitcoin network context
- [order_flow_analyzer.py](order_flow_analyzer.py): order flow and taker-imbalance context
- [orderbook_depth_features.py](orderbook_depth_features.py): BTC L2 order book depth analysis (43 microstructure features)
- [btc_forecast_model.py](btc_forecast_model.py): ML ensemble for BTC price direction prediction
- [btc_multitimeframe.py](btc_multitimeframe.py): multi-timeframe (15m/1h/4h) weighted forecast combiner
- [btc_price_dataset.py](btc_price_dataset.py): 128+ feature engineering pipeline from OHLCV candles
- [btc_onchain_features.py](btc_onchain_features.py): derivatives data enrichment (funding, OI, L/S ratio)
- [btc_sentiment_features.py](btc_sentiment_features.py): Fear & Greed, Google Trends, Reddit NLP sentiment
- [btc_forecast_eval.py](btc_forecast_eval.py): walk-forward live prediction evaluation
- [feature_builder.py](feature_builder.py): grouped candidate feature construction
- [strategy_layers.py](strategy_layers.py): entry rule layer and veto logic

### 3. Learning and Research Layer
Training and evaluation files:
- [trade_quality.py](trade_quality.py): lifecycle truth, learning eligibility, signal-label normalization
- [trade_lifecycle_audit.py](trade_lifecycle_audit.py): lifecycle quality reports
- [trade_feedback_learner.py](trade_feedback_learner.py): feedback summaries and learning signals
- [retrainer.py](retrainer.py): retraining verdicts and promotion-block explanations
- [historical_dataset_builder.py](historical_dataset_builder.py): historical research dataset construction
- [contract_target_builder.py](contract_target_builder.py): training target generation
- [model_feature_catalog.py](model_feature_catalog.py): shared feature-family definitions
- [feature_ablation.py](feature_ablation.py): feature-family ablation harness
- [benchmark_strategy.py](benchmark_strategy.py): simpler benchmark strategy
- [real_pipeline.py](real_pipeline.py): end-to-end research pipeline runner

## Live Execution Flow

Per cycle, the bot now syncs and evaluates in this order:
1. Sync orders and fills from Polymarket into local storage
2. Rebuild `live_positions` from trusted fills and synthetic sync events
3. Reconcile runtime in-memory trades against exchange-backed local state
4. Run mismatch and drift checks
5. Freeze new entries for the cycle if the runtime is out of sync
6. Refresh technical, macro, on-chain, order-flow, and BTC live/index context
7. Score candidates, apply rule layers and governor constraints, then place entries only if the state is healthy

This design is meant to prevent new trades from opening while the bot is uncertain about exchange truth.

## Risk Controls and Safeguards

### Performance governor
The bot reads rolling live metrics and automatically degrades behavior when recent performance weakens.

Governor levels:
- `level 0`: normal sizing and normal gates
- `level 1`: reduced size and stricter candidate gates
- `level 2`: minimum-size posture and top-signal-only behavior

Tracked metrics include:
- rolling win rate
- average pnl
- profit factor
- realized drawdown
- RL exit share
- operational close share

Output:
- [performance_governor.csv](logs/performance_governor.csv)

### Balance and sizing safety
Live sizing now uses spendable CLOB balance and respects:
- capital reserve
- per-trade risk caps
- hard max bet caps
- exchange minimum notional rules

When the wallet cannot safely support the exchange floor, the bot records a `low_balance_pause` instead of pretending the skip was a normal sizing rejection.

### Reconciliation and dead-orderbook handling
The runtime includes:
- dead-orderbook safeguards
- live-position rebuild filtering
- stale local close cleanup
- forced reconciliation close paths
- SQLite integrity checks and optional reset tooling

Operational audit entry points:
- [audit_runtime_state.py](audit_runtime_state.py)
- [cleanup_dead_tokens.py](cleanup_dead_tokens.py)

## BTC Price Prediction Pipeline

The bot includes a full BTC price prediction system with multi-timeframe ML models and alternative data enrichment.

### Architecture

```
Pillar 5: Multi-Timeframe Forecast (15m/1h/4h weighted ensemble)
Pillar 6: Sentiment Features (Fear & Greed Index, Google Trends, Twitter/X + Reddit NLP)
Pillar 7: Order Book Depth (Binance L2 imbalance, slope, whale walls)
Pillar 8: Walk-Forward Live Evaluation (prediction vs actual tracking)
```

### Core modules

| Module | Purpose |
|--------|---------|
| [btc_price_dataset.py](btc_price_dataset.py) | 128+ feature engineering from OHLCV (RSI, MACD, ADX, ATR, Bollinger, Stochastic RSI, CCI, MFI, Williams %R, Donchian, VWAP, Garman-Klass volatility, lag features, momentum stats, cyclical time encoding) |
| [btc_forecast_model.py](btc_forecast_model.py) | Ensemble of 4 models (2x LightGBM + HistGradientBoosting + MLP) with purged walk-forward CV, feature importance pruning, exponential recency weighting |
| [btc_multitimeframe.py](btc_multitimeframe.py) | Manages 15m/1h/4h models with weighted voting (0.25/0.35/0.40), confidence gating, agreement threshold |
| [btc_onchain_features.py](btc_onchain_features.py) | Derivatives data: funding rate, open interest, long/short ratio, taker buy/sell volume from Binance Futures |
| [btc_sentiment_features.py](btc_sentiment_features.py) | Fear & Greed Index (contrarian signal), Google Trends (retail proxy), Twitter/X + Reddit VADER NLP sentiment |
| [orderbook_depth_features.py](orderbook_depth_features.py) | 43 L2 microstructure features: depth imbalance at 5/10/20 levels, cumulative depth at 10/25/50/100 bps, book slope, whale wall detection, volume-weighted midpoint |
| [btc_forecast_eval.py](btc_forecast_eval.py) | Walk-forward live evaluator: logs every prediction vs actual outcome to `logs/btc_forecast_eval.csv`, computes rolling accuracy |
| [download_btc_dataset.py](download_btc_dataset.py) | Downloads historical OHLCV from Binance, supports `--enrich` (derivatives), `--sentiment`, `--multi-timeframe` |

### Model accuracy (latest training)

| Timeframe | Direction Accuracy | Confident Dir Accuracy | Classifier Accuracy | Weight |
|-----------|-------------------|----------------------|--------------------|---------|
| 15m | 51.05% | 51.16% | 49.96% | 0.25 |
| 1h | 47.06% | 45.44% | 50.39% | 0.35 |
| **4h** | **54.93%** | **55.66%** | **55.41%** | **0.40** |

The 4h model is the most accurate (less noise at higher timeframes) and gets the highest weight in the ensemble.

### Quick commands

```bash
# Download all timeframes + train with all enrichments
python download_btc_dataset.py --multi-timeframe --enrich --sentiment --train

# Single timeframe training
python download_btc_dataset.py --interval 15m --days 730 --enrich --sentiment --train

# Collect order book depth training data (1 hour, every 60s)
python -c "from orderbook_depth_features import OrderBookDepthAnalyzer; a = OrderBookDepthAnalyzer(); df = a.collect_depth_timeseries(60, 3600); df.to_csv('data/btc_depth_features.csv', index=False)"
```

### Evaluation output

The walk-forward evaluator logs every prediction to `logs/btc_forecast_eval.csv` with columns:
- `predict_ts`, `eval_ts`, `entry_price`, `exit_price`
- `predicted_direction`, `predicted_return`, `confidence`
- `actual_return`, `actual_direction`, `correct`, `pnl_pct`
- `mtf_agreement`, `mtf_source`

Pending predictions waiting to mature are persisted to `logs/btc_forecast_eval_pending.csv`.

Rolling accuracy is computed in-memory and logged each cycle.

## BTC Live Price and Index Tracking

The bot now has a dedicated BTC live market context module:
- [btc_live_price_tracker.py](btc_live_price_tracker.py)

It tracks:
- spot references from multiple venues
- Binance futures index price
- Binance futures mark price
- funding rate
- 1m / 5m / 15m / 1h live returns
- spot/index and mark/index basis
- source divergence and source-quality score
- a derived live directional bias and confluence score

Runtime outputs:
- [btc_live_snapshot.csv](logs/btc_live_snapshot.csv)
- [candidate_decisions.csv](logs/candidate_decisions.csv) with enriched `details_json`

These fields are already used in live decision support. They are also wired into the research pipeline, but their effect in trained models still depends on more post-patch labeled rows accumulating.

## Learning Integrity and Lifecycle Truth

The repository now separates clean learning events from operational noise.

Key concepts:
- `learning_eligible`
- `entry_context_complete`
- `exit_reason_family`
- `operational_close_flag`
- reconciliation close tracking separated from true strategy closes

Important outputs:
- [trade_lifecycle_audit.csv](logs/trade_lifecycle_audit.csv)
- [trade_feedback_summary.csv](logs/trade_feedback_summary.csv)
- [closed_positions.csv](logs/closed_positions.csv)

Recent work in this area included:
- immediate audit refresh after reconciliation closes
- better signal-label normalization for future rows
- explicit retrainer messaging when recent windows are reconciliation-heavy

## Training, Benchmarking, and Ablation

The repository now has a shared feature-family view for model experiments.

Feature families currently defined in [model_feature_catalog.py](model_feature_catalog.py):
- wallet-copy
- market microstructure
- on-chain network
- BTC spot regime
- BTC live/index

Main research artifacts:
- [historical_dataset.csv](logs/historical_dataset.csv)
- [contract_targets.csv](logs/contract_targets.csv)
- [feature_ablation_report.csv](logs/feature_ablation_report.csv)
- [trade_quality_ablation_report.csv](logs/trade_quality_ablation_report.csv)
- [benchmark_vs_main.csv](logs/benchmark_vs_main.csv)

Important current caveat:
- the BTC live/index family is wired into the dataset and ablation code
- but if the labeled sample predates the new tracker, those fields will still be all-null in `contract_targets.csv`
- in that case the ablation report will show the family as present in code but unused in the fitted sample

## Roadmap and Completion Status

This repository is following the roadmap in this README. Below is the practical project status as of the current codebase.

### Phase 1: Stop the bleeding without shutting the bot off
Status: mostly done

Completed:
- performance governor implemented
- rolling live metric monitoring implemented
- governor level persisted to logs and used in candidate gating
- degraded sizing and stricter entry behavior implemented
- low-balance pause handling implemented

Still worth improving:
- continue tuning thresholds as newer clean live data arrives

### Phase 2: Audit and repair trade lifecycle truth
Status: mostly done

Completed:
- lifecycle audit report added
- learning-eligible vs operational-only separation added
- signal-label normalization improved
- reconciliation closes isolated from true strategy closes
- audit refresh after reconciliation closes implemented

Still worth improving:
- further reduce reliance on historical legacy repair labels as new rows accumulate naturally

### Phase 3: Fix exits so losses are deliberate, not accidental
Status: partially done

Completed:
- dead-orderbook and stale-state cleanup improved
- exit attribution quality improved
- local/runtime reconciliation drift is handled more safely

Still worth improving:
- continue reducing operational/reconciliation exits as a share of total closes
- continue validating exit-family mix as fresh live data arrives

### Phase 4: Make learning trustworthy before making it smarter
Status: mostly done

Completed:
- learning eligibility gates are in place
- entry-context completeness improved sharply
- retrainer verdicts explain why promotion is blocked
- reconciliation-heavy windows now report transparent promotion context

Still worth improving:
- keep increasing the share of naturally clean rows so fewer historical repairs are needed

### Phase 5: Build a benchmark strategy and ablation harness
Status: done for the first usable version

Completed:
- benchmark strategy exists
- benchmark logging exists
- feature-family ablation harness exists
- shared model feature catalog exists
- research pipeline writes ablation output

Still worth improving:
- increase sample quality so the newest feature families affect the benchmark meaningfully

### Phase 6: Add smarter prediction only after the base is healthy
Status: **substantially complete**

Completed:
- BTC live/index decision support added to runtime
- BTC live/index family added to training feature definitions
- Dataset and target builders can merge BTC live/index context
- **BTC price prediction pipeline** (5 modules, 128+ features, ensemble ML)
- **Multi-timeframe forecasting** (15m/1h/4h with weighted voting)
- **Derivatives enrichment** (funding rate, open interest, long/short ratio, taker volume)
- **Sentiment features** (Fear & Greed Index, Google Trends, Twitter/X + Reddit VADER NLP)
- **Order book depth features** (43 L2 microstructure features from Binance Futures)
- **Walk-forward live evaluator** (logs every prediction vs actual outcome)
- 52 tests across 4 test suites

Still worth improving:
- Accumulate enough walk-forward evaluation data to measure true live accuracy
- Add more training data for order book depth features (currently real-time only)
- Consider adding on-chain whale transaction tracking
- Tune ensemble weights based on live evaluation results
- Add automated retraining triggers when live accuracy drops below threshold

## Quick Start

### 1. Live trading
Use this when you want the full live runtime with exchange sync, candidate evaluation, and live order management.

```bash
python run_bot.py
```

Use this only after confirming:
- live wallet credentials are loaded
- the latest runtime audit is healthy
- you are comfortable with the current governor level and balance state

### 2. Paper mode
Use this when you want to exercise the decision engine without sending real live orders.

```bash
python run_paper.py
```

This is the safest default when:
- validating new strategy logic
- checking candidate telemetry
- testing new rule-layer behavior

### 3. Research pipeline
Use this to rebuild datasets, targets, benchmark outputs, and feature ablation reports.

```bash
python real_pipeline.py
```

Typical outputs refreshed by this step include:
- `logs/historical_dataset.csv`
- `logs/contract_targets.csv`
- `logs/feature_ablation_report.csv`
- `logs/benchmark_vs_main.csv`

### 4. Debug runtime drift
Use this when local runtime state may be out of sync with exchange truth, or when you suspect ghost positions / stale ledgers.

Audit only:
```bash
python audit_runtime_state.py --logs-dir logs
```

Audit and reset only if corruption is detected:
```bash
python audit_runtime_state.py --logs-dir logs --reset-if-corrupt
```

Good times to run this:
- after dead-orderbook incidents
- after repeated mismatch freezes
- after partial-fill or reconciliation anomalies

## Environment and Required Config

### Python and dependencies
On a fresh machine, install the project dependencies first:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

If you also need development tooling:

```bash
pip install -r requirements-dev.txt
```

### .env Variable Reference

Create a `.env` file in the project root. The bot loads it automatically via `python-dotenv`.

#### Required for Live Trading

| Variable | Example | Description |
|----------|---------|-------------|
| `TRADING_MODE` | `live` | Set to `live` for real trading, `paper` for simulated |
| `PRIVATE_KEY` | `b81c379f...` | Your wallet private key (hex, no 0x prefix). Used to sign orders on Polymarket CLOB |
| `POLYMARKET_FUNDER` | `0x4d0CD2Fa...` | Your Polymarket proxy wallet address. This is the address that holds your USDC on Polygon |
| `POLYMARKET_SIGNATURE_TYPE` | `2` | How your wallet connects to Polymarket: `0` = direct EOA, `1` = Magic/email login, `2` = MetaMask/Rabby browser wallet |

#### Polymarket L2 API Credentials

These are auto-derived on first run if you provide `PRIVATE_KEY`. You can also set them manually.

| Variable | Example | Description |
|----------|---------|-------------|
| `POLYMARKET_API_KEY` | `ed0e82d6-...` | L2 CLOB API key for order placement and balance queries |
| `POLYMARKET_API_SECRET` | `O9otKHB5C...` | L2 CLOB API secret (base64) |
| `POLYMARKET_API_PASSPHRASE` | `49d58587f...` | L2 CLOB API passphrase (hex) |
| `POLYMARKET_API_CREDS_SIGNATURE_TYPE` | `2` | Signature type used when the credentials were derived (should match `POLYMARKET_SIGNATURE_TYPE`) |
| `POLYMARKET_API_CREDS_FUNDER` | `0x4d0CD2Fa...` | Funder address used when credentials were derived (should match `POLYMARKET_FUNDER`) |

#### Sizing and Risk

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMULATED_STARTING_BALANCE` | `1000` | Starting balance for paper mode simulations (USDC) |
| `MAX_RISK_PER_TRADE` | `50` | Maximum USDC risked per individual trade |

#### Research and Refresh

| Variable | Default | Description |
|----------|---------|-------------|
| `RESEARCH_REFRESH_MAX_AGE_MINUTES` | `30` | Maximum age (minutes) before research context is considered stale and refreshed |
| `MARKET_SNAPSHOT_TTL_HOURS` | `48` | Hours before old market snapshot rows are purged from `logs/markets.csv` |

#### Open Pain (Drawdown Sensitivity)

These control the "open pain" system that penalizes new entries when existing positions are in drawdown.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPEN_PAIN_SENSITIVITY` | `1.0` | Global multiplier for all open-pain effects (0 = disabled, >1 = more aggressive) |
| `OPEN_PAIN_TRIGGER_OPEN_RETURN` | `-0.015` | Open position return threshold that activates pain penalties |
| `OPEN_PAIN_TRIGGER_MAE` | `-0.03` | Maximum adverse excursion threshold |
| `OPEN_PAIN_TRIGGER_DRAWDOWN` | `0.03` | Portfolio drawdown threshold |
| `OPEN_PAIN_TRIGGER_FAST_COUNT` | `1` | Number of fast-losing positions to trigger pain |
| `OPEN_PAIN_CONF_PENALTY_MAX` | `0.18` | Max confidence penalty applied under pain |
| `OPEN_PAIN_RET_PENALTY_MAX` | `0.24` | Max expected-return penalty applied under pain |
| `OPEN_PAIN_CONF_MULTIPLIER_FLOOR` | `0.82` | Minimum confidence multiplier (prevents over-penalizing) |
| `OPEN_PAIN_RET_MULTIPLIER_FLOOR` | `0.76` | Minimum return multiplier floor |
| `OPEN_PAIN_EDGE_MULTIPLIER_FLOOR` | `0.78` | Minimum edge multiplier floor |

#### Performance Governor Level 1 (Degraded Mode)

Activated when recent live performance drops below thresholds. Reduces position sizing and raises entry bars.

| Variable | Default | Description |
|----------|---------|-------------|
| `GOV_LEVEL1_MIN_WIN_RATE` | `0.38` | Trigger if rolling win rate drops below this |
| `GOV_LEVEL1_MIN_PROFIT_FACTOR` | `0.80` | Trigger if profit factor drops below this |
| `GOV_LEVEL1_MAX_NEGATIVE_AVG_PNL` | `-0.10` | Trigger if average PnL is more negative than this |
| `GOV_LEVEL1_MAX_DRAWDOWN` | `30` | Trigger if realized drawdown exceeds this (%) |
| `GOV_LEVEL1_MAX_RL_EXIT_RATE` | `0.45` | Trigger if RL-driven exit share exceeds this |
| `GOV_LEVEL1_MAX_OPERATIONAL_CLOSE_RATE` | `0.15` | Trigger if operational close share exceeds this |
| `GOV_LEVEL1_SIZE_MULTIPLIER` | `0.35` | Position size multiplier when Level 1 is active |
| `GOV_LEVEL1_MIN_ENTRY_CONFIDENCE` | `0.68` | Minimum model confidence required to enter a trade |
| `GOV_LEVEL1_MIN_LIQUIDITY_SCORE` | `0.50` | Minimum market liquidity score required |

#### Performance Governor Level 2 (Maximum Protection)

Most restrictive mode. Only the top signal is considered, minimum sizes enforced.

| Variable | Default | Description |
|----------|---------|-------------|
| `GOV_LEVEL2_MIN_WIN_RATE` | `0.30` | Trigger if rolling win rate drops below this |
| `GOV_LEVEL2_MIN_PROFIT_FACTOR` | `0.60` | Trigger if profit factor drops below this |
| `GOV_LEVEL2_MAX_NEGATIVE_AVG_PNL` | `-0.25` | Trigger if average PnL is more negative than this |
| `GOV_LEVEL2_MAX_DRAWDOWN` | `60` | Trigger if realized drawdown exceeds this (%) |
| `GOV_LEVEL2_MAX_RL_EXIT_RATE` | `0.60` | Trigger if RL-driven exit share exceeds this |
| `GOV_LEVEL2_MAX_OPERATIONAL_CLOSE_RATE` | `0.10` | Trigger if operational close share exceeds this |
| `GOV_LEVEL2_SIZE_MULTIPLIER` | `0.20` | Position size multiplier when Level 2 is active |
| `GOV_LEVEL2_MIN_ENTRY_CONFIDENCE` | `0.35` | Minimum model confidence required to enter a trade |
| `GOV_LEVEL2_MIN_LIQUIDITY_SCORE` | `0.70` | Minimum market liquidity score required |

#### Model Promotion Gates

Control when the retrainer is allowed to promote a new model to production.

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMOTION_MIN_LEARNING_ELIGIBLE_RATIO` | `0.65` | Minimum share of recent trades that must be learning-eligible |
| `PROMOTION_MIN_ENTRY_CONTEXT_COMPLETE_RATIO` | `0.70` | Minimum share of trades with complete entry context |
| `PROMOTION_MAX_OPERATIONAL_CLOSE_RATIO` | `0.30` | Maximum share of operational/reconciliation closes allowed |
| `PROMOTION_MAX_UNKNOWN_SIGNAL_LABEL_RATIO` | `0.20` | Maximum share of trades with unknown signal labels |

#### Other Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RESET_RUNTIME_STATE_ON_DB_CORRUPTION` | `false` | If `true`, runtime audit can archive and rebuild state when corruption is detected |
| `BOT_LOG_LOCAL_TIMEZONE` | `Europe/Lisbon` | Timezone for normalizing naive local log timestamps in dataset builders |
| `ALWAYS_ON_MARKET_SIDE` | `AUTO` | Force always-on market side (`YES`/`NO`), or `AUTO` for ML-driven 3-tier selection: BTC forecast > leaderboard consensus > price fallback |

### Important built-in runtime defaults
Several key defaults currently live in [config.py](config.py) under `TradingConfig`, including:
- `MAX_RISK_PER_TRADE_PCT`
- `MIN_BET_USDC`
- `MIN_ENTRY_USDC`
- `HARD_MAX_BET_USDC`
- `CAPITAL_RESERVE_PCT`
- `MAX_CONCURRENT_POSITIONS`
- `TIME_STOP_MINUTES`

These are code-level defaults, so if you change them, treat them like strategy changes rather than harmless environment tweaks.

## Useful Commands

Run the live bot:
```bash
python run_bot.py
```

Run paper mode:
```bash
python run_paper.py
```

Run live bot plus dashboard:
```bash
python run_bot_and_dashboard.py
```

Run runtime audit:
```bash
python audit_runtime_state.py --logs-dir logs
```

Run audit and reset only if corruption is detected:
```bash
python audit_runtime_state.py --logs-dir logs --reset-if-corrupt
```

Run the research pipeline:
```bash
python real_pipeline.py
```

## Important Logs and Outputs

Execution and state:
- `logs/positions.csv`
- `logs/closed_positions.csv`
- `logs/live_orders.csv`
- `logs/live_fills.csv`
- `logs/candidate_decisions.csv`
- `logs/performance_governor.csv`
- `logs/system_health.csv`

Learning and quality:
- `logs/trade_lifecycle_audit.csv`
- `logs/trade_feedback_summary.csv`
- `logs/model_status.csv`
- `logs/backtest_summary.csv`

BTC prediction and evaluation:
- `logs/btc_forecast_eval.csv` — walk-forward prediction vs actual outcome log
- `logs/btc_forecast_train_log.csv` — training metrics history
- `logs/btc_price_dataset.csv` — labelled training dataset (128+ features)

Market context and research:
- `logs/markets.csv`
- `logs/btc_live_snapshot.csv`
- `logs/historical_dataset.csv`
- `logs/contract_targets.csv`
- `logs/feature_ablation_report.csv`
- `logs/trade_quality_ablation_report.csv`
- `logs/benchmark_vs_main.csv`

## Notes for Contributors

When changing the project, prefer these principles:
- keep exchange truth and local runtime truth aligned before adding aggressiveness
- treat learning quality as a first-class safety constraint, not just a reporting detail
- add new predictive features to research and ablation first, then promote them into live decision-making only after they prove useful
- avoid changing append-only historical CSV schemas casually; add new telemetry through safe dedicated outputs when needed

## Current Bottom Line

This repository is in a much healthier state than the original baseline:
- runtime reconciliation is stronger
- the bot explains more of its decisions and retraining blocks
- lifecycle truth and learning quality are materially improved
- BTC live/index context is now part of both runtime decision support and the research feature graph

The main next step is not more plumbing. It is allowing enough clean post-patch labeled data to accumulate so the newest feature families can be evaluated fairly and promoted only if they add real edge.
