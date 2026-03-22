# Neural Network for Crypto

Public-data Polymarket research system for **BTC-related markets**, **smart-wallet tracking**, **paper-trading simulation**, **historical model training**, and an isolated **`live-test`** branch for future authenticated execution experiments.

## What this repo is

This project is for:
- public Polymarket market discovery
- public wallet / leaderboard analysis
- token-level historical dataset building
- supervised and replay-based research
- paper-trading simulation
- monitoring / dashboarding

On the paper / research line, this repo is **not** for:
- connecting your live Polymarket account on `main`
- storing or requiring real trading credentials for paper mode
- placing real-money orders from the paper path

## Branch policy

### `main`
Intended to stay:
- **public-data only**
- **paper-trading only**
- **research / backtesting only**

You do **not** need a Polymarket API key for the default paper setup.

### `live-test`
Experimental branch for isolated authenticated execution scaffolding and operational hardening.

Recent `live-test` work includes:
- `execution_client.py`
- `order_manager.py`
- `reconciliation_service.py`
- `live_risk_manager.py`
- `db.py`
- `incident_manager.py`
- backend alert normalization
- system health snapshots
- service heartbeat tracking
- incident lifecycle logging
- stronger monitoring dashboard sections

## External data sources

The research / paper path uses public endpoints only:
- **Gamma API** for market discovery
- **Data API** for leaderboard + public wallet trades
- **CLOB read endpoints** for price history and pricing research
- optional public **CLOB WebSocket** for live market updates

## High-level architecture

### Runtime entrypoints
- `run_bot.py` — default launcher
- `run_paper.py` — explicit paper entrypoint
- `run_live_test.py` — explicit live-test entrypoint
- `web_api.py` — local API for dashboard / inspection
- `dashboard.py` — Streamlit monitoring UI

### Core runtime flow
1. fetch BTC-related markets
2. scrape public wallet activity
3. build features from only past-known information
4. run supervised / hybrid inference
5. rank paper opportunities
6. simulate paper positions and exits
7. write logs, health files, alerts, and monitoring outputs
8. optionally retrain when enough real paper outcomes exist

## Important design rules

### Paper path
- public-data only
- paper-trading only
- no live auth required
- no real execution on `main`

### Trade semantics
Use explicit Polymarket semantics:
- `order_side` = `BUY` / `SELL`
- `outcome_side` = `YES` / `NO`
- `token_id`
- `condition_id`
- `entry_intent`

Do **not** infer YES/NO from BUY/SELL.

### PnL accounting
Polymarket-style share accounting is the rule:

```text
shares = capital_usdc / entry_price
pnl = shares * (exit_price - entry_price) - fees
```

### Validation philosophy
- time split / walk-forward validation
- no random leakage-heavy split
- token-level CLOB history is the source of truth for labels / replay
- features must be strictly past-only at signal time

## Main modules

### Market / wallet collection
- `market_monitor.py` — Gamma market discovery / monitoring
- `leaderboard_scraper.py` — public wallet and public trade discovery
- `clob_history.py` — token-level CLOB price history collection
- `market_price_service.py` — live-ish pricing and quote helpers

### Dataset / labeling / features
- `historical_dataset_builder.py`
- `feature_builder.py`
- `target_builder.py`
- `contract_target_builder.py`
- `wallet_alpha_builder.py`
- `sequence_feature_builder.py`
- `schema.py` — shared schema contract rollout

### Models / inference / evaluation
- `supervised_models.py`
- `model_inference.py`
- `stage1_models.py`
- `stage1_inference.py`
- `stage2_temporal_models.py`
- `stage2_temporal_inference.py`
- `stage3_hybrid.py`
- `evaluator.py`
- `time_split_trainer.py`
- `backtester.py`
- `retrainer.py`

### Paper trading / lifecycle
- `signal_engine.py`
- `strategy_layers.py`
- `position_manager.py`
- `pnl_engine.py`
- `trade_lifecycle.py`
- `polytrade_env.py`
- `path_replay_simulator.py`
- `execution_client.py` (paper abstraction on the paper path)

### Live-test-only / operational pieces
- `order_manager.py`
- `reconciliation_service.py`
- `live_risk_manager.py`
- `db.py`
- `api_setup.py`

### Monitoring / ops
- `alerts_engine.py`
- `autonomous_monitor.py`
- `incident_manager.py`
- `dashboard.py`
- `web_api.py`

## Monitoring and operational outputs

Generated in `logs/`:
- `signals.csv`
- `execution_log.csv`
- `daily_summary.txt`
- `markets.csv`
- `whales.csv`
- `market_distribution.csv`
- `alerts.csv`
- `positions.csv`
- `closed_positions.csv`
- `historical_dataset.csv`
- `contract_targets.csv`
- `supervised_eval.csv`
- `time_split_eval.csv`
- `path_replay_backtest.csv`
- `model_status.csv`
- `system_health.csv`
- `service_heartbeats.csv`
- `incidents.csv`

Other important outputs:
- `weights/model_registry.csv`
- `weights/ppo_polytrader.zip` (optional / non-required on startup)
- `logs/trading.db` on `live-test`

## Dashboard layout (`live-test`)

The Streamlit dashboard now follows a monitoring-oriented layout:
- **System Status**
- **Signals & Opportunities**
- **Positions & PnL**
- **Markets, Whales & Alerts**
- **Models & Data Quality**

Key UI additions on `live-test`:
- real data freshness panel
- pipeline health strip
- centralized “Attention Needed” warnings
- global sidebar controls and filters
- richer opportunity cards
- ranked opportunity table with CSV export
- grouped recommended paper actions
- paper equity / drawdown charts
- richer open / closed position ledgers
- market / whale / alert monitoring subtabs
- model performance and data-quality readiness subtabs
- schema health and anomaly detection panels
- raw logs moved into collapsible debug sections

## Monitoring improvements on `live-test`

Recent operational improvements include:
- normalized alert fields: `alert_id`, `severity`, `status`, `source_module`, `message`
- service heartbeat tracking for core runtime modules
- system health snapshots
- incident lifecycle logging with dedupe keys
- cross-source reconciliation checks
- monitoring-grade anomaly detection
- prediction-health / confidence-drift monitoring

## Quick start

## 1) Install

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional test dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

## 2) Local setup

```bash
python api_setup.py
```

## 3) Run paper mode

```bash
python run_bot.py
```

Or explicitly:

```bash
python run_paper.py
```

In another terminal:

```bash
python -m streamlit run dashboard.py
```

Optional local API:

```bash
python -m uvicorn web_api:app --reload
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Windows usage

### Paper / research (`main`)

```powershell
git pull origin main
python -m pip install -r requirements.txt
python run_bot.py
python -m streamlit run dashboard.py
```

### Experimental live-test branch

```powershell
git fetch origin
git checkout live-test
git pull origin live-test
python -m pip install -r requirements.txt
python run_bot.py
python -m streamlit run dashboard.py
```

If the browser looks stale after updates, do a hard refresh:

```text
Ctrl + F5
```

## Testing

Run the test suite with:

```bash
pytest -q
```

The repo now includes basic CI and smoke coverage for more than only low-level math.

## Troubleshooting

### Missing dependency errors

If startup fails with something like `ModuleNotFoundError`, refresh dependencies:

```powershell
python -m pip install -r requirements.txt
```

If needed:

```powershell
python -m pip install -r requirements-dev.txt
```

### Dashboard looks blank or stale

Common causes:
- old Streamlit process still running
- stale browser cache
- missing research artifacts
- schema mismatch in logs
- empty / missing monitoring files

Recommended fix path:

```powershell
python run_bot.py
python -m streamlit run dashboard.py
```

Then hard refresh the browser with `Ctrl + F5`.

## Current reality

This repo is still a **research system under active refactor**.

Current direction:
- supervised / event-driven ranking is primary
- replay-based evaluation is increasingly central
- paper monitoring is becoming much stronger
- live execution work remains isolated to `live-test`

RL is optional fallback only and should not be treated as the main intelligence path.

## Near-term direction

Main ongoing directions include:
- opportunity-scoring refactor
- stronger candidate generation vs ranking separation
- deeper wallet behavior modeling
- more robust microstructure support
- fuller live readiness and reconciliation on `live-test`
- broader schema rollout across the repo
- more operational monitoring and drift checks

## Safety reminder

On `main`, keep this repo:
- public-data only
- paper-trading only
- research / backtesting only

Live execution experiments belong on `live-test`, not on `main`.
