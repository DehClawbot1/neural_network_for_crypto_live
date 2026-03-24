# Neural Network for Crypto

Public-data Polymarket research and paper-trading system focused on **BTC-related markets**, **smart-wallet tracking**, **historical modeling**, **paper execution**, and an isolated **`live-test`** branch for experimental authenticated execution and live-learning scaffolding.

---

## What this repo is for

This project is for:
- public Polymarket market discovery
- public wallet / leaderboard analysis
- historical feature + label generation
- supervised / temporal / hybrid modeling
- replay-based and paper-trading evaluation
- monitoring, incident tracking, and operational dashboards

This project is **not** for on `main`:
- live real-money execution
- storing or requiring Polymarket credentials for paper mode
- authenticated account activity in the default paper path

---

## Branch policy

### `main`
Must stay:
- **public-data only**
- **paper-trading only**
- **research / backtesting only**

No Polymarket API key is needed for the default paper path.

### `live-test`
Experimental branch for:
- authenticated execution scaffolding
- order lifecycle handling
- readiness / risk gating
- reconciliation / DB logging
- live env + live-learning scaffolds
- stronger monitoring / ops tooling

---

## External data sources

Public / research side uses:
- **Gamma API** — market discovery
- **Data API** — leaderboard + public wallet trades
- **CLOB REST** — price / history research
- **CLOB WebSocket** — live quote monitoring

---

## Core architecture

### Entrypoints
- `run_bot.py` — main launcher
- `run_paper.py` — explicit paper launcher
- `run_live_test.py` — explicit live-test launcher
- `run_everything.py` — unified multiprocessing launcher for execution + nearline learning on `live-test`
- `main_shadow.py` — shadow intent capture + async resolver loop
- `dashboard.py` — Streamlit UI
- `pages/1_Account_Profile.py` — Streamlit account-profile / public-profile page
- `web_api.py` — local API / docs

### High-level runtime flow
1. fetch BTC-related markets
2. scrape public wallet activity
3. build past-only features
4. run supervised / temporal / hybrid scoring
5. rank opportunities
6. simulate paper positions and exits
7. write logs / health / alerts / incidents / dashboard outputs
8. optionally retrain from outcomes

---

## Key design rules

### Trade semantics
Use explicit Polymarket semantics:
- `order_side` = `BUY` / `SELL`
- `outcome_side` = `YES` / `NO`
- `token_id`
- `condition_id`
- `entry_intent`

Do **not** infer YES/NO from BUY/SELL.

### PnL accounting
Use share-based Polymarket-style accounting:

```text
shares = capital_usdc / entry_price
pnl = shares * (exit_price - entry_price) - fees
```

### Validation
- no random-split leakage for time-series work
- time split / walk-forward validation is preferred
- token-level CLOB history is the label source of truth
- missing RL weights must not abort startup

---

## Main modules

### Market / wallet data
- `market_monitor.py` — Gamma pagination + BTC market filtering
- `leaderboard_scraper.py` — public leaderboard / wallet-trade scraping
- `clob_history.py` — token-level historical prices
- `market_price_service.py` — quote / midpoint / spread / websocket monitoring

### Feature / label pipeline
- `historical_dataset_builder.py`
- `feature_builder.py`
- `target_builder.py`
- `contract_target_builder.py`
- `wallet_alpha_builder.py`
- `sequence_feature_builder.py`
- `schema.py`
- `autoencoder_features.py` — latent feature compression scaffold

### Models / evaluation
- `supervised_models.py`
- `model_inference.py`
- `stage1_models.py`
- `stage1_inference.py`
- `stage2_temporal_models.py` — temporal sklearn baseline with scaling, class balancing, walk-forward validation, regularization
- `stage2_sequence_models.py` — GRU-based PyTorch sequence scaffold (research-only; not wired into current runtime inference)
- `stage2_transformer_models.py` — transformer / attention scaffold (research-only; not wired into current runtime inference)
- `stage2_temporal_inference.py`
- `stage3_hybrid.py` — hybrid scorer + ensemble agreement gating
- `evaluator.py`
- `time_split_trainer.py`
- `model_tuning.py` — Optuna-based tuning scaffold
- `retrainer.py`
- `backtester.py`

### Paper trading / replay
- `signal_engine.py`
- `strategy_layers.py`
- `position_manager.py`
- `pnl_engine.py`
- `trade_lifecycle.py`
- `polytrade_env.py`
- `path_replay_simulator.py`

### Live-test execution / ops
- `execution_client.py`
- `order_manager.py`
- `reconciliation_service.py`
- `live_risk_manager.py`
- `db.py`
- `api_setup.py`
- `incident_manager.py`
- `live_replay_buffer.py`
- `shadow_purgatory.py`
- `shadow_logger.py`
- `main_shadow.py`
- `shadow_execution_audit.py`
- `shadow_slippage_calibration.py`
- `shadow_doa_resurrection.py`
- `shadow_limit_order_simulator.py`

---

## Live-test additions

The `live-test` branch now includes:
- stored Polymarket L2 API credential support in `execution_client.py`
- fallback credential derivation only when stored L2 creds are missing
- `LivePolyTradeEnv` scaffold using live quote / balance / position state
- live order-manager wiring from env actions
- live experience logging to `logs/live_experience.csv`
- nearline fine-tuning scaffold from live replay buffer
- unified execution + training launcher via multiprocessing

### Live learning scaffolds
- `LivePolyTradeEnv` in `polytrade_env.py`
- `live_replay_buffer.py`
- `rl_trainer.py::fine_tune_from_live_buffer(...)`
- live reward path scaffold based on balance deltas / fill polling
- expanded live observation space including:
  - bid / ask / midpoint / spread
  - live balance
  - unrealized PnL
  - order-book imbalance
  - time decay to resolution
  - correlated BTC feed

---

## Monitoring / ops

### Monitoring files in `logs/`
- `system_health.csv`
- `service_heartbeats.csv`
- `incidents.csv`
- `alerts.csv`
- `signals.csv`
- `execution_log.csv`
- `positions.csv`
- `closed_positions.csv`
- `path_replay_backtest.csv`
- `model_status.csv`
- `live_experience.csv`

### Monitoring improvements on `live-test`
- normalized alert fields:
  - `alert_id`
  - `severity`
  - `status`
  - `source_module`
  - `message`
- service heartbeat tracking
- incident lifecycle logging
- cross-source reconciliation checks
- anomaly detection
- prediction-health / drift monitoring
- data-quality / schema-health views
- shadow intent / resolution logging in `logs/shadow_results.csv`
- execution-tax auditing for shadow trades
- slippage calibration / veto-rate auditing
- DOA resurrection / ghost-win auditing
- limit-order simulation for vetoed trades

---

## Dashboard layout (`live-test`)

The dashboard is now organized into:
- **System Status**
- **Signals & Opportunities**
- **Positions & PnL**
- **Markets, Whales & Alerts**
- **Models & Data Quality**

### Important dashboard capabilities
- real data freshness panel
- pipeline health strip
- centralized **Attention Needed** section
- richer sidebar controls and global filters
- opportunity cards + ranking table + CSV export
- grouped recommended paper actions
- PnL / equity / drawdown charts
- richer open / closed position ledgers
- market / whale / alert subtabs
- model-performance and data-quality readiness subtabs
- debug raw logs in collapsible sections

---

## Modeling upgrades recently added

### Temporal sklearn improvements
- feature scaling for MLPs
- class balancing before classifier fit
- walk-forward `TimeSeriesSplit`
- regularization / early stopping / learning-rate tuning hooks

### Deep-learning scaffolds
- GRU sequence model scaffold (research-only; not part of the active runtime scoring path)
- transformer / attention scaffold (research-only; not part of the active runtime scoring path)
- profit-weighted loss scaffold
- autoencoder latent-feature builder

### Hybrid / ensemble logic
- random-forest + neural agreement columns
- ensemble probability / agreement gating
- live-candidate gating when both sides agree above threshold

---

## Quick start

### Install

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional dev / test packages:

```bash
python -m pip install -r requirements-dev.txt
```

### Environment setup

```bash
python api_setup.py
```

### Run paper mode

```bash
python run_bot.py
```

Or explicitly:

```bash
python run_paper.py
```

### Run dashboard

```bash
python -m streamlit run dashboard.py
```

### Run local API

```bash
python -m uvicorn web_api:app --reload
```

Docs:

```text
http://127.0.0.1:8000/docs
```

---

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

### Unified live-test execution + learning

```powershell
python run_everything.py
```

Hard refresh browser if UI seems stale:

```text
Ctrl + F5
```

---

## Main outputs

Generated in `logs/`:
- `signals.csv`
- `raw_candidates.csv`
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
- `wallet_alpha.csv`
- `wallet_alpha_history.csv`
- `sequence_dataset.csv`
- `shadow_results.csv`
- `supervised_eval.csv`
- `time_split_eval.csv`
- `stage2_temporal_eval.csv`
- `path_replay_backtest.csv`
- `model_status.csv`
- `system_health.csv`
- `service_heartbeats.csv`
- `incidents.csv`
- `live_experience.csv`
- `autoencoder_latent_features.csv`

Generated in `weights/`:
- `tp_classifier.joblib`
- `return_regressor.joblib`
- `stage2_temporal_classifier.joblib`
- `stage2_temporal_regressor.joblib`
- `stage2_sequence_classifier.pt` (research-only artifact; not loaded by current runtime inference)
- `stage2_sequence_regressor.pt` (research-only artifact; not loaded by current runtime inference)
- `stage2_transformer.pt` (research-only artifact; not loaded by current runtime inference)
- `feature_autoencoder.pt`
- `meta_model_bundle_*.pkl`
- `model_registry.csv`
- `ppo_polytrader.zip`

---

## Testing

Run the full local suite with:

```bash
pytest -q
```

Run the scoped CI-equivalent check with:

```bash
python -m pytest -q --cov=order_manager --cov=execution_client --cov=reconciliation_service --cov=position_manager --cov=contract_target_builder --cov-fail-under=60
```

Recent additions now cover:
- shadow purgatory / DOA logic
- CLOB retry resilience
- stateful feature-builder wallet stats
- database schema + event logging
- Polymarket auth / info retrieval
- Stage 2 temporal preprocessing
- Stage 2 transformer sequence reshaping
- DOA resurrection auditing

Basic CI and smoke checks are already in the repo.

---

## Troubleshooting

### Missing dependencies

If you hit `ModuleNotFoundError`:

```powershell
python -m pip install -r requirements.txt
```

Optional extras if using advanced model scaffolds:
- `torch`
- `optuna`

### Dashboard looks stale or blank

Typical causes:
- old Streamlit process
- stale browser cache
- missing artifact files
- schema mismatch in logs
- monitoring files empty / stale

Recommended recovery:

```powershell
python run_bot.py
python -m streamlit run dashboard.py
```

Then hard refresh with `Ctrl + F5`.

---

## Current reality

This repo is still a **research system under active refactor**.

Current direction:
- supervised / temporal / hybrid ranking is primary
- replay-based evaluation is core
- paper monitoring is much stronger than before
- `live-test` is where execution and live-learning experiments belong
- RL is optional / experimental, not the paper-path foundation

---

## Next direction

Near-term ongoing work includes:
- deeper opportunity-scoring refactor
- stronger candidate generation vs ranking separation
- richer wallet behavior modeling
- better microstructure support
- stronger live readiness and reconciliation
- broader schema rollout across the repo
- further drift / incident / ops hardening

---

## Safety reminder

On `main`, keep this repo:
- public-data only
- paper-trading only
- research / backtesting only

Live execution experiments belong on **`live-test`**, not on `main`.
