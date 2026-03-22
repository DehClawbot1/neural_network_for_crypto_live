# Neural Network for Crypto

Public-data Polymarket research system for **BTC-related markets**, **smart-wallet tracking**, **paper-trading simulation**, and **historical model training**.

## What this project is

This repo is for:
- public market discovery
- public wallet / leaderboard analysis
- historical dataset building
- supervised model research
- path-replay backtesting
- paper-trading dashboards

This repo is **not** for:
- connecting your live Polymarket account
- placing real orders
- real-money execution
- storing or requiring your private trading credentials

## Safety / mode

Current intended mode:
- **public-data only**
- **paper-trading only**
- **research / backtesting only**

You do **not** need a Polymarket API key for the current setup.

The project uses public endpoints only:
- **Gamma API** for market discovery
- **Data API** for leaderboard + public wallet trades
- **CLOB read endpoints** for price history
- optionally public **CLOB WebSocket** for live market updates

## Current architecture

## Runtime

### `run_bot.py`
Main launcher.

It now:
1. validates environment
2. checks existing weights
3. runs the research pipeline refresh
4. starts the continuous supervisor loop

### `supervisor.py`
Continuous monitoring loop for:
- fetching public BTC-related market/account activity
- scoring paper opportunities
- simulating paper positions
- updating logs for the dashboard

## Data collection

### `market_monitor.py`
Uses **Gamma** market discovery and tracks:
- `condition_id`
- `clob_token_ids`
- `yes_token_id`
- `no_token_id`
- liquidity / volume / last trade / end date

### `leaderboard_scraper.py`
Uses the public **Data API** to:
- fetch top crypto wallets
- scan recent public trades
- extract BTC-related signal candidates

### `clob_history.py`
Uses public **CLOB `/prices-history`** for token-level price history.

This is the correct data source for:
- forward-return labels
- TP-before-SL labels
- MFE / MAE
- replay simulation

## Dataset / features / labels

### `historical_dataset_builder.py`
Builds the project dataset around:
- one signal
- one timestamp
- one market
- only information available at that moment

It now merges:
- market microstructure fields
- rolling wallet metrics
- BTC context features
- wallet alpha summaries

### `wallet_alpha_builder.py`
Builds wallet quality features such as:
- rolling trade count
- rolling forward return
- rolling win rate
- rolling alpha proxy
- TP precision proxy
- recent streak

### `target_builder.py`
Builds BTC context features like:
- `btc_spot_return_5m`
- `btc_spot_return_15m`
- `btc_realized_vol_15m`
- `btc_volume_proxy`

### `contract_target_builder.py`
Builds event-style contract labels, including:
- `tp_before_sl_60m`
- `forward_return_15m`
- `mfe_60m`
- `mae_60m`

## Models / evaluation

### `supervised_models.py`
Trains supervised baseline models for:
- classification: `tp_before_sl_60m`
- regression: `forward_return_15m`

### `model_inference.py`
Loads trained supervised models and outputs:
- `p_tp_before_sl`
- `expected_return`
- `edge_score`

### `time_split_trainer.py`
Uses ordered train / validation / test splits instead of random splits.

### `walk_forward_evaluator.py`
Provides a simple walk-forward evaluation pass.

### `evaluator.py`
Writes research metrics such as:
- accuracy
- precision
- recall
- F1
- Sharpe-like metric
- drawdown

## Simulation / paper trading

### `pnl_engine.py`
Implements correct Polymarket-style share accounting.

Core formula:

```text
shares = capital_usdc / entry_price
pnl = shares * (exit_price - entry_price) - fees
```

### `position_manager.py`
Tracks open and closed paper positions using:
- `outcome_side` (`YES` / `NO`)
- `position_action` (`ENTER` / `EXIT`)
- share-based mark-to-market logic

### `path_replay_simulator.py`
Replays future price paths bar by bar and computes:
- entry time
- exit time
- holding time
- exit reason
- gross / net pnl
- MFE
- MAE
- max drawdown during trade

### `strategy_layers.py`
Starts separating:
- prediction layer
- entry rule layer
- exit rule layer

## UI

### `dashboard.py`
Streamlit dashboard with:
- **Overview**
- **Opportunities**
- **Markets & Whales**
- **Learning**
- **Raw Data**

Recent UI improvements include:
- better opportunity cards
- confidence bars
- most successful trades view
- replay metrics
- learning metrics surfaced in the browser

## Quick start

## 1) Install

```bash
python -m pip install -r requirements.txt
```

## 2) Create / validate local env

```bash
python api_setup.py
```

## 3) Run the system

```bash
python run_bot.py
```

In another terminal:

```bash
python -m streamlit run dashboard.py
```

Optional local API:

```bash
python -m uvicorn web_api:app --reload
```

Docs:

```text
http://127.0.0.1:8000/docs
```

## Recommended Windows usage

If the repo already exists locally:

```powershell
git pull origin main
python run_bot.py
```

And in a second PowerShell window:

```powershell
python -m streamlit run dashboard.py
```

## Main output files

Generated in `logs/`:

- `signals.csv`
- `daily_summary.txt`
- `markets.csv`
- `whales.csv`
- `market_distribution.csv`
- `alerts.csv`
- `positions.csv`
- `closed_positions.csv`
- `historical_dataset.csv`
- `btc_targets.csv`
- `contract_targets.csv`
- `wallet_alpha.csv`
- `wallet_alpha_history.csv`
- `supervised_eval.csv`
- `time_split_eval.csv`
- `path_replay_backtest.csv`
- `model_status.csv`

## Current reality

This repo is improving fast, but it is still a **research system under active refactor**.

The newer supervised / event-driven path is the direction of travel.
The older dummy RL path still exists in parts of the codebase, but it should no longer be treated as the core intelligence for the real goal.

## Next intended direction

- use Gamma/Data/CLOB more directly across the pipeline
- improve wallet rolling metrics further
- improve token-level historical labeling
- rank signals primarily from trained model outputs
- keep the whole project in paper/research mode
