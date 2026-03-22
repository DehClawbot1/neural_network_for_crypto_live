# Neural Network for Crypto: Polymarket Public-Data Research + Paper-Trading System

This repository contains a local research and paper-trading system built around **public Polymarket data**.

It is designed to:
- monitor BTC-related prediction markets in real time
- watch public trader / whale activity
- rank paper-trading opportunities
- simulate fills and log paper trades
- build historical datasets for ML / RL research
- expose results through a browser dashboard and a local API

It is **not** a live trading or live betting system.

## Safety Mode

This project is intentionally configured for:
- **public-data monitoring**
- **paper trading**
- **simulation / backtesting**
- **research analytics**

It does **not**:
- connect to a live Polymarket account
- place real orders
- use live API keys for execution
- make real-money wagering decisions

## 🧠 Current Architecture

### Core Runtime

1. **`api_setup.py`**
   - validates `.env`
   - creates a safe paper-trading template if needed

2. **`run_bot.py`**
   - easiest one-command launcher
   - checks environment
   - checks weights
   - checks retraining status
   - starts the supervisor

3. **`supervisor.py`**
   - main continuous loop
   - fetches public data
   - builds features
   - scores opportunities
   - simulates paper trades
   - writes analytics/log outputs

### Data Collection

4. **`leaderboard_scraper.py`**
   - monitors public top crypto wallets from the Polymarket leaderboard
   - currently scans a much larger public wallet set for BTC-related activity
   - uses retry/backoff behavior for more resilient API collection

5. **`market_monitor.py`**
   - fetches BTC-related market snapshots
   - saves market tracking data to `logs/markets.csv`

6. **`whale_tracker.py`**
   - summarizes public wallet activity
   - writes whale tracking data to `logs/whales.csv`

7. **`alerts_engine.py`**
   - detects public-data signals such as probability moves and whale clustering
   - writes alerts to `logs/alerts.csv`

### Model / Feature Layer

8. **`feature_builder.py`**
   - builds grouped normalized features from public market + wallet activity
   - now includes grouped sub-scores such as:
     - `whale_pressure`
     - `market_structure_score`
     - `volatility_risk`
     - `time_decay_score`

9. **`signal_engine.py`**
   - converts grouped feature inputs into ranked paper-trading opportunities
   - outputs labels such as:
     - `IGNORE`
     - `LOW-CONFIDENCE WATCH`
     - `STRONG PAPER OPPORTUNITY`
     - `HIGHEST-RANKED PAPER SIGNAL`

10. **`polytrade_env.py`**
    - custom Gymnasium environment for RL training
    - updated for expanded Phase B feature vectors

11. **`rl_trainer.py`**
    - trains the PPO model
    - saves weights to `weights/ppo_polytrader.zip`

12. **`retrainer.py`**
    - checks historical dataset growth
    - triggers retraining when threshold conditions are met
    - writes model status to `logs/model_status.csv`

### Analytics / Simulation

13. **`simulation_engine.py`**
    - tracks richer simulated positions

14. **`trader_analytics.py`**
    - builds leaderboard / trader performance analytics
    - writes `logs/trader_analytics.csv`

15. **`backtester.py`**
    - computes a simple research backtest summary from ranked signal history
    - writes `logs/backtest_summary.csv`

16. **`historical_dataset_builder.py`**
    - consolidates logs into ML-friendly datasets
    - writes `logs/historical_dataset.csv`

17. **`autonomous_monitor.py`**
    - writes health / monitoring status for the local system
    - writes `logs/system_health.csv`

### User Interfaces

18. **`dashboard.py`**
    - Streamlit browser UI
    - friendlier tabs for:
      - opportunities
      - markets
      - whales & alerts
      - learning status
      - raw data

19. **`web_api.py`**
    - local FastAPI server
    - exposes information endpoints for local use only

20. **`position_manager.py`**
    - manages open and closed paper positions
    - tracks mark-to-market changes and simulated exit reasons

## 🚀 Installation

### 1) Clone the repository

```bash
git clone https://github.com/DehClawbot1/neural_network_for_crypto.git
cd neural_network_for_crypto
```

### 2) Optional: create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows CMD**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3) Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 🛠 Initial Setup

Run:

```bash
python api_setup.py
```

This creates a safe local `.env` for paper trading if one does not exist.

## 🤖 Easiest Way to Run Everything

### Bot

```bash
python run_bot.py
```

What it does:
- validates environment
- checks model weights
- checks whether retraining should happen
- starts the supervisor loop

### Browser Dashboard

In another terminal:

```bash
python -m streamlit run dashboard.py
```

### Local Information API

In another terminal:

```bash
python -m uvicorn web_api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## 🧪 Manual Commands

### Train / retrain the model manually

```bash
python rl_trainer.py
```

### Run the supervisor directly

```bash
python supervisor.py
```

### Run the backtester directly

```bash
python backtester.py
```

## 📊 Main Output Files

Generated in `logs/`:

- `signals.csv` - ranked paper-trading opportunities + grouped feature values
- `daily_summary.txt` - simulated paper-trade ledger
- `markets.csv` - BTC market snapshots
- `whales.csv` - public whale activity summaries
- `market_distribution.csv` - distribution of monitored wallet activity across markets
- `alerts.csv` - public-data alerts
- `positions.csv` - currently open simulated positions
- `closed_positions.csv` - simulated closes and exit reasons
- `trader_analytics.csv` - trader / wallet analytics
- `backtest_summary.csv` - backtest summary metrics
- `historical_dataset.csv` - ML-friendly consolidated dataset
- `model_status.csv` - retraining / model progress
- `system_health.csv` - monitoring status

## 📁 Repository Structure

```text
neural_network_for_crypto/
├── README.md
├── requirements.txt
├── .gitignore
├── .env
├── api_setup.py
├── run_bot.py
├── dashboard.py
├── web_api.py
├── leaderboard_scraper.py
├── market_monitor.py
├── whale_tracker.py
├── alerts_engine.py
├── feature_builder.py
├── signal_engine.py
├── polytrade_env.py
├── rl_trainer.py
├── retrainer.py
├── simulation_engine.py
├── trader_analytics.py
├── backtester.py
├── historical_dataset_builder.py
├── autonomous_monitor.py
├── supervisor.py
├── logs/
└── weights/
```

## 🧠 Model Learning Flow

The project is designed to keep improving over time in **paper mode**:

1. gather new public market + wallet data
2. score and simulate paper opportunities
3. store the outputs into logs and datasets
4. grow the historical dataset
5. trigger retraining when enough new data exists
6. continue using the updated saved model

This means the neural-network side is:
- **saved to disk** as weights
- **loaded during runtime**
- **updated over time** via retraining logic

## Current Phases Implemented

### Phase 1
- market tracker
- odds dashboard
- BTC market watcher
- whale activity tracker
- alerts when probabilities move

### Phase 2
- leaderboard / trader performance analytics
- strategy backtester
- historical dataset builder for ML

### Phase 3 (safe version)
- richer simulation hooks
- autonomous monitoring
- retraining support
- local information API

### Phase A
- grouped feature architecture
- stronger normalization
- sub-score calculation

### Phase B
- expanded observation vector for the model
- backward-compatible runtime fallback for older weights
- richer feature logging into `signals.csv`

### Phase C
- browser-visible factor / confidence breakdowns
- learning status display
- user-friendlier dashboard organization

## Notes

- This project is for **research, simulation, monitoring, and paper-trading only**.
- It uses **public Polymarket information**.
- It does **not** use live execution credentials.
- It does **not** place real bets.
