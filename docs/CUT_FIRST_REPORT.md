# Cut First Report

Status: 2026-04-17
Scope: last 200 BTC closed trades plus candidate/forecast/runtime evidence

## Executive Summary

The first thing to cut is the BTC `rl_exit` path in live decisioning.

Why:
- In the last 200 BTC closed trades, `rl_exit` accounts for 148 trades.
- Those 148 `rl_exit` trades account for about `-81.60` net PnL.
- The full last-200 BTC sample is about `-80.13` net PnL, so the damage is overwhelmingly concentrated in `rl_exit`.

The second thing to cut is any path that allows entries with `IGNORE`-grade signals or near-zero conviction to become live positions.

Why:
- 193 of the last 200 BTC closed trades carry `signal_label=IGNORE` in the persisted trade record.
- Average `confidence_at_entry` on the `rl_exit` set is about `0.204`.
- No trades in the last-200 BTC sample had `confidence_at_entry >= 0.45`.

The third thing to cut is complexity that re-scores the same trade multiple times without preserving clean attribution.

Why:
- Forecast evaluation looks materially better than realized trade outcomes.
- `logs/btc_forecast_eval.csv` shows positive directional signal quality.
- `logs/closed_positions.csv` shows strongly negative realized outcomes.
- That gap implies the leak is downstream of forecasting.

## Evidence

### 1. Forecast quality is better than trade quality

From `logs/btc_forecast_eval.csv`:
- Total directional evaluations: 1766
- Overall directional accuracy: about `60.8%`
- Last 200 directional evaluations: about `74.6%`
- Overall average simulated `pnl_pct`: about `0.0038`

Interpretation:
- There may be a real BTC informational edge.
- The live trade pipeline is not preserving it.

### 2. Last 200 BTC closes are negative and concentrated in one exit path

From `logs/closed_positions.csv` filtered to BTC and sorted by `closed_at`:
- Last 200 BTC closes: `-80.1342` net PnL
- Win rate: `15.0%`
- Median PnL: `-0.0164`

By `close_reason`:
- `rl_exit`: 148 trades, `-81.6032` PnL, `14.9%` win rate
- `external_manual_close`: 42 trades, `+2.2055` PnL, `9.5%` win rate
- `take_profit_roi`: 4 trades, `+0.4599` PnL, `100%` win rate
- `stop_loss`: 3 trades, `-1.0613` PnL

Interpretation:
- The main loss engine is not stop-losses.
- The main loss engine is discretionary RL-driven exits.

### 3. Entry quality is too weak for a profitable live system

From the same last-200 BTC close sample:
- `signal_label=IGNORE`: 193 trades, about `-79.45` PnL
- `HIGHEST-RANKED PAPER SIGNAL`: 6 trades, about `-0.22` PnL
- Average `confidence_at_entry` for `rl_exit` trades: `0.2043`

Confidence buckets:
- `(0.15, 0.20]`: 88 trades, `-40.3244` PnL
- `(0.20, 0.25]`: 67 trades, `-33.0795` PnL
- `(0.25, 0.30]`: 16 trades, `-1.3602` PnL
- `(0.30, 1.0]`: 6 trades, `-5.8687` PnL

Interpretation:
- The system is taking a lot of low-conviction BTC risk.
- Either logging is drifting badly, or low-grade signals are becoming live trades.
- Both are unacceptable.

### 4. Trade attribution on the worst path is incomplete

Observed telemetry gaps in the same sample:
- `entry_signal_snapshot_json` was often `NaN`
- `entry_btc_predicted_return` often missing
- `entry_btc_forecast_confidence` often missing
- `runup_from_entry` and `drawdown_from_peak` mostly missing
- `execution_feedback.csv` does not exist
- `model_decisions.csv` does not exist

Interpretation:
- The project cannot cleanly prove whether the RL exit was correct.
- That alone is enough reason to demote or disable the RL exit path until better evidence exists.

## What To Cut First

### Cut 1. Disable BTC `rl_exit` from controlling live exits

Priority: Critical

Where:
- `supervisor.py` around the live/paper exit branch that closes with `reason="rl_exit"`

Why this is first:
- It is the single clearest concentrated loss source in realized BTC results.
- It is not supported by sufficiently complete attribution telemetry.
- The repo already has simpler exit paths that are easier to reason about.

Recommended action:
- Disable RL exit for BTC in live and paper stages.
- Keep it only in offline replay or shadow comparison mode.
- Route BTC exits through a smaller ruleset only:
  - hard emergency stop
  - stop loss
  - take profit
  - time stop
  - optional trailing stop

Success condition:
- New BTC trades no longer close with `close_reason=rl_exit`.
- Exit outcome quality becomes attributable and auditable.

### Cut 2. Block BTC entries below a hard confidence and signal-quality floor

Priority: Critical

Where:
- `signal_engine.py`
- `strategy_layers.py`
- any entry-open path that can bypass those labels

Why this is second:
- The last-200 BTC sample should not be dominated by `IGNORE`-labeled trades.
- Even if that label is stale, the average entry confidence is still too low.

Recommended action:
- Add a hard BTC live-entry floor such as:
  - `confidence_at_entry >= 0.30` minimum to paper
  - `confidence_at_entry >= 0.45` minimum to live
- Refuse live BTC entry when `signal_label == "IGNORE"` or watch-level only.
- Persist the exact entry decision snapshot used to open the trade.

Success condition:
- Every live BTC open can be traced to a positive, non-IGNORE signal state.

### Cut 3. Collapse the scoring stack to one decision score and one sizing score

Priority: High

Where:
- `signal_engine.py`
- `strategy_layers.py`
- `money_manager.py`
- `execution_engine.py`

Why this is third:
- Forecast, signal, strategy, execution, and sizing all modify trade quality.
- This makes it hard to know which layer actually made the decision.

Recommended action:
- For BTC v2, use only:
  - fair probability
  - market probability
  - edge after cost
  - size
- Treat TA, wallet, regime, and other overlays as explanatory metadata unless they have isolated proof.

Success condition:
- One row explains exactly why a BTC trade was taken or skipped.

### Cut 4. Remove or quarantine low-value BTC families until one family proves positive

Priority: High

Where:
- BTC market-family routing in runtime selection

Evidence from last 200 BTC closes:
- `btc_price_threshold`: 104 trades, `-74.10` PnL
- `btc_directional_intraday`: 75 trades, `-4.12` PnL
- `btc_other`: 13 trades, `-1.41` PnL
- `btc_downside_threshold`: 8 trades, `-0.50` PnL

Recommended action:
- Keep only one BTC family temporarily.
- The best candidate is `btc_directional_intraday`, not because it is good, but because it is far less bad than `btc_price_threshold`.
- Freeze `btc_price_threshold` first.

Success condition:
- Active BTC live capital is concentrated in one measurable family.

### Cut 5. Stop using live feedback multipliers as a control system

Priority: Medium

Where:
- `trade_feedback_learner.py`

Why:
- The current sample is too noisy and too negative to justify adaptive live multipliers.
- It is a multiplier engine, not a trustworthy online learner.

Recommended action:
- Keep reporting.
- Disable live rescaling effects on BTC entries/exits until a cleaner offline validation loop is in place.

Success condition:
- Learning stays analytical, not operational, until it proves value offline.

## What To Keep

- `btc_forecast_eval.py`
- `deployment_gate.py`
- reconciliation and state safety controls
- simple hard-stop and take-profit logic
- thesis documentation

These parts help answer whether there is a real edge and keep the process safe.

## Minimal BTC V2 Path

If the goal is to get to a profitable BTC bot faster, the shortest path is:

1. Keep one BTC family only.
2. Disable RL exits.
3. Disable live feedback multipliers.
4. Require a hard live-entry confidence floor.
5. Trade only when estimated `edge_after_cost > threshold`.
6. Use only simple exits.
7. Log one joined attribution row per trade:
   `forecast -> signal -> candidate -> order -> fill -> close -> pnl`

## Recommended Order Of Work

1. Disable BTC `rl_exit` in live and paper paths.
2. Add hard BTC entry floors and forbid `IGNORE` signals from opening trades.
3. Freeze `btc_price_threshold`.
4. Turn `trade_feedback_learner` into report-only mode for BTC.
5. Build a clean joined trade-attribution log for every new BTC trade.

## Bottom Line

The project should not try to become a smarter bot first.

It should become a narrower bot first.

The current evidence says:
- forecasting may have value
- realized trade conversion is broken
- RL exits are the first concrete thing to cut
- low-conviction BTC entries are the second
- duplicated scoring complexity is the third
