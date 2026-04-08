# Audit Report

Baseline:
- Target branch: `main`
- Target commit at audit start: `14afdbd`
- Pre-existing drift excluded from the audit target: `.claude/worktrees/brave-bell`
- Known environment blocker: `pytest tests/test_supervisor_smoke.py` fails during `scipy/sklearn` import, before repo-local supervisor logic executes

## Audit Ledger

| ID | Severity | Type | Subsystem | Symptom | Root Cause | Proof | Patch Status | Verification Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-001 | P1 | repo bug | heartbeat logging | `service_heartbeats.csv` had mixed row widths and pandas only loaded `30270` rows from `41829` lines | `AutonomousMonitor.write_heartbeat()` expanded arbitrary `extra` keys into CSV columns, so the file schema changed mid-stream | `logs/service_heartbeats.csv: lines=41829 pandas_rows=30270` | Fixed | `ops_state_sync._read_service_heartbeats_csv()` now recovers `41828` rows; `py_compile` passed |
| A-002 | P1 | repo bug | ops state sync | DB sync undercounted heartbeats because malformed CSV rows were skipped | `ops_state_sync.py` used generic `pd.read_csv(... on_bad_lines="skip")` on a schema-drifting heartbeat file | flexible parser loaded all rows and preserved the overflow payload in `rows` | Fixed | `rows_loaded_flexible 41828`; `py_compile` passed |
| A-003 | P1 | repo bug | runtime signal logging | `raw_candidates.csv` and `signals.csv` suffered append-schema drift | `supervisor.log_raw_candidates()` appended the full dynamic frame; `log_ranked_signal()` emitted a narrower row shape than the historical file header | `logs/raw_candidates.csv: lines=1797059 pandas_rows=57903`; `logs/signals.csv` row-width mismatch around lines `10662+` | Fixed for future writes | focused test suite passed; `py_compile` passed |
| A-004 | P2 | repo bug | test suite | `tests/test_strategy_layers.py` failed against the current soft-penalty entry logic | tests still asserted hard veto behavior after the strategy layer was intentionally relaxed | `2 failed, 21 passed` before patch in focused pytest slice | Fixed | focused suite now `28 passed` |
| A-005 | P3 | config/doc drift | launcher/docs | startup text and docs described outdated behavior | `run_bot.py` still said "sleep 60 seconds and repeat"; `README.md` still referenced `PLAN.md` | direct source inspection | Fixed | `py_compile` passed |
| E-001 | P1 | environment blocker | verification environment | broad supervisor smoke cannot import `supervisor` | external `scipy/sklearn` import stack fails with `TypeError: issubclass() arg 2 must be a class...` | `python -m pytest tests/test_supervisor_smoke.py -q` | Not patched in repo | Blocker remains |
| O-001 | P1 | operational gap | bot supervision | the bot previously stopped cycling for a long window with all services going silent together | bot process was not being continuously supervised by a resilient service/scheduler path | `service_heartbeats.csv` gap evidence and prior host/process audit | Not patched in repo | Operational follow-up needed |
| D-001 | P2 | data integrity risk | historical audit artifacts | existing `raw_candidates.csv`, `signals.csv`, and `service_heartbeats.csv` already contain malformed historical rows from older schema drift | historical files were written before the schema-stabilizing patch; auto-rewriting them would mutate audit evidence | line-count vs pandas-row-count checks | Root cause patched; historical files preserved | Offline repair/rotation still recommended |

## Patch Batches Applied

### Batch A: live correctness and safety
- Stabilized heartbeat writes to a fixed schema in `autonomous_monitor.py`.
- Added a flexible heartbeat reader in `ops_state_sync.py` so malformed historical heartbeat rows are no longer lost during DB sync.
- Stabilized future `raw_candidates.csv` and `signals.csv` writes in `supervisor.py` by forcing explicit legacy-compatible column sets.

Verification:
- `python -m py_compile autonomous_monitor.py ops_state_sync.py supervisor.py run_bot.py tests\\test_strategy_layers.py`
- `python -m pytest tests\\test_strategy_layers.py tests\\test_feature_builder.py tests\\test_contract_target_builder.py tests\\test_contract_target_builder_horizon.py tests\\test_btc_forecast_eval.py tests\\test_btc_live_price_tracker.py tests\\test_trade_quality.py -q`

### Batch B: config, startup, and operability
- Corrected the misleading startup cadence message in `run_bot.py`.
- Corrected the README roadmap reference so it no longer points to a missing `PLAN.md`.
- Updated stale strategy tests to match the current soft-penalty behavior in `strategy_layers.py`.

Verification:
- included in the Batch A compile/test pass above

## Remaining Risks

1. Historical log drift remains in existing large CSV artifacts.
- `raw_candidates.csv` is the most severe case: current generic pandas reads only a small fraction of the total lines because earlier schema drift already malformed the file.
- The root cause for future writes is patched, but the existing file was preserved as audit evidence and was not rewritten automatically.

2. Broad supervisor verification is still blocked by the environment.
- The `scipy/sklearn` import-time crash prevents broad `supervisor` smoke tests from being used as final truth.
- Narrow module-level tests and compile checks are passing, but a clean environment is still needed for full end-to-end supervisor verification.

3. Process supervision remains an operational issue.
- The codebase now has better diagnostics, but the bot still needs an external restart/watchdog mechanism so a stopped process does not turn into a silent overnight gap.

## Recommended Next Actions

1. Repair or rotate the malformed historical audit artifacts offline.
- Safest option: archive the current malformed `raw_candidates.csv` and `signals.csv`, then start fresh stabilized files on the patched code.
- If the historical data must be preserved for training, write a one-off repair script that reconstructs rows using the known legacy/current schemas.

2. Fix the environment-level `scipy/sklearn` import blocker.
- Until that is fixed, broad supervisor tests will remain partially blind.

3. Put the bot under resilient process supervision.
- Use Task Scheduler, NSSM, or another restart-capable runner, plus a "no heartbeat in N minutes" alert.
