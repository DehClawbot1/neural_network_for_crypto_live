import logging
import os
import signal
import time
import re
from pathlib import Path

from brain_paths import resolve_brain_context
from brain_training_orchestrator import (
    build_family_datasets,
    evaluate_brain_candidate_rows,
    register_and_promote_brain_models,
    train_brain_models,
)
from historical_dataset_builder import HistoricalDatasetBuilder
from target_builder import TargetBuilder
from dataset_aligner import DatasetAligner
from supervised_trainer import SupervisedTrainer
from evaluator import Evaluator
from wallet_alpha_builder import WalletAlphaBuilder
from feature_ablation import FeatureAblationReporter
from walk_forward_evaluator import WalkForwardEvaluator
from time_split_trainer import TimeSplitTrainer
from path_replay_simulator import PathReplaySimulator
from clob_history import CLOBHistoryClient
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# BUG FIX (BUG 3): Hard limit on pipeline runtime
MAX_PIPELINE_SECONDS = int(os.getenv("MAX_PIPELINE_SECONDS", "600"))  # 10 min default

def _default_max_clob_tokens():
    always_on_only = os.getenv("ALWAYS_ON_ONLY", "true").strip().lower() in {"1", "true", "yes", "on"}
    # Pinned single-market mode can use a leaner research universe.
    return 200 if always_on_only else 500

MAX_CLOB_TOKENS = int(os.getenv("MAX_CLOB_TOKENS", str(_default_max_clob_tokens())))  # cap tokens fetched
MAX_CLOB_DAYS = int(os.getenv("MAX_CLOB_DAYS", "3"))  # reduce from 7 to 3 days
CLOB_SELECTION_AUDIT_FILE = Path("logs") / "clob_selection_audit.csv"
CANDIDATE_CYCLE_STATS_FILE = Path("logs") / "candidate_cycle_stats.csv"



class PipelineTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise PipelineTimeout(f"Research pipeline exceeded {MAX_PIPELINE_SECONDS}s hard limit")


def _safe_read_csv(path: Path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        if not pd.notna(parsed):
            return float(default)
        return parsed
    except Exception:
        return float(default)


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            return float(value) != 0.0
        except Exception:
            return False
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _build_ranked_clob_token_ids(markets_df: pd.DataFrame) -> list[str]:
    if markets_df is None or markets_df.empty:
        return []

    work = markets_df.copy()
    if "end_date" in work.columns:
        work["end_date"] = pd.to_datetime(work["end_date"], utc=True, errors="coerce")
    else:
        work["end_date"] = pd.NaT

    now_ts = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow()

    def _market_priority(row: pd.Series) -> int:
        text_blob = " ".join(
            str(row.get(col) or "")
            for col in ("market_family", "market_title", "question", "slug")
        ).lower()
        if "btc" in text_blob or "bitcoin" in text_blob:
            return 0
        if "weather" in text_blob or "temperature" in text_blob:
            return 1
        return 2

    def _spread_penalty(row: pd.Series) -> float:
        spread = _safe_float(row.get("spread"), 1.0)
        if spread > 0:
            return spread
        best_bid = _safe_float(row.get("best_bid"), 0.0)
        best_ask = _safe_float(row.get("best_ask"), 0.0)
        if best_bid > 0 and best_ask > 0 and best_ask >= best_bid:
            return best_ask - best_bid
        return 1.0

    work["_active_rank"] = work.get("active", False).apply(lambda v: 0 if _is_truthy(v) else 1)
    work["_closed_rank"] = work.get("closed", False).apply(lambda v: 1 if _is_truthy(v) else 0)
    work["_expired_rank"] = work["end_date"].apply(lambda ts: 1 if pd.notna(ts) and ts <= now_ts else 0)
    work["_market_priority"] = work.apply(_market_priority, axis=1)
    work["_liquidity_rank"] = pd.to_numeric(work.get("liquidity", 0.0), errors="coerce").fillna(0.0)
    work["_volume_rank"] = pd.to_numeric(work.get("volume", 0.0), errors="coerce").fillna(0.0)
    work["_spread_rank"] = work.apply(_spread_penalty, axis=1)

    sort_cols = [
        "_active_rank",
        "_closed_rank",
        "_expired_rank",
        "_market_priority",
        "_liquidity_rank",
        "_volume_rank",
        "_spread_rank",
    ]
    ascending = [True, True, True, True, False, False, True]
    work = work.sort_values(sort_cols, ascending=ascending, kind="mergesort")

    ranked_tokens: list[str] = []
    seen_tokens: set[str] = set()
    for _, row in work.iterrows():
        for col in ("yes_token_id", "no_token_id"):
            token = str(row.get(col) or "").strip().strip('"').strip("'")
            if not token or not re.fullmatch(r"\d{8,}", token):
                continue
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            ranked_tokens.append(token)
    return ranked_tokens


def _summarize_ranked_clob_selection(markets_df: pd.DataFrame, selected_token_ids: list[str]) -> dict:
    if markets_df is None or markets_df.empty or not selected_token_ids:
        return {}

    selected = set(str(token).strip() for token in selected_token_ids if str(token).strip())
    if not selected:
        return {}

    work = markets_df.copy()
    mask = work.get("yes_token_id", pd.Series(index=work.index, dtype=object)).astype(str).isin(selected)
    mask = mask | work.get("no_token_id", pd.Series(index=work.index, dtype=object)).astype(str).isin(selected)
    work = work[mask].copy()
    if work.empty:
        return {}

    def _family_bucket(row: pd.Series) -> str:
        text_blob = " ".join(
            str(row.get(col) or "")
            for col in ("market_family", "market_title", "question", "slug")
        ).lower()
        if "btc" in text_blob or "bitcoin" in text_blob:
            return "btc"
        if "weather" in text_blob or "temperature" in text_blob:
            return "weather"
        return "other"

    liquidity = pd.to_numeric(work.get("liquidity", 0.0), errors="coerce").fillna(0.0)
    volume = pd.to_numeric(work.get("volume", 0.0), errors="coerce").fillna(0.0)
    spread = pd.to_numeric(work.get("spread", 0.0), errors="coerce").fillna(0.0)
    family_counts = work.apply(_family_bucket, axis=1).value_counts().to_dict()

    return {
        "markets": int(len(work)),
        "tokens": int(len(selected)),
        "btc_markets": int(family_counts.get("btc", 0)),
        "weather_markets": int(family_counts.get("weather", 0)),
        "other_markets": int(family_counts.get("other", 0)),
        "median_liquidity": round(float(liquidity.median()), 4) if not liquidity.empty else 0.0,
        "median_volume": round(float(volume.median()), 4) if not volume.empty else 0.0,
        "median_spread": round(float(spread.replace(0, pd.NA).dropna().median()), 6) if not spread.empty else 0.0,
    }


def _append_clob_selection_audit(
    summary: dict,
    *,
    total_ranked_tokens: int,
    max_clob_tokens: int,
    always_on_only: bool,
    max_clob_days: int,
    output_path: Path = CLOB_SELECTION_AUDIT_FILE,
) -> None:
    if not summary:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "total_ranked_tokens": int(total_ranked_tokens),
        "selected_tokens": int(summary.get("tokens", 0) or 0),
        "selected_markets": int(summary.get("markets", 0) or 0),
        "btc_markets": int(summary.get("btc_markets", 0) or 0),
        "weather_markets": int(summary.get("weather_markets", 0) or 0),
        "other_markets": int(summary.get("other_markets", 0) or 0),
        "median_liquidity": float(summary.get("median_liquidity", 0.0) or 0.0),
        "median_volume": float(summary.get("median_volume", 0.0) or 0.0),
        "median_spread": float(summary.get("median_spread", 0.0) or 0.0),
        "max_clob_tokens": int(max_clob_tokens),
        "max_clob_days": int(max_clob_days),
        "always_on_only": bool(always_on_only),
        "selection_utilization": round(
            float(summary.get("tokens", 0) or 0) / max(1, int(max_clob_tokens)),
            6,
        ),
        "selection_coverage": round(
            float(summary.get("tokens", 0) or 0) / max(1, int(total_ranked_tokens)),
            6,
        ),
    }
    payload = pd.DataFrame([row])
    header = not output_path.exists()
    payload.to_csv(output_path, mode="a", header=header, index=False)


def _refresh_clob_selection_outcome_audit(
    audit_path: Path = CLOB_SELECTION_AUDIT_FILE,
    candidate_stats_path: Path = CANDIDATE_CYCLE_STATS_FILE,
) -> None:
    if not audit_path.exists() or not candidate_stats_path.exists():
        return

    try:
        audit_df = pd.read_csv(audit_path, engine="python", on_bad_lines="skip")
        cycle_df = pd.read_csv(candidate_stats_path, engine="python", on_bad_lines="skip")
    except Exception:
        return

    if audit_df.empty or cycle_df.empty or "timestamp" not in audit_df.columns or "timestamp" not in cycle_df.columns:
        return

    audit_df["timestamp"] = pd.to_datetime(audit_df["timestamp"], utc=True, errors="coerce")
    cycle_df["timestamp"] = pd.to_datetime(cycle_df["timestamp"], utc=True, errors="coerce")
    audit_df = audit_df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    cycle_df = cycle_df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    downstream_cols = {
        "linked_cycle_id": None,
        "linked_cycle_timestamp": None,
        "minutes_to_linked_cycle": None,
        "downstream_candidates_seen": 0,
        "downstream_candidates_tradable": 0,
        "downstream_candidates_rejected": 0,
        "downstream_entries_sent": 0,
        "downstream_fills_received": 0,
    }
    for col, default in downstream_cols.items():
        if col not in audit_df.columns:
            audit_df[col] = default

    cycle_rows = cycle_df.dropna(subset=["timestamp"]).to_dict("records")
    if not cycle_rows:
        return

    updated = False
    schema_changed = False
    for idx, row in audit_df.iterrows():
        audit_ts = row.get("timestamp")
        if pd.isna(audit_ts):
            continue
        if str(row.get("linked_cycle_id") or "").strip():
            continue

        match = next((c for c in cycle_rows if c["timestamp"] >= audit_ts), None)
        if match is None:
            continue

        delta_minutes = (match["timestamp"] - audit_ts).total_seconds() / 60.0
        audit_df.at[idx, "linked_cycle_id"] = str(match.get("cycle_id") or "")
        audit_df.at[idx, "linked_cycle_timestamp"] = match["timestamp"].isoformat()
        audit_df.at[idx, "minutes_to_linked_cycle"] = round(float(delta_minutes), 4)
        audit_df.at[idx, "downstream_candidates_seen"] = int(match.get("candidates_seen", 0) or 0)
        audit_df.at[idx, "downstream_candidates_tradable"] = int(match.get("candidates_tradable", 0) or 0)
        audit_df.at[idx, "downstream_candidates_rejected"] = int(match.get("candidates_rejected", 0) or 0)
        audit_df.at[idx, "downstream_entries_sent"] = int(match.get("entries_sent", 0) or 0)
        audit_df.at[idx, "downstream_fills_received"] = int(match.get("fills_received", 0) or 0)
        updated = True

    if any(col not in pd.read_csv(audit_path, nrows=0).columns for col in downstream_cols):
        schema_changed = True

    if updated or schema_changed:
        audit_df.to_csv(audit_path, index=False)


def _ensure_dashboard_supervised_eval(logs_dir="logs"):
    logs_path = Path(logs_dir)
    target_file = logs_path / "supervised_eval.csv"

    time_split_df = _safe_read_csv(logs_path / "time_split_eval.csv")
    walk_forward_df = _safe_read_csv(logs_path / "walk_forward_eval.csv")
    stage2_df = _safe_read_csv(logs_path / "stage2_temporal_eval.csv")
    backtest_df = _safe_read_csv(logs_path / "backtest_summary.csv")
    source_paths = [
        logs_path / "time_split_eval.csv",
        logs_path / "walk_forward_eval.csv",
        logs_path / "stage2_temporal_eval.csv",
        logs_path / "backtest_summary.csv",
    ]
    source_mtimes = [path.stat().st_mtime for path in source_paths if path.exists()]
    target_mtime = target_file.stat().st_mtime if target_file.exists() else None
    legacy_eval_enabled = os.getenv("ENABLE_LEGACY_BTC_DIRECTION_MODEL", "false").strip().lower() in {"1", "true", "yes", "on"}
    if legacy_eval_enabled and target_file.exists():
        return
    if target_file.exists() and target_mtime is not None and source_mtimes and target_mtime >= max(source_mtimes):
        return

    accuracy = None
    rows_evaluated = None
    evaluation_split = "fallback"
    metric_source = None

    if not time_split_df.empty and "test_accuracy" in time_split_df.columns:
        accuracy = pd.to_numeric(time_split_df["test_accuracy"], errors="coerce").dropna()
        accuracy = float(accuracy.iloc[-1]) if not accuracy.empty else None
        if "test_rows" in time_split_df.columns:
            rows = pd.to_numeric(time_split_df["test_rows"], errors="coerce").dropna()
            rows_evaluated = int(rows.iloc[-1]) if not rows.empty else None
        evaluation_split = "time_split_test"
        metric_source = "time_split_eval.csv"
    elif not walk_forward_df.empty and "accuracy" in walk_forward_df.columns:
        accuracy = pd.to_numeric(walk_forward_df["accuracy"], errors="coerce").dropna()
        accuracy = float(accuracy.iloc[-1]) if not accuracy.empty else None
        if "test_rows" in walk_forward_df.columns:
            rows = pd.to_numeric(walk_forward_df["test_rows"], errors="coerce").dropna()
            rows_evaluated = int(rows.iloc[-1]) if not rows.empty else None
        evaluation_split = "walk_forward"
        metric_source = "walk_forward_eval.csv"
    elif not stage2_df.empty and "temporal_walk_forward_accuracy" in stage2_df.columns:
        accuracy = pd.to_numeric(stage2_df["temporal_walk_forward_accuracy"], errors="coerce").dropna()
        accuracy = float(accuracy.iloc[-1]) if not accuracy.empty else None
        evaluation_split = "stage2_temporal_walk_forward"
        metric_source = "stage2_temporal_eval.csv"

    sharpe = None
    max_drawdown = None
    mean_strategy_return = None
    if not backtest_df.empty:
        if "sharpe_like" in backtest_df.columns:
            series = pd.to_numeric(backtest_df["sharpe_like"], errors="coerce").dropna()
            sharpe = float(series.iloc[-1]) if not series.empty else None
        if "max_drawdown" in backtest_df.columns:
            series = pd.to_numeric(backtest_df["max_drawdown"], errors="coerce").dropna()
            max_drawdown = float(series.iloc[-1]) if not series.empty else None
        if "average_pnl" in backtest_df.columns:
            series = pd.to_numeric(backtest_df["average_pnl"], errors="coerce").dropna()
            mean_strategy_return = float(series.iloc[-1]) if not series.empty else None

    if accuracy is None and sharpe is None and max_drawdown is None and mean_strategy_return is None:
        return

    payload = pd.DataFrame([
        {
            "accuracy": accuracy,
            "precision": None,
            "recall": None,
            "f1": None,
            "mean_strategy_return": mean_strategy_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "rows_evaluated": rows_evaluated,
            "evaluation_split": evaluation_split,
            "metric_source": metric_source or "fallback_artifacts",
            "generated_at": pd.Timestamp.utcnow().isoformat(),
        }
    ])
    payload.to_csv(target_file, index=False)
    logging.info("Wrote fallback supervised_eval.csv for dashboard compatibility.")


def run_research_pipeline():
    # Set hard timeout (Unix only; on Windows this is a no-op but the
    # per-step checks below still enforce the budget)
    pipeline_start = time.time()
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(MAX_PIPELINE_SECONDS)
    except (AttributeError, ValueError):
        pass  # Windows doesn't have SIGALRM

    def _check_budget(step_name):
        elapsed = time.time() - pipeline_start
        remaining = MAX_PIPELINE_SECONDS - elapsed
        if remaining <= 0:
            raise PipelineTimeout(f"Pipeline budget exhausted at step '{step_name}' after {elapsed:.0f}s")
        logging.info("Pipeline budget: %.0fs elapsed, %.0fs remaining (at %s)", elapsed, remaining, step_name)

    _pipeline_step = "init"

    def _run_step(step_name, fn):
        nonlocal _pipeline_step
        _pipeline_step = step_name
        try:
            fn()
        except (ValueError, TypeError) as _exc:
            logging.error(
                "Research pipeline step '%s' raised %s: %s",
                step_name, type(_exc).__name__, _exc, exc_info=True,
            )
            raise

    try:
        _refresh_clob_selection_outcome_audit()
        _pipeline_step = "resolve_contexts"
        always_on_only = os.getenv("ALWAYS_ON_ONLY", "true").strip().lower() in {"1", "true", "yes", "on"}
        btc_context = resolve_brain_context("btc", shared_logs_dir="logs", shared_weights_dir="weights")
        weather_context = resolve_brain_context("weather_temperature", shared_logs_dir="logs", shared_weights_dir="weights")
        logging.info("Building BTC direction targets...")
        _run_step("btc_targets", lambda: TargetBuilder().write(days=30, horizon_minutes=60))
        _check_budget("btc_targets")

        if os.getenv("ENABLE_LEGACY_BTC_DIRECTION_MODEL", "false").strip().lower() in {"1", "true", "yes", "on"}:
            logging.info("Building BTC-brain aligned dataset for legacy direction model...")
            _run_step("historical_dataset", lambda: HistoricalDatasetBuilder(brain_context=btc_context).write())
            _run_step("dataset_aligner", lambda: DatasetAligner(logs_dir=str(btc_context.logs_dir), shared_logs_dir="logs").write())
            _check_budget("historical_dataset")
            logging.info("Training legacy supervised BTC direction model...")
            _run_step("supervised_trainer", lambda: SupervisedTrainer(logs_dir=str(btc_context.logs_dir), weights_dir=str(btc_context.weights_dir)).train())
            _run_step("evaluator", lambda: Evaluator(logs_dir=str(btc_context.logs_dir), weights_dir=str(btc_context.weights_dir)).evaluate())
        else:
            logging.info("Skipping legacy btc_direction_model path; runtime scoring uses tp/return/stage1/stage2 artifacts.")

        _check_budget("pre_clob_fetch")

        # ── Cap number of tokens fetched ──
        logging.info("Fetching token-level CLOB price history...")
        markets_df = pd.read_csv("logs/markets.csv", engine="python", on_bad_lines="skip") if HistoricalDatasetBuilder().logs_dir.joinpath("markets.csv").exists() else pd.DataFrame()
        token_ids = _build_ranked_clob_token_ids(markets_df)

        # Cap at MAX_CLOB_TOKENS to prevent 15+ min fetch times
        total_ranked_tokens = len(token_ids)
        if len(token_ids) > MAX_CLOB_TOKENS:
            logging.warning(
                "Capping CLOB token fetch from %d to %d tokens (set MAX_CLOB_TOKENS to change)",
                len(token_ids), MAX_CLOB_TOKENS,
            )
            token_ids = token_ids[:MAX_CLOB_TOKENS]
        selection_summary = _summarize_ranked_clob_selection(markets_df, token_ids)
        if selection_summary:
            logging.info(
                "CLOB selection summary: tokens=%d markets=%d btc=%d weather=%d other=%d median_liquidity=%.2f median_volume=%.2f median_spread=%.4f",
                selection_summary["tokens"],
                selection_summary["markets"],
                selection_summary["btc_markets"],
                selection_summary["weather_markets"],
                selection_summary["other_markets"],
                selection_summary["median_liquidity"],
                selection_summary["median_volume"],
                selection_summary["median_spread"],
            )
            _append_clob_selection_audit(
                selection_summary,
                total_ranked_tokens=total_ranked_tokens,
                max_clob_tokens=MAX_CLOB_TOKENS,
                always_on_only=always_on_only,
                max_clob_days=MAX_CLOB_DAYS,
            )
            _refresh_clob_selection_outcome_audit()

        if token_ids:
            _run_step("clob_history", lambda: CLOBHistoryClient().append_history(token_ids, days=MAX_CLOB_DAYS, interval="1m"))
        _check_budget("clob_history")

        logging.info("Building contract-level labels and wallet alpha...")
        _run_step("wallet_alpha", lambda: WalletAlphaBuilder().write())
        _check_budget("wallet_alpha")

        _run_step("family_datasets", lambda: build_family_datasets(
            shared_logs_dir="logs",
            shared_weights_dir="weights",
            forward_minutes=15,
            max_hold_minutes=60,
            tp_move=0.04,
            sl_move=0.03,
        ))
        _check_budget("brain_datasets")

        try:
            targets_path = btc_context.logs_dir / "contract_targets.csv"
            alpha_history_path = Path("logs") / "wallet_alpha_history.csv"
            if targets_path.exists() and alpha_history_path.exists():
                targets = pd.read_csv(targets_path, engine="python", on_bad_lines="skip")
                alpha_hist = pd.read_csv(alpha_history_path, engine="python", on_bad_lines="skip")
                if not targets.empty and not alpha_hist.empty and "timestamp" in targets.columns and "timestamp" in alpha_hist.columns:
                    logging.info("Merging point-in-time wallet alpha into target features...")
                    targets["timestamp"] = pd.to_datetime(targets["timestamp"], utc=True, errors="coerce")
                    alpha_hist["timestamp"] = pd.to_datetime(alpha_hist["timestamp"], utc=True, errors="coerce")
                    join_key = "wallet_copied" if "wallet_copied" in targets.columns else "trader_wallet"
                    if join_key in targets.columns and "wallet_copied" in alpha_hist.columns:
                        if join_key != "wallet_copied":
                            alpha_hist = alpha_hist.rename(columns={"wallet_copied": join_key})
                        targets = targets.dropna(subset=["timestamp", join_key])
                        alpha_hist = alpha_hist.dropna(subset=["timestamp", join_key])
                        merged_parts = []
                        for wallet, group in targets.groupby(join_key):
                            history = alpha_hist[alpha_hist[join_key] == wallet]
                            if history.empty:
                                merged_parts.append(group)
                                continue
                            merged = pd.merge_asof(
                                group.sort_values("timestamp"),
                                history.sort_values("timestamp"),
                                on="timestamp",
                                direction="backward",
                            )
                            merged_parts.append(merged.loc[:, ~merged.columns.duplicated()])
                        if merged_parts:
                            targets = pd.concat(merged_parts, ignore_index=True)
                            targets = targets.loc[:, ~targets.columns.duplicated()]
                            targets.to_csv(targets_path, index=False)
                            logging.info("Alpha merge complete. Targets now enriched with wallet context.")
        except (ValueError, TypeError) as _merge_exc:
            logging.error("Research pipeline step 'alpha_merge' failed: %s", _merge_exc, exc_info=True)
            raise

        _check_budget("alpha_merge")
        for context in (btc_context, weather_context):
            _step_train = f"{context.brain_id}_training"
            _step_registry = f"{context.brain_id}_registry"
            _candidate_weights_dir_holder = [None]

            def _do_train(ctx=context, _holder=_candidate_weights_dir_holder):
                _holder[0] = train_brain_models(ctx, candidate_prefix="research")

            _run_step(_step_train, _do_train)
            candidate_weights_dir = _candidate_weights_dir_holder[0]
            _check_budget(_step_train)

            run_id = pd.Timestamp.utcnow().strftime(f"research_{context.market_family}_%Y%m%d%H%M%S")
            _eval_rows_holder = [pd.DataFrame()]

            def _do_eval(ctx=context, _rid=run_id, _cwd=candidate_weights_dir, _holder=_eval_rows_holder):
                _holder[0] = evaluate_brain_candidate_rows(ctx, run_id=_rid, candidate_weights_dir=_cwd)

            _run_step(f"{context.brain_id}_eval", _do_eval)
            candidate_rows = _eval_rows_holder[0]

            _reg_holder = [pd.DataFrame()]

            def _do_register(ctx=context, _rows=candidate_rows, _cwd=candidate_weights_dir, _holder=_reg_holder):
                _holder[0] = register_and_promote_brain_models(ctx, candidate_rows=_rows, candidate_weights_dir=_cwd)

            _run_step(_step_registry, _do_register)
            registered = _reg_holder[0]

            if not registered.empty and "promotion_status" in registered.columns:
                promoted = registered[registered["promotion_status"].fillna("").astype(str) == "promoted"]
            else:
                promoted = pd.DataFrame()
            logging.info(
                "[%s] Model registry updated with %s rows; promoted %s artifact slices.",
                context.brain_id,
                len(registered.index) if not registered.empty else 0,
                len(promoted.index) if not promoted.empty else 0,
            )
            _check_budget(_step_registry)

        btc_logs_dir = str(btc_context.logs_dir)
        _btc_only_steps = [
            ("walk_forward_eval", lambda: WalkForwardEvaluator(logs_dir=btc_logs_dir).evaluate()),
            ("time_split_trainer", lambda: TimeSplitTrainer(logs_dir=btc_logs_dir).run()),
            ("feature_ablation", lambda: FeatureAblationReporter(logs_dir=btc_logs_dir).write()),
            ("path_replay", lambda: PathReplaySimulator(logs_dir=btc_logs_dir, shared_logs_dir="logs").write()),
            ("dashboard_eval", lambda: _ensure_dashboard_supervised_eval(btc_logs_dir)),
        ]
        for _step_name, _step_fn in _btc_only_steps:
            try:
                _step_fn()
            except (ValueError, TypeError) as _step_exc:
                logging.error("Research pipeline step '%s' failed: %s", _step_name, _step_exc, exc_info=True)
                raise
            _check_budget(_step_name)

        elapsed = time.time() - pipeline_start
        logging.info("Research pipeline complete in %.0fs.", elapsed)

    except PipelineTimeout as exc:
        logging.warning("Pipeline timeout: %s — partial artifacts are usable.", exc)
    except (ValueError, TypeError) as exc:
        logging.error(
            "Research pipeline failed at step '%s': %s: %s",
            _pipeline_step, type(exc).__name__, exc, exc_info=True,
        )
        raise
    finally:
        try:
            signal.alarm(0)
        except (AttributeError, ValueError):
            pass


if __name__ == "__main__":
    run_research_pipeline()
