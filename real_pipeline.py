import logging
import os
import signal
import time
import re

from historical_dataset_builder import HistoricalDatasetBuilder
from target_builder import TargetBuilder
from dataset_aligner import DatasetAligner
from supervised_trainer import SupervisedTrainer
from evaluator import Evaluator
from supervised_models import SupervisedModels
from stage1_models import Stage1Models
from sequence_feature_builder import SequenceFeatureBuilder
from stage2_temporal_models import Stage2TemporalModels
from contract_target_builder import ContractTargetBuilder
from wallet_alpha_builder import WalletAlphaBuilder
from walk_forward_evaluator import WalkForwardEvaluator
from time_split_trainer import TimeSplitTrainer
from path_replay_simulator import PathReplaySimulator
from clob_history import CLOBHistoryClient
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── BUG FIX (BUG 3): Hard limit on pipeline runtime ──
MAX_PIPELINE_SECONDS = int(os.getenv("MAX_PIPELINE_SECONDS", "600"))  # 10 min default
MAX_CLOB_TOKENS = int(os.getenv("MAX_CLOB_TOKENS", "50"))  # cap tokens fetched
MAX_CLOB_DAYS = int(os.getenv("MAX_CLOB_DAYS", "3"))  # reduce from 7 to 3 days


class PipelineTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise PipelineTimeout(f"Research pipeline exceeded {MAX_PIPELINE_SECONDS}s hard limit")


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

    try:
        logging.info("Building BTC direction targets...")
        TargetBuilder().write(days=30, horizon_minutes=60)
        _check_budget("btc_targets")

        logging.info("Building historical dataset...")
        HistoricalDatasetBuilder().write()
        DatasetAligner().write()
        _check_budget("historical_dataset")

        if os.getenv("ENABLE_LEGACY_BTC_DIRECTION_MODEL", "false").strip().lower() in {"1", "true", "yes", "on"}:
            logging.info("Training legacy supervised BTC direction model...")
            SupervisedTrainer().train()
            Evaluator().evaluate()
        else:
            logging.info("Skipping legacy btc_direction_model path; runtime scoring uses tp/return/stage1/stage2 artifacts.")

        _check_budget("pre_clob_fetch")

        # ── BUG FIX (BUG 3): Cap number of tokens fetched ──
        logging.info("Fetching token-level CLOB price history...")
        markets_df = pd.read_csv("logs/markets.csv", engine="python", on_bad_lines="skip") if HistoricalDatasetBuilder().logs_dir.joinpath("markets.csv").exists() else pd.DataFrame()
        token_ids = []
        def _normalize_token_id(raw):
            token = str(raw or "").strip().strip('"').strip("'")
            if not token:
                return None
            return token if re.fullmatch(r"\d{8,}", token) else None
        if not markets_df.empty:
            for col in ["yes_token_id", "no_token_id"]:
                if col in markets_df.columns:
                    for value in markets_df[col].dropna().tolist():
                        token = _normalize_token_id(value)
                        if token:
                            token_ids.append(token)
        token_ids = sorted(set(token_ids))

        # Cap at MAX_CLOB_TOKENS to prevent 15+ min fetch times
        if len(token_ids) > MAX_CLOB_TOKENS:
            logging.warning(
                "Capping CLOB token fetch from %d to %d tokens (set MAX_CLOB_TOKENS to change)",
                len(token_ids), MAX_CLOB_TOKENS,
            )
            token_ids = token_ids[:MAX_CLOB_TOKENS]

        if token_ids:
            CLOBHistoryClient().append_history(token_ids, days=MAX_CLOB_DAYS, interval="1m")
        _check_budget("clob_history")

        logging.info("Building contract-level labels and wallet alpha...")
        ContractTargetBuilder().write(forward_minutes=15, max_hold_minutes=60, tp_move=0.04, sl_move=0.03)
        _check_budget("contract_targets")

        WalletAlphaBuilder().write()
        _check_budget("wallet_alpha")

        targets_path = HistoricalDatasetBuilder().logs_dir / "contract_targets.csv"
        alpha_history_path = HistoricalDatasetBuilder().logs_dir / "wallet_alpha_history.csv"
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

        _check_budget("alpha_merge")

        SupervisedModels().train()
        _check_budget("supervised_models")

        Stage1Models().train()
        _check_budget("stage1_models")

        SequenceFeatureBuilder().write()
        _check_budget("sequence_features")

        Stage2TemporalModels().train()
        _check_budget("stage2_temporal")

        WalkForwardEvaluator().evaluate()
        TimeSplitTrainer().run()
        PathReplaySimulator().write()

        elapsed = time.time() - pipeline_start
        logging.info("Research pipeline complete in %.0fs.", elapsed)

    except PipelineTimeout as exc:
        logging.warning("Pipeline timeout: %s — partial artifacts are usable.", exc)
    finally:
        try:
            signal.alarm(0)
        except (AttributeError, ValueError):
            pass


if __name__ == "__main__":
    run_research_pipeline()
