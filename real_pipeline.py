import logging
import os

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


def run_research_pipeline():
    logging.info("Building historical dataset...")
    HistoricalDatasetBuilder().write()

    logging.info("Building BTC direction targets...")
    TargetBuilder().write(days=30, horizon_minutes=60)
    DatasetAligner().write()

    if os.getenv("ENABLE_LEGACY_BTC_DIRECTION_MODEL", "false").strip().lower() in {"1", "true", "yes", "on"}:
        logging.info("Training legacy supervised BTC direction model...")
        SupervisedTrainer().train()
        Evaluator().evaluate()
    else:
        logging.info("Skipping legacy btc_direction_model path; runtime scoring uses tp/return/stage1/stage2 artifacts.")

    logging.info("Fetching token-level CLOB price history...")
    markets_df = pd.read_csv("logs/markets.csv", engine="python", on_bad_lines="skip") if HistoricalDatasetBuilder().logs_dir.joinpath("markets.csv").exists() else pd.DataFrame()
    token_ids = []
    if not markets_df.empty:
        for col in ["yes_token_id", "no_token_id"]:
            if col in markets_df.columns:
                token_ids.extend([str(x) for x in markets_df[col].dropna().tolist() if str(x)])
    if token_ids:
        CLOBHistoryClient().append_history(sorted(set(token_ids)), days=7, interval="1m")

    logging.info("Building contract-level labels and wallet alpha...")
    ContractTargetBuilder().write(forward_minutes=15, max_hold_minutes=60, tp_move=0.04, sl_move=0.03)
    WalletAlphaBuilder().write()

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

    SupervisedModels().train()
    Stage1Models().train()
    SequenceFeatureBuilder().write()
    Stage2TemporalModels().train()
    WalkForwardEvaluator().evaluate()
    TimeSplitTrainer().run()
    PathReplaySimulator().write()

    logging.info("Research pipeline complete.")


if __name__ == "__main__":
    run_research_pipeline()

