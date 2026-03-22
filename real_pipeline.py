import logging

from historical_dataset_builder import HistoricalDatasetBuilder
from target_builder import TargetBuilder
from dataset_aligner import DatasetAligner
from supervised_trainer import SupervisedTrainer
from evaluator import Evaluator
from supervised_models import SupervisedModels
from stage1_models import Stage1Models
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

    logging.info("Training supervised BTC direction model...")
    SupervisedTrainer().train()
    Evaluator().evaluate()

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
    SupervisedModels().train()
    Stage1Models().train()
    WalkForwardEvaluator().evaluate()
    TimeSplitTrainer().run()
    PathReplaySimulator().write()

    logging.info("Research pipeline complete.")


if __name__ == "__main__":
    run_research_pipeline()
