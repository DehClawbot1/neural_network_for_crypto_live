import logging

from historical_dataset_builder import HistoricalDatasetBuilder
from target_builder import TargetBuilder
from dataset_aligner import DatasetAligner
from supervised_trainer import SupervisedTrainer
from evaluator import Evaluator
from contract_target_builder import ContractTargetBuilder
from wallet_alpha_builder import WalletAlphaBuilder
from walk_forward_evaluator import WalkForwardEvaluator
from time_split_trainer import TimeSplitTrainer
from path_replay_simulator import PathReplaySimulator

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

    logging.info("Building contract-level labels and wallet alpha...")
    ContractTargetBuilder().write(horizon_rows=5)
    WalletAlphaBuilder().write()
    WalkForwardEvaluator().evaluate()
    TimeSplitTrainer().run()
    PathReplaySimulator().write()

    logging.info("Research pipeline complete.")


if __name__ == "__main__":
    run_research_pipeline()
