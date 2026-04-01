import argparse

from trade_feedback_learner import TradeFeedbackLearner
from trade_manager import TradeManager


def main():
    parser = argparse.ArgumentParser(description="Backfill closed trade lifecycle into DB positions and feedback artifacts.")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--include-reconciliation-feedback", action="store_true")
    args = parser.parse_args()

    manager = TradeManager(logs_dir=args.logs_dir)
    db_result = manager.backfill_closed_positions_db_from_csv()

    learner = TradeFeedbackLearner(logs_dir=args.logs_dir)
    feedback_result = learner.backfill_from_closed_positions_csv(
        include_reconciliation=args.include_reconciliation_feedback
    )

    print(
        {
            "closed_csv_rows": int(db_result.get("csv_rows", 0)),
            "db_rows_upserted": int(db_result.get("db_rows_upserted", 0)),
            "feedback_reports_processed": int(feedback_result.get("processed_reports", 0)),
        }
    )


if __name__ == "__main__":
    main()
