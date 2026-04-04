from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from btc_forecast_eval import BTCForecastEvaluator
from btc_forecast_model import BTCForecastModel
from btc_multitimeframe import BTCMultiTimeframeForecaster
from supervisor_btc_pipeline import apply_btc_pipeline
from technical_analyzer import TechnicalAnalyzer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a safe one-cycle supervisor BTC smoke without any order submission."
    )
    parser.add_argument(
        "--logs-dir",
        default="logs_supervisor_btc_smoke",
        help="Directory for smoke artifacts and BTC eval logs.",
    )
    parser.add_argument(
        "--eval-horizon-seconds",
        type=int,
        default=0,
        help="Evaluator horizon used by the smoke run. Defaults to 0 for immediate maturity checks.",
    )
    parser.add_argument(
        "--seed-pending-eval",
        action="store_true",
        help="Seed one synthetic pending forecast so the cycle also exercises the eval write path.",
    )
    return parser


def _maybe_seed_pending_eval(evaluator: BTCForecastEvaluator, price_hint: float) -> bool:
    if price_hint <= 0:
        return False
    evaluator.record_prediction(
        {
            "btc_forecast_ready": True,
            "btc_predicted_direction": 1,
            "btc_predicted_return_15": 0.0,
            "btc_forecast_confidence": 0.5,
            "btc_mtf_agreement": 0.0,
            "btc_mtf_n_agree": 0,
            "btc_mtf_n_total": 0,
            "btc_mtf_source": "smoke_seed",
        },
        current_price=price_hint,
        source="smoke_seed",
    )
    return True


def run_smoke(logs_dir: str, eval_horizon_seconds: int, seed_pending_eval: bool) -> dict:
    smoke_dir = Path(logs_dir)
    smoke_dir.mkdir(parents=True, exist_ok=True)

    technical_analyzer = TechnicalAnalyzer()
    btc_forecast_model = BTCForecastModel()
    btc_mtf_forecaster = BTCMultiTimeframeForecaster()
    btc_forecast_evaluator = BTCForecastEvaluator(
        logs_dir=str(smoke_dir),
        horizon_seconds=eval_horizon_seconds,
    )

    macro_context = {}
    try:
        macro_context.update(technical_analyzer.analyze() or {})
    except Exception as exc:
        logger.warning("Technical analyzer base context failed during smoke: %s", exc)

    seed_price_hint = float(macro_context.get("btc_live_price") or 0.0)
    seeded = False
    if seed_pending_eval:
        seeded = _maybe_seed_pending_eval(btc_forecast_evaluator, seed_price_hint)

    macro_context, pipeline_audit = apply_btc_pipeline(
        macro_context,
        technical_analyzer,
        btc_mtf_forecaster,
        btc_forecast_model,
        btc_forecast_evaluator,
    )

    eval_summary = btc_forecast_evaluator.summary()
    selected_macro_snapshot = {
        key: macro_context.get(key)
        for key in [
            "btc_forecast_ready",
            "btc_predicted_direction",
            "btc_predicted_return_15",
            "btc_forecast_confidence",
            "btc_mtf_agreement",
            "btc_mtf_n_agree",
            "btc_mtf_n_total",
            "btc_mtf_source",
            "btc_live_price",
            "ob_ready",
            "ob_midpoint",
            "reddit_sentiment",
            "fgi_value",
        ]
    }
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "logs_dir": str(smoke_dir.resolve()),
        "order_submission_allowed": False,
        "cycle_count": 1,
        "seed_pending_eval_requested": seed_pending_eval,
        "seed_pending_eval_applied": seeded,
        "eval_horizon_seconds": eval_horizon_seconds,
        "pipeline_audit": pipeline_audit,
        "eval_summary": eval_summary,
        "selected_macro_snapshot": selected_macro_snapshot,
        "macro_context_keys": sorted(macro_context.keys()),
        "artifacts": {
            "audit_json": str((smoke_dir / "supervisor_btc_smoke_audit.json").resolve()),
            "eval_csv": str((smoke_dir / "btc_forecast_eval.csv").resolve()),
            "pending_csv": str((smoke_dir / "btc_forecast_eval_pending.csv").resolve()),
        },
    }

    audit_path = smoke_dir / "supervisor_btc_smoke_audit.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    args = _build_parser().parse_args()
    result = run_smoke(
        logs_dir=args.logs_dir,
        eval_horizon_seconds=args.eval_horizon_seconds,
        seed_pending_eval=args.seed_pending_eval,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
