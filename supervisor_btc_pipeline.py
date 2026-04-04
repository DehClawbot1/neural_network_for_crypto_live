from __future__ import annotations

import logging
from typing import Callable


logger = logging.getLogger(__name__)


def _default_sentiment_fetcher() -> dict:
    from btc_sentiment_features import BTCSentimentFeatures

    return BTCSentimentFeatures().fetch_current_snapshot()


def _default_orderbook_fetcher() -> dict:
    from orderbook_depth_features import fetch_btc_depth_snapshot

    return fetch_btc_depth_snapshot()


def apply_btc_pipeline(
    macro_context: dict,
    technical_analyzer,
    btc_mtf_forecaster,
    btc_forecast_model,
    btc_forecast_evaluator=None,
    *,
    sentiment_fetcher: Callable[[], dict] | None = None,
    orderbook_fetcher: Callable[[], dict] | None = None,
) -> tuple[dict, dict]:
    """
    Apply the supervisor BTC forecast/sentiment/orderbook/eval pipeline to
    an existing macro context and return an audit summary.
    """

    working_context = dict(macro_context or {})
    audit = {
        "forecast_ready": False,
        "forecast_source": None,
        "forecast_mode": None,
        "forecast_error": None,
        "sentiment_added": False,
        "sentiment_error": None,
        "orderbook_ready": False,
        "orderbook_error": None,
        "btc_live_price": None,
        "eval_enabled": btc_forecast_evaluator is not None,
        "eval_error": None,
        "eval_evaluated_count": 0,
        "eval_recorded": False,
        "pending_before_eval": None,
        "pending_after_eval": None,
        "pending_after_record": None,
    }

    try:
        cds = technical_analyzer.candle_data_service
        btc_fc = {}
        if getattr(btc_mtf_forecaster, "is_ready", False):
            candle_dfs = {}
            for tf_interval, tf_limit in [("15m", 250), ("1h", 250), ("4h", 250)]:
                try:
                    candle_dfs[tf_interval] = cds.refresh_latest_closed_candles(
                        tf_interval, limit=tf_limit, timezone_name="UTC"
                    )
                except Exception:
                    pass
            btc_fc = btc_mtf_forecaster.predict(candle_dfs) or {}
            audit["forecast_mode"] = "multi_timeframe"
        elif getattr(btc_forecast_model, "is_ready", False):
            btc_fc = btc_forecast_model.predict_from_candles(
                cds.refresh_latest_closed_candles("15m", limit=250, timezone_name="UTC")
            ) or {}
            audit["forecast_mode"] = "single_timeframe"
        if btc_fc:
            working_context.update(btc_fc)
        audit["forecast_ready"] = bool(btc_fc.get("btc_forecast_ready"))
        audit["forecast_source"] = btc_fc.get("btc_mtf_source") or audit["forecast_mode"]
    except Exception as exc:
        audit["forecast_error"] = str(exc)
        logger.debug("BTC forecast skipped: %s", exc)

    try:
        if sentiment_fetcher is None:
            sentiment_fetcher = _default_sentiment_fetcher
        btc_sentiment_snapshot = sentiment_fetcher() or {}
        if btc_sentiment_snapshot:
            working_context.update(btc_sentiment_snapshot)
            audit["sentiment_added"] = True
    except Exception as exc:
        audit["sentiment_error"] = str(exc)
        logger.debug("BTC sentiment skipped: %s", exc)

    try:
        if orderbook_fetcher is None:
            orderbook_fetcher = _default_orderbook_fetcher
        ob_features = orderbook_fetcher() or {}
        if ob_features.get("ob_ready"):
            working_context.update(ob_features)
            audit["orderbook_ready"] = True
    except Exception as exc:
        audit["orderbook_error"] = str(exc)
        logger.debug("BTC order book depth skipped: %s", exc)

    try:
        btc_live_price = (
            working_context.get("btc_live_price")
            or working_context.get("ob_midpoint")
            or 0
        )
        audit["btc_live_price"] = float(btc_live_price or 0.0)
        if btc_forecast_evaluator is not None and btc_live_price > 0:
            audit["pending_before_eval"] = btc_forecast_evaluator.summary().get(
                "pending_predictions", 0
            )
            matured = btc_forecast_evaluator.evaluate_matured(btc_live_price)
            audit["eval_evaluated_count"] = len(matured or [])
            audit["pending_after_eval"] = btc_forecast_evaluator.summary().get(
                "pending_predictions", 0
            )
            btc_fc_for_eval = {
                key: value for key, value in working_context.items() if key.startswith("btc_")
            }
            btc_forecast_evaluator.record_prediction(
                btc_fc_for_eval,
                btc_live_price,
                source=working_context.get("btc_mtf_source", "unknown"),
            )
            audit["pending_after_record"] = btc_forecast_evaluator.summary().get(
                "pending_predictions", 0
            )
            audit["eval_recorded"] = (
                audit["pending_after_record"] is not None
                and audit["pending_after_eval"] is not None
                and audit["pending_after_record"] > audit["pending_after_eval"]
            )
    except Exception as exc:
        audit["eval_error"] = str(exc)
        logger.debug("BTC forecast eval skipped: %s", exc)

    return working_context, audit
