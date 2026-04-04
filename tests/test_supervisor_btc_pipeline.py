import unittest

from supervisor_btc_pipeline import apply_btc_pipeline


class _FakeCandleService:
    def refresh_latest_closed_candles(self, interval, limit=250, timezone_name="UTC"):
        return {"interval": interval, "limit": limit, "timezone_name": timezone_name}


class _FakeTechnicalAnalyzer:
    def __init__(self):
        self.candle_data_service = _FakeCandleService()


class _FakeMTFForecaster:
    is_ready = True

    def predict(self, candle_dfs):
        return {
            "btc_forecast_ready": True,
            "btc_predicted_direction": 1,
            "btc_predicted_return_15": 0.012,
            "btc_forecast_confidence": 0.61,
            "btc_mtf_source": "mtf_vote",
            "btc_live_price": 70250.0,
        }


class _FakeSingleForecastModel:
    is_ready = False


class _FakeEvaluator:
    def __init__(self):
        self.pending_predictions = 0
        self.record_calls = []

    def summary(self):
        return {"pending_predictions": self.pending_predictions}

    def evaluate_matured(self, current_price):
        self.pending_predictions = max(0, self.pending_predictions - 1)
        return [{"current_price": current_price}]

    def record_prediction(self, prediction, current_price, source="unknown"):
        if prediction.get("btc_forecast_ready"):
            self.pending_predictions += 1
            self.record_calls.append(
                {
                    "prediction": prediction,
                    "current_price": current_price,
                    "source": source,
                }
            )


class TestSupervisorBTCPipeline(unittest.TestCase):
    def test_apply_btc_pipeline_records_forecast_and_eval_audit(self):
        evaluator = _FakeEvaluator()
        evaluator.pending_predictions = 1

        macro_context, audit = apply_btc_pipeline(
            {},
            _FakeTechnicalAnalyzer(),
            _FakeMTFForecaster(),
            _FakeSingleForecastModel(),
            evaluator,
            sentiment_fetcher=lambda: {"btc_sentiment_score": 0.25},
            orderbook_fetcher=lambda: {"ob_ready": True, "ob_midpoint": 70255.0},
        )

        self.assertTrue(audit["forecast_ready"])
        self.assertEqual(audit["forecast_mode"], "multi_timeframe")
        self.assertTrue(audit["sentiment_added"])
        self.assertTrue(audit["orderbook_ready"])
        self.assertEqual(audit["eval_evaluated_count"], 1)
        self.assertTrue(audit["eval_recorded"])
        self.assertEqual(audit["pending_before_eval"], 1)
        self.assertEqual(audit["pending_after_eval"], 0)
        self.assertEqual(audit["pending_after_record"], 1)
        self.assertEqual(macro_context["btc_mtf_source"], "mtf_vote")
        self.assertEqual(len(evaluator.record_calls), 1)


if __name__ == "__main__":
    unittest.main()
