import unittest
import warnings

import numpy as np
import pandas as pd

import supervisor


class TestSupervisorSmoke(unittest.TestCase):
    def test_helpers_exist(self):
        self.assertTrue(hasattr(supervisor, "prepare_observation"))
        self.assertTrue(hasattr(supervisor, "prepare_position_observation"))
        self.assertTrue(hasattr(supervisor, "quote_entry_price"))
        self.assertTrue(hasattr(supervisor, "quote_exit_price"))

    def test_choose_cycle_sleep_interval_prefers_active_positions(self):
        sleep_seconds, reason = supervisor.choose_cycle_sleep_interval(
            open_positions_count=2,
            entry_freeze_active=False,
            active_position_poll_seconds=5.0,
            idle_poll_seconds=15.0,
            entry_freeze_poll_seconds=10.0,
        )
        self.assertEqual(sleep_seconds, 5.0)
        self.assertEqual(reason, "fast-polling active trades")

    def test_choose_cycle_sleep_interval_prefers_freeze_recheck_over_idle(self):
        sleep_seconds, reason = supervisor.choose_cycle_sleep_interval(
            open_positions_count=0,
            entry_freeze_active=True,
            active_position_poll_seconds=5.0,
            idle_poll_seconds=15.0,
            entry_freeze_poll_seconds=10.0,
        )
        self.assertEqual(sleep_seconds, 10.0)
        self.assertEqual(reason, "entry freeze active; rechecking soon")

    def test_performance_governor_top_signal_allows_candidates_until_slot_consumed(self):
        allow_first = supervisor.performance_governor_top_signal_decision(
            {"top_signal_only": True},
            consumed_count=0,
        )
        allow_second = supervisor.performance_governor_top_signal_decision(
            {"top_signal_only": True},
            consumed_count=0,
        )
        consumed_count = supervisor.performance_governor_consume_top_signal_slot(
            {"top_signal_only": True},
            consumed_count=0,
        )
        allow_third = supervisor.performance_governor_top_signal_decision(
            {"top_signal_only": True},
            consumed_count=consumed_count,
        )

        self.assertTrue(allow_first)
        self.assertTrue(allow_second)
        self.assertEqual(consumed_count, 1)
        self.assertFalse(allow_third)

    def test_fill_missing_series_values_avoids_fillna_future_warning(self):
        series = pd.Series([None, 1.0, np.nan], dtype=object)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            filled = supervisor._fill_missing_series_values(series, 0.5)
        self.assertEqual(filled.tolist(), [0.5, 1.0, 0.5])

    def test_top_rank_summary_prefers_decision_score_and_outcome_side_fallback(self):
        ranked_row = pd.Series(
            {
                "signal_label": "STRONG WEATHER OPPORTUNITY",
                "decision_score": 0.82,
                "confidence": 0.35,
                "market_title": "Weather Test",
                "outcome_side": "YES",
                "side": np.nan,
            }
        )
        ranked_side = (
            ranked_row.get("side")
            if pd.notna(ranked_row.get("side"))
            else ranked_row.get("outcome_side", ranked_row.get("side"))
        )
        summary_line = (
            f"1. {ranked_row.get('signal_label')} | "
            f"confidence={supervisor._safe_float(ranked_row.get('decision_score', ranked_row.get('confidence', supervisor.PredictionLayer.select_signal_score(ranked_row))), default=0.0):.2f} | "
            f"market={ranked_row.get('market_title')} | "
            f"side={ranked_side}"
        )
        self.assertIn("confidence=0.82", summary_line)
        self.assertIn("side=YES", summary_line)


if __name__ == "__main__":
    unittest.main()
