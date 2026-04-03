import unittest

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


if __name__ == "__main__":
    unittest.main()
