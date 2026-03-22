import unittest

import supervisor


class TestSupervisorSmoke(unittest.TestCase):
    def test_helpers_exist(self):
        self.assertTrue(hasattr(supervisor, "prepare_observation"))
        self.assertTrue(hasattr(supervisor, "prepare_position_observation"))
        self.assertTrue(hasattr(supervisor, "quote_entry_price"))
        self.assertTrue(hasattr(supervisor, "quote_exit_price"))


if __name__ == "__main__":
    unittest.main()
