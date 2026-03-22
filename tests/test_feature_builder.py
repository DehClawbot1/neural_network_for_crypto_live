import unittest

import pandas as pd

from feature_builder import FeatureBuilder


class TestFeatureBuilder(unittest.TestCase):
    def test_build_features_smoke(self):
        signals = pd.DataFrame([
            {
                "timestamp": "2026-03-22T00:00:00Z",
                "market_title": "BTC Up or Down",
                "trader_wallet": "0xabc",
                "size": 50,
                "outcome_side": "YES",
                "order_side": "BUY",
                "price": 0.4,
            }
        ])
        markets = pd.DataFrame([
            {
                "question": "BTC Up or Down",
                "liquidity": 10000,
                "volume": 5000,
                "last_trade_price": 0.41,
                "best_bid": 0.40,
                "best_ask": 0.42,
                "end_date": "2026-03-29T00:00:00Z",
                "slug": "btc-up-down",
            }
        ])
        df = FeatureBuilder().build_features(signals, markets)
        self.assertFalse(df.empty)
        self.assertIn("outcome_side", df.columns)
        self.assertIn("order_side", df.columns)


if __name__ == "__main__":
    unittest.main()
