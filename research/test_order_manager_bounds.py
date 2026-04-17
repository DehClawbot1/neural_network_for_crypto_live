import os
from order_manager import OrderManager

os.environ["EXEC_STATIC_SPREAD_LIMIT"] = "0.10"
os.environ["EXEC_STATIC_SLIPPAGE_LIMIT"] = "0.03"
os.environ["POLY_PRICE_MIN"] = "0.01"
os.environ["POLY_PRICE_MAX"] = "0.99"

class MockClient:
    def get_orderbook(self, token_id):
        # Fake 15% spread
        return {"bids": [{"price": "0.40", "size": "100"}], "asks": [{"price": "0.55", "size": "100"}]}
    def get_price(self, token_id, side):
        return 0.40 if side == "SELL" else 0.55
    def get_spread(self, token_id):
        return 0.15 # 15% spread
    
    def get_balance_allowance(self, asset_type=None, token_id=None):
        return {"balance": "1000"}

class MockRisk:
    def pre_trade_check(self, *args, **kwargs):
        class Decision:
            allowed = True
        return Decision()
        
manager = OrderManager()
manager.client = MockClient()
manager.risk = MockRisk()

print("Test 1: Price out of bounds (1.05)")
row, res = manager.submit_entry(token_id="0xTEST", price=1.05, size=10, side="BUY")
print("Status:", row["status"], "- Reason:", row["reason"])

print("\nTest 2: Spread exceeds limit (0.15 > 0.10)")
row, res = manager.submit_entry(token_id="0xTEST", price=0.55, size=10, side="BUY")
print("Status:", row["status"], "- Reason:", row["reason"])

print("\nTest 3: Slippage exceeds limit (price=0.50, quoted=0.55)")
manager.client.get_spread = lambda t: 0.05
row, res = manager.submit_entry(token_id="0xTEST", price=0.50, size=10, side="BUY")
print("Status:", row["status"], "- Reason:", row["reason"])

print("\nTest 4: Orderbook fail closed")
def bad_book(*args):
    raise ConnectionError("Timeout")
manager.client.get_orderbook = bad_book
row, res = manager.submit_entry(token_id="0xTEST", price=0.55, size=10, side="BUY")
print("Status:", row["status"], "- Reason:", row["reason"])
