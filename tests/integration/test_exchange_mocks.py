import pytest
import time

class MockExecutionClient:
    def __init__(self, failure_mode=None):
        self.failure_mode = failure_mode
        self.orders = {}
        
    def get_orderbook(self, token_id):
        if self.failure_mode == "stale_book":
            time.sleep(2) # Simulates heavy lag tripping Operational Lockout
            return {"bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.90", "size": "10"}], "timestamp": "stale"}
        return {"bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.55", "size": "100"}]}
    
    def post_order(self, payload):
        if self.failure_mode == "wrong_balance":
            raise ValueError("Insufficient balance")
        
        order_id = f"mock_{len(self.orders)}"
        self.orders[order_id] = {
            "status": "SUBMITTED",
            "filled_size": 0,
            "target_size": payload.get("size")
        }
        return {"orderID": order_id}

    def get_order(self, order_id):
        if order_id not in self.orders:
            raise ValueError("Order not found")
            
        if self.failure_mode == "partial_fill":
            self.orders[order_id]["filled_size"] = float(self.orders[order_id]["target_size"]) * 0.5
            self.orders[order_id]["status"] = "PARTIALLY_FILLED"
            
        elif self.failure_mode == "cancel_race":
            self.orders[order_id]["status"] = "FILLED"
            
        return self.orders[order_id]

    def cancel_order(self, order_id):
        if self.failure_mode == "cancel_race":
            raise ValueError("Order already filled, cannot cancel")
        self.orders[order_id]["status"] = "CANCELED"
        return {"status": "success"}

def test_partial_fill_mock():
    client = MockExecutionClient(failure_mode="partial_fill")
    res = client.post_order({"size": 100})
    order_id = res["orderID"]
    status_response = client.get_order(order_id)
    
    assert status_response["status"] == "PARTIALLY_FILLED"
    assert status_response["filled_size"] == 50.0

def test_cancel_race_condition():
    client = MockExecutionClient(failure_mode="cancel_race")
    res = client.post_order({"size": 100})
    order_id = res["orderID"]
    
    with pytest.raises(ValueError, match="already filled"):
        client.cancel_order(order_id)
    
    status = client.get_order(order_id)
    assert status["status"] == "FILLED"

def test_stale_book_latency():
    client = MockExecutionClient(failure_mode="stale_book")
    start = time.time()
    book = client.get_orderbook("token")
    elapsed = time.time() - start
    
    assert elapsed >= 2.0
    assert book["timestamp"] == "stale"
