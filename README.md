# Balance Fix + Market Orders + Money Management

## The Problem

Your bot keeps saying `insufficient_funds` even though you have money on Polymarket.

## Root Cause

The Polymarket CLOB API returns balance in **microdollars** (1,000,000 = $1.00).

The tutorial code shows the correct way:
```python
balance = auth_client.get_balance_allowance(...)
usdc_balance = int(balance['balance']) / 1e6  # ← This division was missing!
```

But `order_manager.py` was reading the raw value without dividing:
```python
available_balance = float(readiness.get("balance", 0.0))  # BUG: treats microdollars as dollars
```

So if you had $5.00, the API returned `5000000`, and the bot compared `5000000 >= 10` (which passes) but then the actual CLOB order creation internally also expected normalized amounts, causing a mismatch.

## What's Fixed

### 1. Balance Normalization (`order_manager.py`)
- Added `_normalize_balance()` method that divides by 1e6 when the raw value looks like microdollars
- Added `_get_available_balance()` that logs both raw and normalized values for debugging
- All balance checks now use normalized values

### 2. Market Orders (`order_manager.py`)
- New `submit_market_entry()` method for Fill-or-Kill (FOK) market orders
- Matches the tutorial pattern exactly:
  ```python
  market_order = MarketOrderArgs(token_id=yes_token_id, amount=5.0, side=BUY)
  signed = auth_client.create_market_order(market_order)
  response = auth_client.post_order(signed, OrderType.FOK)
  ```
- Better for fast-moving Bitcoin 5-min markets

### 3. Money Management (`money_manager.py`)
- Bets are now sized as % of balance, not fixed $10/$50
- High confidence (>70%): 5% of balance
- Medium confidence (50-70%): 2% of balance
- Low confidence (<50%): 1% of balance
- Reduces bet size after consecutive losses
- Never exceeds 25% total exposure across all positions
- Min bet: $0.50, Max bet: $20.00

### 4. Supervisor Wiring (`supervisor.py`)
- Entry path uses market orders when `USE_MARKET_ORDERS=True` (default)
- Bet sizing uses MoneyManager instead of fixed amounts
- Better logging of balance and bet decisions

## How to Apply

```bash
# 1. Copy all files to your project root
cp fixes/*.py /path/to/your/project/

# 2. Run the apply script
cd /path/to/your/project
python apply_all_betting_fixes.py

# 3. Verify balance reads correctly
python diagnose_balance_fix.py

# 4. Start trading
python run_bot.py
```

## Files

| File | Description |
|------|-------------|
| `order_manager.py` | PATCHED: Balance normalization + market order support |
| `config.py` | PATCHED: Money management settings |
| `money_manager.py` | NEW: Intelligent bet sizing |
| `supervisor_betting_patch.py` | NEW: Patches supervisor for market orders |
| `diagnose_balance_fix.py` | NEW: Diagnostic script to verify balance |
| `apply_all_betting_fixes.py` | Apply script that patches supervisor.py + run_bot.py |

## Tuning

Edit `config.py` to adjust money management:

```python
MAX_RISK_PER_TRADE_PCT = 0.05  # 5% per trade (change to 0.10 for 10%)
MIN_BET_USDC = 0.50            # Minimum bet
MAX_BET_USDC = 20.0            # Maximum bet
USE_MARKET_ORDERS = True       # Set False to use limit orders instead
MAX_TOTAL_EXPOSURE_PCT = 0.25  # Max 25% of balance in open positions
```
