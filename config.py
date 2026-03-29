class TradingConfig:
    # Shadow Trading Targets (absolute price moves on 0-1 token scale)
    SHADOW_TP_DELTA = 0.04
    SHADOW_SL_DELTA = 0.03
    SHADOW_WINDOW_MINUTES = 60

    # Paper Trading ROI Targets
    PAPER_TP_ROI = 0.08
    PAPER_TRAILING_STOP = 0.08

    # ── FIX: Lowered from 0.85 to 0.55
    MIN_CONVICTION_FOR_READY = 0.55

    # Audit Probability Buckets for Calibration
    PROB_BUCKETS = [0.0, 0.70, 0.85, 1.0]

    # Thresholds
    VETO_EV_THRESHOLD = 0.005
    CALIBRATION_BIAS_THRESHOLD = 20.0

    # ── NEW: Retrain after this many closed trades
    RETRAIN_AFTER_TRADES = 5

    # ── NEW: Money Management Config ──
    # Maximum % of available balance to risk per single trade
    MAX_RISK_PER_TRADE_PCT = 0.05  # 5% of balance per trade

    # Minimum bet size in USDC (below this, skip the trade)
    MIN_BET_USDC = 1.00  # Polymarket CLOB minimum order size is $1

    # Maximum bet size in USDC (cap regardless of balance)
    MAX_BET_USDC = 5.0  # $20 max per trade

    # Default bet for high confidence (>0.70) as % of balance
    HIGH_CONFIDENCE_BET_PCT = 0.08  # 5%

    # Default bet for medium confidence (0.50-0.70) as % of balance
    MEDIUM_CONFIDENCE_BET_PCT = 0.05  # 5% (was 2%, raised so bets reach $1 minimum)

    # Default bet for low confidence (<0.50) as % of balance
    LOW_CONFIDENCE_BET_PCT = 0.04  # 4% (was 1%, raised so bets reach $1 minimum)

    # Use market orders (FOK) instead of limit orders for live trading
    USE_MARKET_ORDERS = True

    # Maximum number of concurrent open positions
    MAX_CONCURRENT_POSITIONS = 4

    # Time stop: close positions after this many minutes
    TIME_STOP_MINUTES = 120

    # Maximum total exposure as % of balance
    MAX_TOTAL_EXPOSURE_PCT = 0.25  # 25% of balance across all positions

    # Whether the CLOB API returns balance in microdollars (raw integer / 1e6 = dollars)
    # Set to False if your py-clob-client version already normalizes to dollars
    BALANCE_IS_MICRODOLLARS = True

    # Paper trading simulated balance
    SIMULATED_STARTING_BALANCE = 1000.0
