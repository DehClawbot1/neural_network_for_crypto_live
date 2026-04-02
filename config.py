class TradingConfig:
    # Shadow trading / rule-exit targets on the 0..1 token price scale
    SHADOW_TP_DELTA = 0.04
    SHADOW_SL_DELTA = 0.03
    SHADOW_WINDOW_MINUTES = 60

    # Paper / rule-management ROI targets
    PAPER_TP_ROI = 0.08
    PAPER_TRAILING_STOP = 0.08

    # Minimum model conviction before the system should even consider action
    MIN_CONVICTION_FOR_READY = 0.55

    # Trade cadence targeting
    TARGET_ENTRY_INTERVAL_MINUTES = 5
    ENTRY_AGGRESSION_TOP_K = 3
    ENTRY_INACTIVITY_SCORE_RELAX = 0.08
    ENTRY_INACTIVITY_SPREAD_RELAX = 0.10
    ENTRY_INACTIVITY_LIQUIDITY_RELAX_FACTOR = 0.35
    ENTRY_INACTIVITY_CONFIDENCE_BOOST = 0.08
    ENTRY_INACTIVITY_EXPECTED_RETURN_FLOOR = -0.002

    # Audit probability buckets for calibration dashboards
    PROB_BUCKETS = [0.0, 0.55, 0.70, 0.85, 1.0]

    # Thresholds
    VETO_EV_THRESHOLD = 0.005
    CALIBRATION_BIAS_THRESHOLD = 20.0

    # Retrain cadence
    RETRAIN_AFTER_TRADES = 5

    # Risk sizing
    MAX_RISK_PER_TRADE_PCT = 0.15
    MIN_BET_USDC = 1.00
    # Strategic anti-churn floor: bot should avoid opening tiny $1 micro trades by default.
    MIN_ENTRY_USDC = 3.00
    # If a partial reduce would create a tiny executed leg or a tiny leftover bag, prefer a full exit.
    MIN_REDUCE_NOTIONAL_USDC = 2.50
    MIN_POSITION_REMAINDER_USDC = 2.50
    # Legacy cap retained for compatibility; dynamic sizing uses HARD_MAX_BET_USDC.
    MAX_BET_USDC = 25.0
    # Hard per-trade fail-safe cap (absolute USD) to prevent runaway bet sizes.
    HARD_MAX_BET_USDC = 250.0
    # Keep part of wallet untouched so bot cannot deploy 100% capital.
    CAPITAL_RESERVE_PCT = 0.05  # Reduced from 0.20 to allow more trades
    # Minimum dynamic sizing component (as % of tradable balance), capped below.
    MIN_BET_DYNAMIC_PCT = 0.002
    MAX_DYNAMIC_MIN_BET_USDC = 5.0
    HIGH_CONFIDENCE_BET_PCT = 0.15 # Increased
    MEDIUM_CONFIDENCE_BET_PCT = 0.10 # Increased
    LOW_CONFIDENCE_BET_PCT = 0.04

    # Execution mode
    USE_MARKET_ORDERS = True

    # User requested limit: manage up to 5 simultaneous positions independently
    MAX_CONCURRENT_POSITIONS = 20

    # Set to 5 per requirements
    TIME_STOP_MINUTES = 120
    MAX_TOTAL_EXPOSURE_PCT = 0.95

    # py-clob-client balance compatibility flag
    BALANCE_IS_MICRODOLLARS = True

    # Paper trading fallback balance
    SIMULATED_STARTING_BALANCE = 1000.0
