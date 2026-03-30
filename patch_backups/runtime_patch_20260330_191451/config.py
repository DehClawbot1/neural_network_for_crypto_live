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
    MAX_BET_USDC = 25.0
    HIGH_CONFIDENCE_BET_PCT = 0.08
    MEDIUM_CONFIDENCE_BET_PCT = 0.05
    LOW_CONFIDENCE_BET_PCT = 0.04

    # Execution mode
    USE_MARKET_ORDERS = True

    # User requested limit: manage up to 5 simultaneous positions independently
    MAX_CONCURRENT_POSITIONS = 5

    # Rule exits
    TIME_STOP_MINUTES = 120
    MAX_TOTAL_EXPOSURE_PCT = 0.85

    # py-clob-client balance compatibility flag
    BALANCE_IS_MICRODOLLARS = True

    # Paper trading fallback balance
    SIMULATED_STARTING_BALANCE = 1000.0
