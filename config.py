class TradingConfig:
    # Shadow Trading Targets (absolute price moves on 0-1 token scale)
    SHADOW_TP_DELTA = 0.04
    SHADOW_SL_DELTA = 0.03
    SHADOW_WINDOW_MINUTES = 60

    # Paper Trading ROI Targets
    PAPER_TP_ROI = 0.25
    PAPER_TRAILING_STOP = 0.08

    # ── FIX: Lowered from 0.85 to 0.55
    #    0.85 was blocking every signal; new models need trades to learn from
    MIN_CONVICTION_FOR_READY = 0.55

    # Audit Probability Buckets for Calibration
    PROB_BUCKETS = [0.0, 0.70, 0.85, 1.0]

    # Thresholds
    VETO_EV_THRESHOLD = 0.005
    CALIBRATION_BIAS_THRESHOLD = 20.0

    # ── NEW: Retrain after this many closed trades
    RETRAIN_AFTER_TRADES = 5
