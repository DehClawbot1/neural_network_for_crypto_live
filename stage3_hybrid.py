import pandas as pd


class Stage3HybridScorer:
    """
    Hybrid scorer combining tabular outputs, temporal outputs, and execution/risk penalties.
    Also provides a simple ensemble-agreement gate so tree-based and neural-style
    probabilities can both be required before a higher-confidence trade is allowed.
    """

    def __init__(self, transaction_cost=0.01, risk_penalty=0.5, agreement_threshold=0.65):
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty
        self.agreement_threshold = agreement_threshold

    def run(self, df: pd.DataFrame):
        if df is None or df.empty:
            return pd.DataFrame()

        out = df.copy()
        if "p_tp_before_sl" not in out.columns:
            out["p_tp_before_sl"] = 0.0
        if "expected_return" not in out.columns:
            out["expected_return"] = 0.0
        if "lower_confidence_bound" not in out.columns:
            out["lower_confidence_bound"] = 0.0
        if "temporal_expected_return" not in out.columns:
            out["temporal_expected_return"] = 0.0

        wallet_alpha = out.get("wallet_alpha_30d", 0.0)
        if not isinstance(wallet_alpha, pd.Series):
            wallet_alpha = pd.Series([0.0] * len(out), index=out.index)
        trade_size_quality = out.get("normalized_trade_size", 0.0)
        if not isinstance(trade_size_quality, pd.Series):
            trade_size_quality = pd.Series([0.0] * len(out), index=out.index)
        btc_regime_fit = 1.0 - out.get("btc_realized_vol_15m", 0.0).fillna(0.0).clip(lower=0.0, upper=1.0) if "btc_realized_vol_15m" in out.columns else pd.Series([1.0] * len(out), index=out.index)
        liquidity_filter = out.get("liquidity_score", 0.0)
        if not isinstance(liquidity_filter, pd.Series):
            liquidity_filter = pd.Series([0.0] * len(out), index=out.index)
        spread_penalty = out.get("spread", 0.0)
        if not isinstance(spread_penalty, pd.Series):
            spread_penalty = pd.Series([0.0] * len(out), index=out.index)
        time_penalty = out.get("time_decay_score", 0.0)
        if not isinstance(time_penalty, pd.Series):
            time_penalty = pd.Series([0.0] * len(out), index=out.index)
        crowding_penalty = out.get("wallet_same_market_history", 0.0)
        if not isinstance(crowding_penalty, pd.Series):
            crowding_penalty = pd.Series([0.0] * len(out), index=out.index)

        out["p_win"] = out["p_tp_before_sl"].astype(float)
        out["p_loss"] = 1.0 - out["p_win"]
        out["expected_payoff"] = out["expected_return"].astype(float).clip(lower=0.0)
        out["expected_loss"] = out["lower_confidence_bound"].astype(float).abs()
        out["temporal_boost"] = out["temporal_expected_return"].astype(float)
        out["supervised_edge"] = out["p_win"] * (out["expected_payoff"] + out["temporal_boost"].clip(lower=0.0) * 0.35)
        out["execution_quality_score"] = (
            wallet_alpha.astype(float).clip(lower=0.0)
            * (0.5 + trade_size_quality.astype(float).clip(lower=0.0))
            * (0.5 + liquidity_filter.astype(float).clip(lower=0.0))
            * btc_regime_fit.astype(float)
        )
        out["entry_ev"] = out["supervised_edge"] - self.transaction_cost - spread_penalty.astype(float)
        out["risk_adjusted_ev"] = out["entry_ev"] - out["p_loss"] * out["expected_loss"] * self.risk_penalty - time_penalty.astype(float) * 0.05 - crowding_penalty.astype(float) * 0.01
        out["hybrid_edge"] = out["risk_adjusted_ev"] * (1.0 + out["execution_quality_score"].clip(lower=0.0))

        if "temporal_p_tp_before_sl" in out.columns:
            out["rf_probability"] = out["p_tp_before_sl"].astype(float)
            out["nn_probability"] = out["temporal_p_tp_before_sl"].astype(float)
            out["ensemble_agreement"] = (
                (out["rf_probability"] >= self.agreement_threshold)
                & (out["nn_probability"] >= self.agreement_threshold)
            ).astype(int)
            out["ensemble_probability"] = (out["rf_probability"] + out["nn_probability"]) / 2.0
            out["ensemble_live_candidate"] = (
                (out["ensemble_agreement"] == 1)
                & (out["ensemble_probability"] >= self.agreement_threshold)
            ).astype(int)
        else:
            out["rf_probability"] = out["p_tp_before_sl"].astype(float)
            out["nn_probability"] = out.get("temporal_p_tp_before_sl", 0.5)
            if not isinstance(out["nn_probability"], pd.Series):
                out["nn_probability"] = pd.Series([float(out["nn_probability"])] * len(out), index=out.index)
            out["nn_probability"] = out["nn_probability"].astype(float)
            out["ensemble_agreement"] = 0
            out["ensemble_probability"] = out["rf_probability"]
            out["ensemble_live_candidate"] = 0
        return out

