import pandas as pd


class Stage3HybridScorer:
    """
    Hybrid scorer combining tabular outputs, temporal outputs, and execution/risk penalties.
    """

    def __init__(self, transaction_cost=0.01, risk_penalty=0.5):
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty

    def run(self, df: pd.DataFrame):
        if df is None or df.empty:
            return pd.DataFrame()

        out = df.copy()
        out["p_win"] = out.get("p_tp_before_sl", 0.0).astype(float)
        out["p_loss"] = 1.0 - out["p_win"]
        out["expected_payoff"] = out.get("expected_return", 0.0).astype(float).clip(lower=0.0)
        out["expected_loss"] = out.get("lower_confidence_bound", 0.0).astype(float).abs()
        out["temporal_boost"] = out.get("temporal_expected_return", 0.0).astype(float)
        out["hybrid_edge"] = (
            out["p_win"] * (out["expected_payoff"] + out["temporal_boost"].clip(lower=0.0) * 0.35)
            - out["p_loss"] * out["expected_loss"] * self.risk_penalty
            - self.transaction_cost
        )
        return out
