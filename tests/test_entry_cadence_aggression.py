from strategy_layers import EntryRuleLayer
from entry_aggression import apply_entry_cadence_boost


def test_apply_entry_cadence_boost_promotes_top_candidate():
    row = {
        "confidence": 0.22,
        "force_candidate": False,
    }
    boosted = apply_entry_cadence_boost(row, minutes_idle=12.0, target_minutes=5.0, candidate_rank=1)
    assert boosted["activity_target_mode"] is True
    assert boosted["force_candidate"] is True
    assert boosted["confidence"] > row["confidence"]
    assert boosted["entry_score_relax"] > 0


def test_entry_rule_relaxation_can_allow_borderline_candidate():
    rule = EntryRuleLayer(min_score=0.20, max_spread=0.30, min_liquidity=0.50, min_liquidity_score=0.05)
    row = {
        "confidence": 0.15,
        "spread": 0.33,
        "liquidity": 0.20,
        "entry_score_relax": 0.10,
        "entry_spread_relax": 0.05,
        "entry_liquidity_relax_factor": 0.30,
    }
    result = rule.evaluate(row)
    assert result["allow"] is True
    assert result["score_threshold"] <= 0.10 + 1e-9
    assert result["spread_threshold"] >= 0.35 - 1e-9
