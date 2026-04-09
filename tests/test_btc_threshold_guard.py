from btc_threshold_guard import (
    btc_price_threshold_positions_conflict,
    find_conflicting_btc_price_threshold_position,
    parse_btc_price_threshold_market,
)


def test_parse_btc_threshold_market_from_title():
    parsed = parse_btc_price_threshold_market(
        market="Will the price of Bitcoin be above $72,000 on April 15?",
        market_slug="",
    )

    assert parsed["threshold_price"] == 72000.0
    assert parsed["expiry_key"] == "april-15"


def test_parse_btc_threshold_market_from_slug():
    parsed = parse_btc_price_threshold_market(
        market="",
        market_slug="bitcoin-above-68k-on-april-15",
    )

    assert parsed["threshold_price"] == 68000.0
    assert parsed["expiry_key"] == "april-15"


def test_btc_threshold_conflict_for_yes_above_higher_and_no_above_lower_same_expiry():
    left = {
        "market": "Will the price of Bitcoin be above $72,000 on April 15?",
        "outcome_side": "YES",
    }
    right = {
        "market": "Will the price of Bitcoin be above $68,000 on April 15?",
        "outcome_side": "NO",
    }

    assert btc_price_threshold_positions_conflict(left, right) is True


def test_btc_threshold_allows_yes_lower_and_no_higher_same_expiry():
    left = {
        "market": "Will the price of Bitcoin be above $68,000 on April 15?",
        "outcome_side": "YES",
    }
    right = {
        "market": "Will the price of Bitcoin be above $72,000 on April 15?",
        "outcome_side": "NO",
    }

    assert btc_price_threshold_positions_conflict(left, right) is False


def test_btc_threshold_allows_different_expiry_dates():
    left = {
        "market": "Will the price of Bitcoin be above $72,000 on April 15?",
        "outcome_side": "YES",
    }
    right = {
        "market": "Will the price of Bitcoin be above $68,000 on April 16?",
        "outcome_side": "NO",
    }

    assert btc_price_threshold_positions_conflict(left, right) is False


def test_find_conflicting_threshold_position_returns_existing_conflict():
    candidate = {
        "market": "Will the price of Bitcoin be above $72,000 on April 15?",
        "outcome_side": "YES",
    }
    open_positions = [
        {
            "market": "Will the price of Bitcoin be above $68,000 on April 15?",
            "outcome_side": "NO",
            "condition_id": "cond-68",
        }
    ]

    result = find_conflicting_btc_price_threshold_position(candidate, open_positions)

    assert result is not None
    assert result["existing"]["condition_id"] == "cond-68"
    assert result["expiry_key"] == "april-15"
