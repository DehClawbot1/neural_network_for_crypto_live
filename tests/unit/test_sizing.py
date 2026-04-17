import pytest
from money_manager import MoneyManager

def test_sizing_drawdown_limit():
    mm = MoneyManager(family="weather", sleeve_pct=1.0)
    
    res_global_dd = mm.calculate_bet_size(
        available_balance=1000.0,
        edge=0.05,
        global_portfolio_drawdown=1100.0, # Breached MAX
        max_portfolio_drawdown=1000.0
    )
    
    assert res_global_dd["final_size"] == 0.0
    assert res_global_dd["reason"] == "max_portfolio_drawdown_exceeded"

def test_sizing_uncertainty_penalty():
    mm = MoneyManager(family="crypto", sleeve_pct=1.0)
    
    res = mm.calculate_bet_size(
        available_balance=1000.0,
        uncertainty=0.75,  # Too high, size decays to 0
        edge=0.10
    )
    
    assert res["final_size"] == 0.0
    assert res["uncertainty_penalty"] == 0.0
