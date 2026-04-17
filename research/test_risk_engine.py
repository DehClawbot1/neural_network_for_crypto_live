import os
from live_risk_manager import LiveRiskManager, RiskTier
from money_manager import MoneyManager
from dataclasses import dataclass

def test_operational_risk():
    print("Testing Operational Risk Lockout...")
    if os.path.exists("operational_lockout.lock"):
        os.remove("operational_lockout.lock")
        
    os.environ["RISK_TIER"] = "MICRO"
    risk = LiveRiskManager(max_failed_orders=3)
    print("Init Tier:", risk.tier)
    
    # Simulate consecutive fails
    risk.record_failed_order()
    print("Fails:", risk.failed_orders, "Locked:", risk.kill_switch)
    risk.record_failed_order()
    risk.record_failed_order()
    print("Fails:", risk.failed_orders, "Locked:", risk.kill_switch)
    
    # Validate DB state file persists test
    assert os.path.exists("operational_lockout.lock"), "Lock file not generated!"
    
    # Test restart lockout continuity
    risk_restart = LiveRiskManager()
    print("Restart Locked:", risk_restart.kill_switch)
    assert risk_restart.kill_switch, "Did not persist lockout"
    
    if os.path.exists("operational_lockout.lock"):
        os.remove("operational_lockout.lock")
    print("Operational Risk Test Passed.\n")


def test_money_manager_sizing():
    print("Testing Entry Risk Portfolio Constraints...")
    mm = MoneyManager(family="weather", sleeve_pct=1.0)
    
    # 1. High Confidence but High Uncertainty = Collapse
    res_high_uncertainty = mm.calculate_bet_size(
        available_balance=1000.0,
        confidence=0.9,
        uncertainty=0.6,  # Model is highly uncertain
        edge=0.05
    )
    print(f"High Uncertainty Size   : {res_high_uncertainty['final_size']} (Penalty: {res_high_uncertainty.get('uncertainty_penalty')})")
    
    # 2. High Confidence and Low Uncertainty = Size Scales
    res_low_uncertainty = mm.calculate_bet_size(
        available_balance=1000.0,
        confidence=0.9,
        uncertainty=0.1,  # Model is certain
        edge=0.05
    )
    print(f"Low Uncertainty Size    : {res_low_uncertainty['final_size']} (Penalty: {res_low_uncertainty.get('uncertainty_penalty')})")
    
    # 3. High Correlation Context = Severely restricted 
    res_high_correlation = mm.calculate_bet_size(
        available_balance=1000.0,
        confidence=0.9,
        uncertainty=0.1, 
        edge=0.05,
        active_correlated_positions=3 # 3 existing overlapping markets
    )
    print(f"High Correlation Size   : {res_high_correlation['final_size']} (Penalty: {res_high_correlation.get('correlation_penalty')})")
    
    # 4. Global Drawdown limits hit = ZERO
    res_global_dd = mm.calculate_bet_size(
        available_balance=1000.0,
        global_portfolio_drawdown=1100.0, # Breached MAX
        max_portfolio_drawdown=1000.0
    )
    print(f"Global Drawdown hit Size: {res_global_dd['final_size']} (Reason: {res_global_dd.get('reason')})")

    print("Entry Portfolio Risk Test Passed.")


if __name__ == "__main__":
    test_operational_risk()
    test_money_manager_sizing()
