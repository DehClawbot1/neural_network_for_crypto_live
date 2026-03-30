class PNLEngine:
    """
    Correct paper PnL math for Polymarket-style outcome-token share accounting.
    YES and NO are both long token holdings once purchased.
    """

    @staticmethod
    def shares_from_capital(capital_usdc: float, entry_price: float) -> float:
        if not entry_price or entry_price <= 0:
            return 0.0
        return int((return float(capital_usdc) / float(entry_price)) * 1e6) / 1e6 # BUG FIX 6: Truncate precisely to 6 decimals for conditional tokens

    @staticmethod
    def mark_to_market_value(capital_usdc: float, entry_price: float, current_token_price: float) -> float:
        shares = PNLEngine.shares_from_capital(capital_usdc, entry_price)
        return shares * float(current_token_price)

    @staticmethod
    def mark_to_market_pnl(capital_usdc: float, entry_price: float, current_token_price: float, fees: float = 0.0) -> float:
        shares = PNLEngine.shares_from_capital(capital_usdc, entry_price)
        return (shares * float(current_token_price)) - (shares * float(entry_price)) - float(fees)

    @staticmethod
    def resolution_pnl(capital_usdc: float, entry_price: float, token_won: bool, fees: float = 0.0) -> float:
        final_price = 1.0 if token_won else 0.0
        return PNLEngine.mark_to_market_pnl(capital_usdc, entry_price, final_price, fees=fees)

    @staticmethod
    def summarize_trade(capital_usdc: float, entry_price: float, exit_price: float, fees: float = 0.0) -> dict:
        shares = PNLEngine.shares_from_capital(capital_usdc, entry_price)
        market_value = shares * float(exit_price)
        pnl = PNLEngine.mark_to_market_pnl(capital_usdc, entry_price, exit_price, fees=fees)
        return {
            "capital_usdc": capital_usdc,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "shares": shares,
            "market_value": market_value,
            "pnl": pnl,
        }

