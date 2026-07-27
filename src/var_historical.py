"""
var_historical.py

Historical Simulation VaR and Expected Shortfall (ES).

Makes NO assumption about the shape of the return distribution — it looks
directly at the empirical distribution of past returns and asks: "looking
at what actually happened, what was the Nth-percentile worst day?"
"""

import numpy as np


def historical_var(returns, confidence=0.95, portfolio_value=100_000):
    alpha = 1 - confidence
    var_return = np.percentile(returns, alpha * 100)
    var_loss = -var_return * portfolio_value
    return var_loss


def historical_es(returns, confidence=0.95, portfolio_value=100_000):
    alpha = 1 - confidence
    var_return = np.percentile(returns, alpha * 100)
    tail_returns = returns[returns <= var_return]
    es_return = tail_returns.mean()
    es_loss = -es_return * portfolio_value
    return es_loss


if __name__ == "__main__":
    from data_loader import load_portfolio_returns, PORTFOLIO_VALUE

    prices, log_returns, portfolio_returns = load_portfolio_returns()

    print(f"\nPortfolio value: R{PORTFOLIO_VALUE:,.0f}\n")
    for conf in [0.95, 0.99]:
        var = historical_var(portfolio_returns, confidence=conf, portfolio_value=PORTFOLIO_VALUE)
        es = historical_es(portfolio_returns, confidence=conf, portfolio_value=PORTFOLIO_VALUE)
        print(f"{conf:.0%} confidence:")
        print(f"  Historical VaR:      R{var:,.2f}")
        print(f"  Expected Shortfall:  R{es:,.2f}")
        print(f"  → on {(1-conf)*100:.0f}% of trading days, expect to lose more than R{var:,.2f};")
        print(f"    on those bad days specifically, average loss is R{es:,.2f}\n")