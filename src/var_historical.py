"""
var_historical.py

Historical Simulation VaR and Expected Shortfall (ES).

This is the simplest of three standard VaR methods (historical, parametric,
and Monte Carlo — the other two come later) because it makes NO assumption
about the shape of the return distribution. It just asks: "looking at what
actually happened, what was the Nth-percentile worst day?"

Trade-off worth knowing: historical VaR can't imagine a crash worse than
anything already in your lookback window, and it reacts slowly to genuine
regime changes.
"""

import numpy as np


def historical_var(returns, confidence=0.95, portfolio_value=100_000):
    """
    The loss threshold that returns exceeded only (1 - confidence) of the
    time historically. Returns a POSITIVE number representing a Rand loss,
    by convention (even though the underlying return is negative).
    """
    alpha = 1 - confidence
    var_return = np.percentile(returns, alpha * 100)
    var_loss = -var_return * portfolio_value
    return var_loss


def historical_es(returns, confidence=0.95, portfolio_value=100_000):
    """
    Expected Shortfall (a.k.a. Conditional VaR): the AVERAGE loss in the
    worst (1 - confidence) of cases.
    """
    alpha = 1 - confidence
    var_return = np.percentile(returns, alpha * 100)
    tail_returns = returns[returns <= var_return]
    es_return = tail_returns.mean()
    es_loss = -es_return * portfolio_value
    return es_loss


if __name__ == "__main__":
    from data_loader import fetch_prices, cents_to_rand, compute_log_returns, compute_portfolio_returns, PORTFOLIO_VALUE

    prices = fetch_prices()
    prices = cents_to_rand(prices)
    log_returns = compute_log_returns(prices)
    portfolio_returns = compute_portfolio_returns(log_returns)

    print(f"\nPortfolio value: R{PORTFOLIO_VALUE:,.0f}\n")
    for conf in [0.95, 0.99]:
        var = historical_var(portfolio_returns, confidence=conf, portfolio_value=PORTFOLIO_VALUE)
        es = historical_es(portfolio_returns, confidence=conf, portfolio_value=PORTFOLIO_VALUE)
        print(f"{conf:.0%} confidence:")
        print(f"  Historical VaR:      R{var:,.2f}")
        print(f"  Expected Shortfall:  R{es:,.2f}")
        print(f"  → on {(1-conf)*100:.0f}% of trading days, expect to lose more than R{var:,.2f};")
        print(f"    on those bad days specifically, average loss is R{es:,.2f}\n")