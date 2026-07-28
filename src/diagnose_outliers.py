"""
diagnose_outliers.py

var_parametric.py surfaced excess kurtosis of ~274 -- vastly higher than
what a diversified 19-stock equity portfolio should plausibly show from
ordinary market behavior. Kurtosis is extremely sensitive to a small number
of extreme outliers, so before trusting ANY VaR/ES numbers built on this
data, we need to find out exactly which day(s) and which ticker(s) are
driving it -- and whether it's a real market event or a data artifact.
"""

from data_loader import fetch_prices, cents_to_rand, compute_log_returns, compute_portfolio_returns, WEIGHTS

prices = fetch_prices()
prices = cents_to_rand(prices)
log_returns = compute_log_returns(prices)
portfolio_returns = compute_portfolio_returns(log_returns, WEIGHTS)

print("Top 10 most extreme portfolio daily returns:")
extreme_days = portfolio_returns.abs().sort_values(ascending=False).head(10)

for date in extreme_days.index:
    print(f"\n{date.date()}  portfolio return: {portfolio_returns[date]:+.4f}")
    day_returns = log_returns.loc[date].sort_values()
    print(day_returns.to_string())

print("\n\nMost extreme single-day return per individual ticker:")
for ticker in log_returns.columns:
    idx_max = log_returns[ticker].abs().idxmax()
    print(f"  {ticker}: {log_returns.loc[idx_max, ticker]:+.4f} on {idx_max.date()}")