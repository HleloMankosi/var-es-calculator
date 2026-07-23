"""
data_loader.py

Fetches historical price data for the portfolio and converts it into the
return series that every risk calculation downstream depends on.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# ---- Portfolio configuration ----
# Equal-weighted 5-asset portfolio to start. Swap tickers/weights any time.
# For JSE stocks via Yahoo Finance, use the ".JO" suffix, e.g. "NPN.JO" (Naspers).
TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "TLT"]
WEIGHTS = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
START_DATE = "2015-01-01"
END_DATE = None  # None = up to today
PORTFOLIO_VALUE = 100_000  # in USD, matching the tickers above

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_prices(tickers=TICKERS, start=START_DATE, end=END_DATE, use_cache=True):
    """
    Download adjusted close prices for all tickers and cache to CSV, so we're
    not re-hitting Yahoo Finance every time we run this while developing.
    """
    cache_path = DATA_DIR / "prices.csv"

    if use_cache and cache_path.exists():
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"Loaded {len(prices)} cached rows from {cache_path}")
        return prices

    prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    prices = prices.dropna()  # drop any day where a ticker's data is missing
    prices.to_csv(cache_path)
    print(f"Fetched {len(prices)} days of data for {len(tickers)} tickers.")
    return prices

def compute_log_returns(prices):
    """
    Convert prices to daily LOG returns: ln(P_t / P_t-1).

    Log returns (not simple % returns) are the standard in risk modeling
    because they're time-additive — a week's log return is just the sum of
    that week's daily log returns, which isn't true for simple returns.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def compute_portfolio_returns(log_returns, weights=WEIGHTS):
    """
    Combine individual asset log returns into one portfolio return series
    using fixed weights.

    Note: this assumes daily rebalancing back to target weights. A real
    portfolio drifts between rebalances — worth mentioning as a stated
    simplification in your README later, not something to hide.
    """
    if len(weights) != log_returns.shape[1]:
        raise ValueError("Number of weights must match number of tickers.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1.0 (got {weights.sum()}).")

    portfolio_returns = log_returns @ weights
    portfolio_returns.name = "portfolio_return"
    return portfolio_returns

if __name__ == "__main__":
    prices = fetch_prices()
    log_returns = compute_log_returns(prices)
    portfolio_returns = compute_portfolio_returns(log_returns)

    print("\nSample of daily portfolio returns:")
    print(portfolio_returns.tail())
    print(f"\nAnnualized volatility: {portfolio_returns.std() * np.sqrt(252):.2%}")
    print(f"Annualized mean return: {portfolio_returns.mean() * 252:.2%}")