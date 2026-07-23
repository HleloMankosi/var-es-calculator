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
# JSE blue-chip basket, equal-weighted. Yahoo Finance uses the ".JO" suffix
# for JSE-listed stocks.
TICKERS = ["NPN.JO", "FSR.JO", "SBK.JO", "MTN.JO", "SOL.JO"]
# Naspers, FirstRand, Standard Bank, MTN Group, Sasol
WEIGHTS = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
START_DATE = "2015-01-01"
END_DATE = None  # None = up to today
PORTFOLIO_VALUE = 100_000  # in real Rand (ZAR), after cents conversion below

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _cache_path(tickers):
    """
    Build a cache filename that reflects exactly which tickers were requested.
    """
    safe_names = [t.replace(".", "") for t in tickers]
    filename = f"prices_{'_'.join(safe_names)}.csv"
    return DATA_DIR / filename


def fetch_prices(tickers=TICKERS, start=START_DATE, end=END_DATE, use_cache=True):
    """
    Download adjusted close prices for all tickers and cache to CSV.

    Note: prices come back exactly as Yahoo Finance reports them — for JSE
    stocks, that's South African CENTS (ZAc), not Rand. This function does
    NOT convert; use cents_to_rand() afterward for that.
    """
    cache_path = _cache_path(tickers)

    if use_cache and cache_path.exists():
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"Loaded {len(prices)} cached rows from {cache_path.name}")
        return prices

    prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    prices = prices.dropna()
    prices.to_csv(cache_path)
    print(f"Fetched {len(prices)} days of data for {len(tickers)} tickers.")
    return prices


def cents_to_rand(prices):
    """
    JSE equity prices come back from Yahoo Finance in South African cents
    (ZAc), not Rand. Converting here means every price level downstream is
    in real Rand. This has NO effect on returns/volatility (ratios cancel
    the /100 out) — it only matters for displayed price levels.
    """
    return prices / 100


def compute_log_returns(prices):
    """
    Convert prices to daily LOG returns: ln(P_t / P_t-1).
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def compute_portfolio_returns(log_returns, weights=WEIGHTS):
    """
    Combine individual asset log returns into one portfolio return series
    using fixed weights.
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
    prices = cents_to_rand(prices)

    print("\nFirst 5 rows (Rand):")
    print(prices.head())
    print("\nLast 5 rows (Rand):")
    print(prices.tail())

    log_returns = compute_log_returns(prices)
    portfolio_returns = compute_portfolio_returns(log_returns)

    print("\nSample of daily portfolio returns:")
    print(portfolio_returns.tail())
    print(f"\nAnnualized volatility: {portfolio_returns.std() * np.sqrt(252):.2%}")
    print(f"Annualized mean return: {portfolio_returns.mean() * 252:.2%}")