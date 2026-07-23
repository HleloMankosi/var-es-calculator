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
# JSE Top 40-derived basket, equal-weighted. Yahoo Finance uses the ".JO"
# suffix for JSE-listed stocks.
#
# This final list of 19 was chosen using check_ticker_availability() below,
# run against a broader candidate list. Three candidates were excluded:
#   - BID.JO (Bid Corporation): only data from 2016-05-27 (spun off from
#     Bidvest in 2016)
#   - OMU.JO (Old Mutual): only data from 2018-06-27 (2018 restructuring)
#   - MCG.JO (MultiChoice): no data returned at all (404 from Yahoo)
# Mixing those in would force the whole portfolio's start date to 2018,
# discarding years of history for the other 19 names.
TICKERS = [
    "NPN.JO", "FSR.JO", "SBK.JO", "MTN.JO", "SOL.JO",
    "CPI.JO", "VOD.JO", "ABG.JO", "SLM.JO", "DSY.JO",
    "SHP.JO", "NED.JO", "BVT.JO", "APN.JO", "GRT.JO",
    "CLS.JO", "EXX.JO", "WHL.JO", "MRP.JO",
]
WEIGHTS = np.full(len(TICKERS), 1 / len(TICKERS))  # equal-weighted
START_DATE = "2015-01-01"
END_DATE = None
PORTFOLIO_VALUE = 100_000

# Broader candidate list used to arrive at the final TICKERS above via
# check_ticker_availability(). Kept here for reproducibility/reference.
CANDIDATE_TICKERS = [
    "NPN.JO", "FSR.JO", "SBK.JO", "MTN.JO", "SOL.JO",
    "CPI.JO", "VOD.JO", "ABG.JO", "SLM.JO", "DSY.JO",
    "SHP.JO", "BID.JO", "NED.JO", "BVT.JO", "APN.JO",
    "GRT.JO", "OMU.JO", "MCG.JO", "CLS.JO", "EXX.JO",
    "WHL.JO", "MRP.JO",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _cache_path(tickers):
    safe_names = [t.replace(".", "") for t in tickers]
    filename = f"prices_{'_'.join(safe_names)}.csv"
    return DATA_DIR / filename


def fetch_prices(tickers=TICKERS, start=START_DATE, end=END_DATE, use_cache=True):
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


def check_ticker_availability(tickers, start=START_DATE, end=END_DATE):
    """
    Diagnostic tool: checks each candidate ticker independently and reports
    its actual data history. Several JSE Top 40 names only exist as their
    current listing from a specific corporate event (spin-off, unbundling,
    restructuring) rather than from START_DATE — feeding those into a
    multi-ticker fetch + dropna() would silently truncate the ENTIRE
    dataset back to whichever ticker listed most recently.
    """
    results = []
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        if len(data) == 0:
            results.append({"ticker": ticker, "status": "NO DATA", "first_date": None, "rows": 0})
        else:
            results.append({
                "ticker": ticker,
                "status": "OK",
                "first_date": data.index.min().date(),
                "rows": len(data),
            })

    report = pd.DataFrame(results)
    print(report.to_string(index=False))
    return report


def cents_to_rand(prices):
    """
    JSE equity prices come back from Yahoo Finance in South African cents
    (ZAc), not Rand. This has NO effect on returns/volatility (ratios cancel
    the /100 out) — it only matters for displayed price levels.
    """
    return prices / 100


def compute_log_returns(prices):
    """Daily LOG returns: ln(P_t / P_t-1)."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def compute_portfolio_returns(log_returns, weights=WEIGHTS):
    """Combine individual asset log returns into one portfolio return series."""
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