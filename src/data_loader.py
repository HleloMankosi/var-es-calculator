"""
data_loader.py

Fetches historical price data for the portfolio, cleans known data-quality
issues, and converts it into the return series that every risk calculation
downstream depends on.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# ---- Portfolio configuration ----
TICKERS = [
    "NPN.JO", "FSR.JO", "SBK.JO", "MTN.JO", "SOL.JO",
    "CPI.JO", "VOD.JO", "ABG.JO", "SLM.JO", "DSY.JO",
    "SHP.JO", "NED.JO", "BVT.JO", "APN.JO", "GRT.JO",
    "CLS.JO", "EXX.JO", "WHL.JO", "MRP.JO",
]
WEIGHTS = np.full(len(TICKERS), 1 / len(TICKERS))
START_DATE = "2015-01-01"
END_DATE = None
PORTFOLIO_VALUE = 100_000

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


def detect_price_glitches(prices, ratio_threshold=10):
    """
    Flags implausible single-day price ratios: day-over-day moves more
    extreme than ratio_threshold in either direction.

    Why 10 is a safe cutoff: the worst GENUINE single-day move anywhere in
    this dataset is Sasol's ~47% drop during the March 2020 crash (a ratio
    of about 0.53) — nowhere close to a 10x move. But real data-quality
    glitches found in this project (a single mispriced day for SLM.JO,
    VOD.JO, and SBK.JO in 2025) showed ~100x single-day ratios, which is
    the signature of a unit error (cents vs. Rand) at the data source, not
    a market event. A threshold of 10 catches those with a wide safety
    margin above anything real.
    """
    ratio = prices / prices.shift(1)
    is_glitch = (ratio > ratio_threshold) | (ratio < 1 / ratio_threshold)
    return is_glitch


def clean_price_glitches(prices, ratio_threshold=10):
    """
    Detects and repairs single-day price glitches by replacing flagged
    values with NaN and linearly interpolating from the nearest valid
    prices on either side.

    Every correction is printed explicitly — silently "fixing" data in a
    risk tool without disclosing what was changed would undermine the
    whole point of the tool. Any correction like this needs to be
    auditable, not hidden.
    """
    is_glitch = detect_price_glitches(prices, ratio_threshold)
    n_glitches = int(is_glitch.sum().sum())

    if n_glitches > 0:
        print(f"Detected {n_glitches} implausible single-day price glitch(es):")
        for ticker in prices.columns:
            flagged_dates = prices.index[is_glitch[ticker]]
            for date in flagged_dates:
                print(f"  {ticker} on {date.date()}: flagged value {prices.loc[date, ticker]:,.2f} "
                      f"(ratio vs. prior day: {(prices.loc[date, ticker] / prices[ticker].shift(1).loc[date]):.2f}x)")
    else:
        print("No price glitches detected.")

    cleaned = prices.copy()
    cleaned[is_glitch] = np.nan
    cleaned = cleaned.interpolate(method="linear")
    return cleaned


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


def load_portfolio_returns(tickers=TICKERS, weights=WEIGHTS, start=START_DATE,
                             end=END_DATE, use_cache=True, clean_glitches=True):
    """
    One-call pipeline: fetch prices, convert cents to Rand, clean known
    single-day price glitches, then compute log returns and the combined
    portfolio return series.

    Centralizing this here means every script (var_historical.py,
    var_parametric.py, visualize.py, etc.) imports the SAME source of
    truth for these steps, instead of each duplicating — and potentially
    drifting out of sync with — the same few lines.
    """
    prices = fetch_prices(tickers, start, end, use_cache)
    prices = cents_to_rand(prices)
    if clean_glitches:
        prices = clean_price_glitches(prices)
    log_returns = compute_log_returns(prices)
    portfolio_returns = compute_portfolio_returns(log_returns, weights)
    return prices, log_returns, portfolio_returns


if __name__ == "__main__":
    prices, log_returns, portfolio_returns = load_portfolio_returns()

    print("\nLast 5 rows (Rand, cleaned):")
    print(prices.tail())

    print("\nSample of daily portfolio returns:")
    print(portfolio_returns.tail())
    print(f"\nAnnualized volatility: {portfolio_returns.std() * np.sqrt(252):.2%}")
    print(f"Annualized mean return: {portfolio_returns.mean() * 252:.2%}")