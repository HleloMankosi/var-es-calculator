"""
data_loader.py

Fetches historical price data for the portfolio, cleans known data-quality
issues, and converts it into the return series that every risk calculation
downstream depends on.
"""

import hashlib
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# -Portfolio configuration-
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

# A cached price file older than this is re-downloaded.
CACHE_MAX_AGE_HOURS = 24


def _cache_path(tickers, start, end):
    # A short hash keeps the filename well under Windows path-length limits,
    # and the start/end tag stops a cache for one date range being reused
    # for another.
    key = hashlib.md5(",".join(tickers).encode()).hexdigest()[:8]
    return DATA_DIR / f"prices_{len(tickers)}tickers_{key}_{start}_{end or 'latest'}.csv"


def _cache_is_fresh(path, max_age_hours=CACHE_MAX_AGE_HOURS):
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours <= max_age_hours


def warn_if_stale(prices, max_business_days=5):
    """Warns when the newest price in the data is more than a few business days old."""
    last = prices.index.max()
    gap = int(np.busday_count(last.date(), date.today()))
    if gap > max_business_days:
        print(f"WARNING: latest price is {last.date()} ({gap} business days old).")


def fetch_prices(tickers=TICKERS, start=START_DATE, end=END_DATE, use_cache=True):
    """
    Returns adjusted close prices with columns in the SAME ORDER as `tickers`
    (yfinance returns them alphabetically, which would silently misalign
    any non-equal weights). Uses a cache that expires after
    CACHE_MAX_AGE_HOURS; if a refresh fails, falls back to the stale cache
    with a warning instead of crashing.
    """
    tickers = list(tickers)
    cache_path = _cache_path(tickers, start, end)

    if use_cache and _cache_is_fresh(cache_path):
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)[tickers]
        print(f"Loaded {len(prices)} cached rows from {cache_path.name}")
    else:
        try:
            prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
            prices = prices.dropna()
            if prices.empty:
                raise ValueError("download returned no usable rows")
            prices = prices[tickers]  # enforce the TICKERS column order
            prices.to_csv(cache_path)
            print(f"Fetched {len(prices)} days of data for {len(tickers)} tickers.")
        except Exception as exc:
            if not cache_path.exists():
                raise
            print(f"WARNING: download failed ({exc}); falling back to STALE cache.")
            prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)[tickers]

    warn_if_stale(prices)
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
    Flags only the BAD day: an extreme day-over-day move (beyond
    ratio_threshold in either direction) that reverses on the very next day,
    e.g. a x0.01 collapse followed by a x100 recovery. The following, correct
    day is NOT flagged.

    Why 10 is a safe cutoff: the worst GENUINE single-day move in this
    dataset is nowhere near a 10x move, whereas the glitches found in this
    project (single mispriced days for SLM.JO, VOD.JO and SBK.JO in 2025)
    showed ~100x single-day ratios, the signature of a cents/Rand unit error
    at the data source, not a market event.
    """
    ratio = prices / prices.shift(1)
    next_ratio = prices.shift(-1) / prices
    extreme = (ratio > ratio_threshold) | (ratio < 1 / ratio_threshold)
    next_extreme = (next_ratio > ratio_threshold) | (next_ratio < 1 / ratio_threshold)
    net = ratio * next_ratio  # ~1 when the move fully reverses
    reverses = (net >= 0.5) & (net <= 2)
    return extreme & next_extreme & reverses


def clean_price_glitches(prices, ratio_threshold=10, unit_factor=100):
    """
    Repairs single-day unit errors by rescaling the bad day by unit_factor.
    Because the glitch is a cents/Rand mix-up, the fix is exact, not an
    interpolation, and no valid price is overwritten. If the rescaled price
    is implausible, that day falls back to linear interpolation.

    Every correction is printed — silently "fixing" data in a risk tool
    without disclosing what was changed would undermine the whole point of
    the tool. Any correction like this needs to be auditable.
    """
    is_glitch = detect_price_glitches(prices, ratio_threshold)
    cleaned = prices.copy()
    ratio = prices / prices.shift(1)
    n_glitches = int(is_glitch.sum().sum())
    needs_interp = False

    if n_glitches > 0:
        print(f"Detected {n_glitches} single-day unit glitch(es); rescaling the bad day only:")
        for ticker in prices.columns:
            for date_ in prices.index[is_glitch[ticker]]:
                before = prices.loc[date_, ticker]
                factor = unit_factor if ratio.loc[date_, ticker] < 1 else 1 / unit_factor
                after = before * factor
                prev = prices[ticker].shift(1).loc[date_]
                if 0.4 < after / prev < 2.5:
                    cleaned.loc[date_, ticker] = after
                    how = f"rescaled x{factor:g}"
                else:
                    cleaned.loc[date_, ticker] = np.nan
                    needs_interp = True
                    how = "interpolated"
                print(f"  {ticker} {date_.date()}: {before:,.2f} -> "
                      f"{cleaned.loc[date_, ticker]:,.2f} ({how})")
    else:
        print("No price glitches detected.")

    if needs_interp:
        cleaned = cleaned.interpolate(method="linear")

    # Anything still extreme (e.g. a glitch lasting several days) needs a human.
    r2 = cleaned / cleaned.shift(1)
    leftover = int(((r2 > ratio_threshold) | (r2 < 1 / ratio_threshold)).sum().sum())
    if leftover:
        print(f"WARNING: {leftover} extreme move(s) remain after cleaning - inspect manually.")

    return cleaned


def compute_log_returns(prices):
    """Daily LOG returns: ln(P_t / P_t-1). Kept for diagnostics; NOT used to
    build the portfolio return (log returns don't aggregate across assets)."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def compute_returns(prices):
    """
    Daily SIMPLE returns: P_t / P_t-1 - 1.

    A daily-rebalanced portfolio's return is the weighted sum of its assets'
    SIMPLE returns, and Rand P&L is portfolio return x portfolio value, so
    this is the series every VaR/ES calculation needs.
    """
    return (prices / prices.shift(1) - 1).dropna()


def compute_portfolio_returns(asset_returns, weights=WEIGHTS):
    """Combine individual asset (simple) returns into one portfolio return
    series, assuming daily rebalancing to the target weights."""
    if len(weights) != asset_returns.shape[1]:
        raise ValueError("Number of weights must match number of tickers.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1.0 (got {weights.sum()}).")

    portfolio_returns = asset_returns @ weights
    portfolio_returns.name = "portfolio_return"
    return portfolio_returns


def load_portfolio_returns(tickers=TICKERS, weights=WEIGHTS, start=START_DATE,
                             end=END_DATE, use_cache=True, clean_glitches=True):
    """
    One-call pipeline: fetch prices, convert cents to Rand, clean known
    single-day price glitches, then compute simple returns and the combined
    portfolio return series.

    Centralizing this here means every script (var_historical.py,
    var_parametric.py, visualize.py, etc.) imports the SAME source of
    truth for these steps, instead of each duplicating — and potentially
    drifting out of sync with — the same few lines.

    Returns (prices, asset_returns, portfolio_returns).
    """
    prices = fetch_prices(tickers, start, end, use_cache)
    prices = cents_to_rand(prices)
    if clean_glitches:
        prices = clean_price_glitches(prices)
    asset_returns = compute_returns(prices)
    portfolio_returns = compute_portfolio_returns(asset_returns, weights)
    return prices, asset_returns, portfolio_returns


if __name__ == "__main__":
    prices, asset_returns, portfolio_returns = load_portfolio_returns()

    print("\nLast 5 rows (Rand, cleaned):")
    print(prices.tail())

    print("\nSample of daily portfolio returns:")
    print(portfolio_returns.tail())
    print(f"\nAnnualized volatility: {portfolio_returns.std() * np.sqrt(252):.2%}")
    print(f"Annualized mean return: {portfolio_returns.mean() * 252:.2%}")