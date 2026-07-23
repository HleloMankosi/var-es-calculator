# Portfolio VaR & Expected Shortfall Calculator

A quantitative risk management tool that calculates Value at Risk (VaR) and Expected Shortfall (ES) on a multi-asset JSE portfolio using historical simulation, with parametric and Monte Carlo methods — plus backtesting — in progress.

Built to understand not just *how* these models work, but *where each one breaks down*.

## Status

- [x] Data pipeline — JSE price fetching via yfinance, cents (ZAc) → Rand conversion, log return calculation
- [x] Historical Simulation VaR & Expected Shortfall
- [ ] Parametric (Variance-Covariance) VaR
- [ ] Monte Carlo VaR
- [ ] Backtesting (Kupiec POF test)
- [ ] Visualization dashboard

## Portfolio

An equal-weighted basket of 5 JSE-listed blue-chip stocks:

| Ticker | Company |
|---|---|
| NPN.JO | Naspers |
| FSR.JO | FirstRand |
| SBK.JO | Standard Bank |
| MTN.JO | MTN Group |
| SOL.JO | Sasol |

## Methodology

**Returns:** Daily log returns (`ln(P_t / P_t-1)`) are used rather than simple percentage returns, since log returns are time-additive — a property simple returns don't have — which matters for aggregating risk across periods.

**A data quirk worth documenting:** Yahoo Finance reports JSE equity prices in South African **cents (ZAc)**, not Rand — confirmed directly from Yahoo's own listing pages. This is converted explicitly in the data pipeline (`cents_to_rand()`). Note that this conversion has zero effect on returns or volatility (they're ratios, so the /100 cancels out) — it only matters for actual displayed price levels and anything computed from price levels directly (e.g. share counts).

**Historical Simulation VaR/ES:** Rather than assuming any particular return distribution, this method looks directly at the empirical distribution of past portfolio returns and asks: *"what was the Nth-percentile worst day?"* (VaR), and *"on the worst days, what was the average loss?"* (Expected Shortfall). This makes no distributional assumptions, but is limited by whatever's actually in the lookback window — it can't imagine a crash worse than anything already observed, and reacts slowly to genuine regime shifts.

## Key finding so far

At 95% confidence, 1-day VaR is **R2,481** on a R100,000 portfolio; at 99% confidence, it's **R4,425** — a ~1.8x jump. Under a normal distribution, that same confidence jump would only be expected to produce roughly a 1.4x increase. This larger-than-normal jump is a classic signature of **fat tails (excess kurtosis)** in real market returns — a well-documented feature of financial data that a parametric (normal-distribution) VaR model would understate. Quantifying this gap precisely is exactly what the upcoming parametric VaR module will let this project do.

## Setup

```powershell
git clone https://github.com/<your-username>/var-es-calculator.git
cd var-es-calculator
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

```powershell
python src/data_loader.py       # fetch/cache price data, print return stats
python src/var_historical.py    # compute Historical VaR & Expected Shortfall
```

## Project structure