# Portfolio VaR & Expected Shortfall Calculator

A quantitative risk management tool that calculates Value at Risk (VaR) and Expected Shortfall (ES) on a JSE equity portfolio using historical simulation, with parametric and Monte Carlo methods — plus backtesting — in progress.

Built to understand not just *how* these models work, but *where each one breaks down*.

## Status

- [x] Data pipeline — JSE price fetching via yfinance, ticker validation, cents (ZAc) → Rand conversion, log return calculation
- [x] Historical Simulation VaR & Expected Shortfall
- [x] Return distribution chart with VaR/ES thresholds
- [ ] Parametric (Variance-Covariance) VaR
- [ ] Monte Carlo VaR
- [ ] Backtesting (Kupiec POF test)

## Portfolio

An equal-weighted basket of 19 JSE Top 40-derived stocks, spanning banking, telecoms, retail, resources, and healthcare:

`NPN.JO, FSR.JO, SBK.JO, MTN.JO, SOL.JO, CPI.JO, VOD.JO, ABG.JO, SLM.JO, DSY.JO, SHP.JO, NED.JO, BVT.JO, APN.JO, GRT.JO, CLS.JO, EXX.JO, WHL.JO, MRP.JO`

**Why 19, not the full Top 40:** three candidates were excluded after a data validation step (`check_ticker_availability()`) revealed real data issues — Bid Corporation (BID.JO) and Old Mutual (OMU.JO) only have history from 2016 and 2018 respectively, following corporate spin-offs/restructurings; MultiChoice (MCG.JO) returned no data at all from Yahoo Finance. Including any of these would have forced the entire portfolio's start date forward, discarding years of history for every other stock.

## Methodology

**Returns:** Daily log returns (`ln(P_t / P_t-1)`), used for their time-additive property.

**Currency handling:** Yahoo Finance reports JSE equity prices in South African cents (ZAc), not Rand — confirmed directly against Yahoo's own listing pages and cross-checked against independently published Rand prices. Converted explicitly via `cents_to_rand()`. This has zero effect on returns or volatility (ratios cancel the /100 out) — it only matters for displayed price levels.

**Historical Simulation VaR/ES:** Makes no assumption about the shape of the return distribution — it looks directly at the empirical distribution of past returns and asks *"what was the Nth-percentile worst day?"* (VaR) and *"on the worst days, what was the average loss?"* (Expected Shortfall).

## Key findings

**1. Diversification reduces volatility meaningfully.** Expanding from a 5-stock to a 19-stock JSE portfolio cut annualized volatility from 46.51% to 30.64%, and reduced both VaR and ES by roughly 15–20% at every confidence level tested:

| Metric | 5-stock portfolio | 19-stock portfolio |
|---|---|---|
| 95% VaR | R2,481.19 | R1,995.53 |
| 95% ES | R4,398.95 | R3,539.68 |
| 99% VaR | R4,424.72 | R3,648.64 |
| 99% ES | R9,606.14 | R7,377.52 |

**2. Diversification does not remove fat tails.** Despite the volatility drop, the ratio of 99% VaR to 95% VaR stayed almost identical (~1.83x for both portfolios) — well above the ~1.4x a normal distribution would predict at those confidence levels. This suggests the fat-tail behavior is a *systematic* market-wide feature (all 19 stocks are still JSE equities, exposed to the same broad market shocks), not something diversification within a single asset class and market can diversify away. This is exactly why VaR/ES modelling and stress testing exist as a separate discipline from portfolio optimization — reducing volatility isn't the same as reducing tail risk.

## Visualization

![Portfolio P&L distribution with VaR/ES thresholds](output/return_distribution.png)

The histogram shows the empirical distribution of daily portfolio P&L, with 95%/99% VaR and Expected Shortfall marked. The left tail extending further than a normal distribution would predict is the visual signature of the fat-tail finding above.

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
python src/visualize.py         # generate the P&L distribution chart
```

## Project structure