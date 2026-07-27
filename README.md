# Portfolio VaR & Expected Shortfall Calculator

A quantitative risk management tool that calculates Value at Risk (VaR) and Expected Shortfall (ES) on a JSE equity portfolio using Historical Simulation, Parametric (Variance-Covariance), and Monte Carlo methods, with backtesting in progress.

Built to understand not just *how* these models work, but *where each one breaks down* — including catching a real data-quality issue along the way.

## Status

- [x] Data pipeline — JSE price fetching via yfinance, ticker validation, cents (ZAc) → Rand conversion, price-glitch detection & correction, log return calculation
- [x] Historical Simulation VaR & Expected Shortfall
- [x] Parametric (Variance-Covariance) VaR & Expected Shortfall
- [x] Monte Carlo VaR (Gaussian and Bootstrap variants)
- [x] Return distribution chart with VaR/ES thresholds
- [ ] Backtesting (Kupiec POF test)

## Portfolio

An equal-weighted basket of 19 JSE Top 40-derived stocks, spanning banking, telecoms, retail, resources, and healthcare:

`NPN.JO, FSR.JO, SBK.JO, MTN.JO, SOL.JO, CPI.JO, VOD.JO, ABG.JO, SLM.JO, DSY.JO, SHP.JO, NED.JO, BVT.JO, APN.JO, GRT.JO, CLS.JO, EXX.JO, WHL.JO, MRP.JO`

**Why 19, not the full Top 40:** three candidates were excluded after a data validation step (`check_ticker_availability()`) revealed Bid Corporation (BID.JO) and Old Mutual (OMU.JO) only have history from 2016/2018 respectively (post spin-off/restructuring), and MultiChoice (MCG.JO) returned no data from Yahoo Finance at all.

## Methodology

**Returns:** Daily log returns (`ln(P_t / P_t-1)`), used for their time-additive property.

**Currency handling:** Yahoo Finance reports JSE equity prices in South African cents (ZAc), not Rand — converted explicitly via `cents_to_rand()`. This has zero effect on returns/volatility (ratios cancel the /100 out); it only matters for displayed price levels.

**Data quality — price glitch detection:** Initial analysis showed an implausible excess kurtosis of 274.5, which triggered an investigation (`diagnose_outliers.py`) rather than being reported as a finding. It traced to six single-day price errors across three tickers (SLM.JO, VOD.JO, SBK.JO) in 2025, each showing a ~100x single-day price ratio — the signature of a cents/Rand unit error at the data source, not a real market move (confirmed by the fact that each "crash" was followed by an exact-magnitude "recovery" the next available trading day, and no other stock in the portfolio moved that day). A detection function (`detect_price_glitches()`, flagging any day-over-day ratio beyond 10x — safely above the worst genuine move in this dataset, Sasol's ~47% single-day drop during the March 2020 crash) and correction function (`clean_price_glitches()`, linear interpolation from neighboring valid prices) now catch and repair these automatically, with every correction printed for auditability rather than silently applied.

**Historical Simulation VaR/ES:** Makes no assumption about the shape of the return distribution — uses the empirical distribution of past returns directly.

**Parametric (Variance-Covariance) VaR/ES:** Assumes returns are Normally distributed, characterized by mean and standard deviation alone. Portfolio volatility is computed two independent ways as a cross-check — directly from the combined return series, and via the covariance matrix (`w' Σ w`) — confirmed to match exactly. Expected Shortfall under Normality has an exact closed-form solution, unlike Historical ES, which must average real observed tail data.

**Monte Carlo VaR/ES:** Two variants implemented. *Gaussian Monte Carlo* simulates 100,000 correlated daily return scenarios by sampling from a multivariate Normal distribution (via Cholesky decomposition of the covariance matrix) — this should converge to the same answer as Parametric VaR, serving as an implementation cross-check. *Bootstrap Monte Carlo* instead resamples actual historical days with replacement (preserving real cross-asset correlations), inheriting the true empirical distribution's shape rather than assuming Normality — this should converge toward Historical Simulation's numbers instead.

## Key findings

**1. A data-quality bug initially made tail risk look far more extreme than it was.** Before cleaning, excess kurtosis measured 274.5 — implausible for a diversified 19-stock equity portfolio. Investigating rather than reporting this number traced it to six single-day data errors (cents/Rand unit mixups at the source for three tickers in 2025). After correction, excess kurtosis dropped to a realistic **6.29**, and annualized volatility fell from 30.64% to **20.98%**. This is arguably the most important finding in the project: verifying data quality before trusting a risk number is not optional in this field.

**2. Diversification reduces volatility meaningfully.** Even after data cleaning, expanding from 5 to 19 JSE stocks meaningfully reduces portfolio volatility and VaR/ES versus a concentrated portfolio, consistent with standard diversification theory.

**3. Real fat tails persist after cleaning — and Expected Shortfall reveals them better than VaR alone.**

| Metric | Historical | Parametric (Normal) | Gap |
|---|---|---|---|
| 95% VaR | R1,989.69 | R2,161.24 | -7.9% |
| 95% ES | R3,072.47 | R2,713.62 | **+13.2%** |
| 99% VaR | R3,518.86 | R3,062.13 | +14.9% |
| 99% ES | R5,209.94 | R3,510.09 | **+48.4%** |

At 95% confidence, Historical VaR is actually *below* the Normal-model prediction — but Historical ES is already *above* it. By 99%, Historical exceeds Parametric on both measures, with the ES gap widening dramatically to +48.4%. This is a known signature of fat-tailed ("leptokurtic") return distributions: they're often more peaked near the center with thinner "shoulders," so a Normal model can look adequate or even conservative at moderate confidence levels — while the real excess risk is concentrated further into the tail, exactly where a single VaR cutoff can miss it but Expected Shortfall (which averages everything beyond that point) reliably catches it.

**4. Four independent methods confirm the fat-tail effect is structural, not a calculation quirk.** Gaussian Monte Carlo (100,000 simulations) matched Parametric VaR within 0.35%, and Bootstrap Monte Carlo matched Historical VaR within 0.16% — validating both simulation engines. More importantly, this collapses cleanly into two camps: the two Normal-distribution-based methods (Parametric, Gaussian MC) agree closely with each other, and the two empirical-distribution-based methods (Historical, Bootstrap MC) agree closely with each other — with a ~15% gap on VaR and a ~49% gap on Expected Shortfall between the two camps at 99% confidence. Four different mathematical approaches converging into exactly two answers is strong evidence the fat-tail effect is a genuine property of JSE equity returns, not an artifact of any single method's implementation.

## Visualization

![Portfolio P&L distribution with VaR/ES thresholds](output/return_distribution.png)

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
python src/data_loader.py       # fetch/cache price data, detect/clean glitches, print return stats
python src/var_historical.py    # compute Historical VaR & Expected Shortfall
python src/var_parametric.py    # compute Parametric VaR & ES, compare against Historical
python src/var_monte_carlo.py   # compute Monte Carlo VaR & ES (Gaussian + Bootstrap), cross-validate vs Historical/Parametric
python src/visualize.py         # generate the P&L distribution chart
python src/diagnose_outliers.py # investigate extreme single-day returns
```

## Project structure

## Known limitations

- Assumes daily rebalancing to fixed target weights
- Historical VaR is bounded by the 2015–present sample window
- Equal weighting is a simplifying starting assumption, not a risk-optimized allocation
- All 19 holdings are JSE equities — no cross-asset-class diversification
- Price-glitch detection uses a fixed 10x ratio threshold; a more adaptive method (e.g. rolling z-score) could catch subtler data issues
- Monte Carlo here is largely a validation exercise rather than strictly necessary — it becomes essential once a portfolio includes non-linear instruments (options, etc.) where Parametric's assumptions break down, but this portfolio is all linear equity exposure

## Roadmap

Kupiec backtesting next — validating which method's VaR estimate actually held up against realized daily outcomes over the historical sample, rather than only comparing methods against each other.