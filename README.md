# Portfolio VaR & Expected Shortfall Calculator

A quantitative risk management tool that calculates, cross-validates, and backtests Value at Risk (VaR) and Expected Shortfall (ES) on a JSE equity portfolio using Historical Simulation, Parametric (Variance-Covariance), and Monte Carlo methods, with historical scenario stress testing.

Built to understand not just *how* these models work, but *where each one breaks down* — including catching a real data-quality issue, formally proving one method's assumptions fail where it matters most, and showing that diversification's value depends on the type of shock a portfolio actually faces.

## Status

- [x] Data pipeline — JSE price fetching via yfinance, ticker validation, cents (ZAc) → Rand conversion, price-glitch detection & correction, log return calculation
- [x] Historical Simulation VaR & Expected Shortfall
- [x] Parametric (Variance-Covariance) VaR & Expected Shortfall
- [x] Monte Carlo VaR (Gaussian and Bootstrap variants)
- [x] Return distribution chart with VaR/ES thresholds
- [x] Kupiec Proportion-of-Failures backtesting
- [x] Historical scenario stress testing (COVID-19, Nenegate)

## Portfolio

An equal-weighted basket of 19 JSE Top 40-derived stocks, spanning banking, telecoms, retail, resources, and healthcare:

`NPN.JO, FSR.JO, SBK.JO, MTN.JO, SOL.JO, CPI.JO, VOD.JO, ABG.JO, SLM.JO, DSY.JO, SHP.JO, NED.JO, BVT.JO, APN.JO, GRT.JO, CLS.JO, EXX.JO, WHL.JO, MRP.JO`

**Why 19, not the full Top 40:** three candidates were excluded after a data validation step (`check_ticker_availability()`) revealed Bid Corporation (BID.JO) and Old Mutual (OMU.JO) only have history from 2016/2018 respectively (post spin-off/restructuring), and MultiChoice (MCG.JO) returned no data from Yahoo Finance at all.

## Methodology

**Returns:** Daily log returns (`ln(P_t / P_t-1)`), used for their time-additive property.

**Currency handling:** Yahoo Finance reports JSE equity prices in South African cents (ZAc), not Rand — converted explicitly via `cents_to_rand()`. This has zero effect on returns/volatility (ratios cancel the /100 out); it only matters for displayed price levels.

**Data quality — price glitch detection:** Initial analysis showed an implausible excess kurtosis of 274.5, which triggered an investigation (`diagnose_outliers.py`) rather than being reported as a finding. It traced to six single-day price errors across three tickers (SLM.JO, VOD.JO, SBK.JO) in 2025, each showing a ~100x single-day price ratio — the signature of a cents/Rand unit error at the data source, not a real market move. A detection function (`detect_price_glitches()`, flagging any day-over-day ratio beyond 10x) and correction function (`clean_price_glitches()`, linear interpolation from neighboring valid prices) now catch and repair these automatically, with every correction printed for auditability.

**Historical Simulation VaR/ES:** Makes no assumption about the shape of the return distribution — uses the empirical distribution of past returns directly.

**Parametric (Variance-Covariance) VaR/ES:** Assumes returns are Normally distributed, characterized by mean and standard deviation alone. Portfolio volatility is verified two independent ways — directly from the combined return series, and via the covariance matrix (`w' Σ w`) — confirmed to match exactly.

**Monte Carlo VaR/ES:** Two variants. *Gaussian Monte Carlo* simulates 100,000 correlated daily scenarios via Cholesky decomposition of the covariance matrix, cross-validating Parametric. *Bootstrap Monte Carlo* resamples actual historical days with replacement, cross-validating Historical.

**Kupiec Proportion-of-Failures backtesting:** Rather than comparing methods only to each other, this tests whether each method's VaR would have actually held up against real historical outcomes. Uses a rolling 250-trading-day window — VaR on any given day is estimated ONLY from the preceding 250 days, never from future data, avoiding the look-ahead bias of testing a model against the same sample it was fit on. Counts "breaches" (days where the actual loss exceeded the VaR threshold) and tests via the Kupiec likelihood-ratio statistic whether the observed breach rate is statistically consistent with the model's stated confidence level.

**Historical scenario stress testing:** Re-applies actual historical crisis returns to the current portfolio composition and weights, answering "if this exact event happened again today, what would we lose?" — a complementary approach to VaR/ES, which describe typical tail risk rather than specific named events.

## Key findings

**1. A data-quality bug initially made tail risk look far more extreme than it was.** Before cleaning, excess kurtosis measured 274.5 — implausible for a diversified 19-stock equity portfolio. Investigating rather than reporting this number traced it to six single-day data errors (cents/Rand unit mixups at the source for three tickers in 2025). After correction, excess kurtosis dropped to a realistic **6.29**, and annualized volatility fell from 30.64% to **20.98%**. Verifying data quality before trusting a risk number is not optional in this field.

**2. Diversification reduces volatility meaningfully**, consistent with standard theory — expanding from 5 to 19 JSE stocks cut annualized volatility and VaR/ES substantially.

**3. Real fat tails persist after cleaning, and Expected Shortfall reveals them better than VaR alone.**

| Metric | Historical | Parametric (Normal) | Gap |
|---|---|---|---|
| 95% VaR | R1,989.69 | R2,161.24 | -7.9% |
| 95% ES | R3,072.47 | R2,713.62 | **+13.2%** |
| 99% VaR | R3,518.86 | R3,062.13 | +14.9% |
| 99% ES | R5,209.94 | R3,510.09 | **+48.4%** |

**4. Four independent methods confirm the fat-tail effect is structural, not a calculation quirk.** Gaussian Monte Carlo matched Parametric within 0.35%; Bootstrap Monte Carlo matched Historical within 0.16% — validating both engines, while the ~15%/~49% gap between the two "camps" (Normal-assumption vs. empirical-distribution methods) persisted.

**5. Backtesting formally proves the Parametric model fails exactly where it matters most.** Using a rolling 250-day window across 2,668 out-of-sample observations, Kupiec's Proportion-of-Failures test **could not reject** Historical VaR at either confidence level (95%: p=0.31; 99%: p=0.32) — its breach rates tracked expectations closely. Parametric VaR passed at 95% confidence (p=0.45) but was **formally rejected at 99% confidence (p=0.0035)**: it breached 43 times against an expected 26.7, a rate 60% higher than its stated confidence level implies. 99% is the confidence level most relevant to real capital requirements (the Basel standard), making this one of the most consequential findings in the project — the Normal-distribution assumption isn't just theoretically imperfect here, it's statistically proven to understate real risk at the threshold that matters most.

## Stress Testing

Rather than asking "what could happen" or "what does the model predict," stress testing re-applies actual historical crisis returns to the current portfolio: "if this exact event happened again today, what would we lose?" Two scenarios were chosen deliberately — one global, one South Africa-specific — since a JSE portfolio faces both.

**COVID-19 Crash (Feb 20 – Mar 23, 2020):** -43.59% cumulative loss (R43,586 on R100,000). Every one of the 19 holdings fell on the worst single day (Mar 23), led by Sasol (-21.5%) and MTN (-19.7%). A textbook systemic shock — diversification within a single market offers no protection when the whole market moves together.

**Nenegate (Dec 8–11, 2015):** -11.41% cumulative loss over just 4 trading days. On the worst day, every major bank was hit hard (FirstRand -16.1%, Absa -15.7%, Standard Bank -14.6%) — but Sasol *gained* +8.2%. Sasol earns substantial revenue in US dollars, so a rand collapse triggered by political crisis acts as a currency tailwind for it, even as rand-funded banks were crushed by the same event. A real, explainable natural hedge inside the portfolio.

**Key insight:** the two scenarios reveal fundamentally different risk structures. Global systemic shocks (COVID) move every holding in the same direction — no diversification benefit exists, consistent with the earlier backtesting finding that tail risk survives diversification. Local political/currency shocks (Nenegate) show genuine within-portfolio dispersion, where sector composition itself provides a hedge. Diversification's value is shock-dependent, not universal.

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
python src/var_monte_carlo.py   # compute Monte Carlo VaR & ES (Gaussian + Bootstrap), cross-validate
python src/backtest.py          # Kupiec POF backtesting: rolling-window validation against real outcomes
python src/stress_test.py       # historical stress scenarios: COVID-19 crash, Nenegate
python src/visualize.py         # generate the P&L distribution chart
python src/diagnose_outliers.py # investigate extreme single-day returns
```

## Known limitations

- Assumes daily rebalancing to fixed target weights
- Equal weighting is a simplifying starting assumption, not a risk-optimized allocation
- All 19 holdings are JSE equities — no cross-asset-class diversification
- Backtesting uses a fixed 250-day rolling window; results could be sensitive to window length (not tested here)
- Monte Carlo here is largely a validation exercise; it becomes essential once a portfolio includes non-linear instruments (options, etc.)
- Stress scenarios re-apply the SAME portfolio weights that existed throughout 2015–present; a portfolio actually held only during 2015 wouldn't have included several of these 19 tickers yet, so this is a hypothetical "if held today" test, not a literal historical reconstruction

## Possible extensions

- GARCH volatility modeling, to test whether time-varying (rather than fixed-window) volatility estimation closes the Parametric gap
- Risk-optimized (rather than equal) portfolio weighting
- Additional stress scenarios (e.g. 2018 EM selloff, load-shedding-driven periods)