# Portfolio VaR & Expected Shortfall Calculator

A quantitative risk management tool that calculates, cross-validates, and backtests Value at Risk (VaR) and Expected Shortfall (ES) on a JSE equity portfolio using Historical Simulation, Parametric (Variance-Covariance), and Monte Carlo methods, with historical scenario stress testing.

Built to understand not just *how* these models work, but *where each one breaks down* — including catching a real data-quality issue, showing statistically that one method's assumptions fail where it matters most, and showing that diversification's value depends on the type of shock a portfolio actually faces.

## Status

- [x] Data pipeline — JSE price fetching via yfinance with a refreshing cache, ticker validation, cents (ZAc) → Rand conversion, price-glitch detection & exact correction, simple return calculation
- [x] Historical Simulation VaR & Expected Shortfall
- [x] Parametric (Variance-Covariance) VaR & Expected Shortfall
- [x] Monte Carlo VaR & ES (Gaussian and Bootstrap variants)
- [x] Return distribution chart with VaR/ES thresholds
- [x] Kupiec Proportion-of-Failures backtesting
- [x] Historical scenario stress testing (COVID-19, Nenegate)

## Portfolio

An equal-weighted basket of 19 JSE Top 40-derived stocks, spanning banking, telecoms, retail, resources, and healthcare:

`NPN.JO, FSR.JO, SBK.JO, MTN.JO, SOL.JO, CPI.JO, VOD.JO, ABG.JO, SLM.JO, DSY.JO, SHP.JO, NED.JO, BVT.JO, APN.JO, GRT.JO, CLS.JO, EXX.JO, WHL.JO, MRP.JO`

**Why 19, not the full Top 40:** three candidates were excluded after a data validation step (`check_ticker_availability()`) revealed Bid Corporation (BID.JO) and Old Mutual (OMU.JO) only have history from 2016/2018 respectively (post spin-off/restructuring), and MultiChoice (MCG.JO) returned no data from Yahoo Finance at all.

## Methodology

**Returns:** Daily *simple* returns (`P_t / P_t-1 - 1`). The portfolio's daily return is the weighted sum of its stocks' simple returns, which is exactly what a portfolio rebalanced daily to target weights earns. Log returns do not aggregate across assets this way: an earlier version that built the portfolio from log returns overstated the 99% ES by about 5% and the COVID stress loss by about 2 percentage points. Log returns are kept only for diagnostics.

**Currency handling:** Yahoo Finance reports JSE equity prices in South African cents (ZAc), not Rand — converted explicitly via `cents_to_rand()`. This has zero effect on returns/volatility (ratios cancel the /100 out); it only matters for displayed price levels and any future share-count calculation.

**Data quality — price glitch detection:** Initial analysis showed an implausible excess kurtosis of 274.5, which triggered an investigation (`diagnose_outliers.py`) rather than being reported as a finding. It traced to three single-day price errors (SLM.JO and VOD.JO on 2025-01-10, SBK.JO on 2025-04-25), each a price at 1/100 of its true level — the signature of a cents/Rand unit error at the data source, not a real market move. `detect_price_glitches()` flags a day-over-day move beyond 10x that reverses on the next day, so only the bad day is flagged. `clean_price_glitches()` then rescales that day by 100 (an exact repair, not an interpolation), prints every correction for auditability, and warns if any extreme move remains unresolved. (An earlier version also flagged the correct following day and overwrote it with an interpolated price; this was found in review and fixed.)

**Data refresh:** Prices are cached for 24 hours and then re-downloaded; if a refresh fails, the stale cache is used with a warning. A second warning appears if the newest price is more than 5 business days old. Columns are forced into `TICKERS` order, since yfinance returns them alphabetically and any non-equal weights would otherwise be misassigned.

**Historical Simulation VaR/ES:** Makes no assumption about the shape of the return distribution — uses the empirical distribution of past returns directly.

**Parametric (Variance-Covariance) VaR/ES:** Assumes returns are Normally distributed, characterized by mean and standard deviation alone. Portfolio volatility is verified two independent ways — directly from the combined return series, and via the covariance matrix (`w' Σ w`) — and the two match.

**Monte Carlo VaR/ES:** Two variants. *Gaussian Monte Carlo* simulates 100,000 correlated daily scenarios via Cholesky decomposition of the covariance matrix, cross-checking the Parametric implementation. *Bootstrap Monte Carlo* resamples actual historical days with replacement, cross-checking the Historical implementation.

**Kupiec Proportion-of-Failures backtesting:** Rather than comparing methods only to each other, this tests whether each method's VaR would have held up against real outcomes. It uses a rolling 250-trading-day window — VaR on any given day is estimated ONLY from the preceding 250 days, never from future data, avoiding the look-ahead bias of testing a model on the sample it was fit on. It counts "breaches" (days where the actual loss exceeded the VaR threshold) and tests via the Kupiec likelihood-ratio statistic whether the observed breach rate is statistically consistent with the model's stated confidence level.

**Historical scenario stress testing:** Re-applies actual historical crisis returns to the current portfolio composition and weights, compounding the daily portfolio returns (`prod(1 + r) - 1`, consistent with daily rebalancing) — answering "if this exact event happened again today, what would we lose?"

## Key findings

*Figures are from the 30 October 2026 run (2,986 trading days from 1 January 2015, R100,000 portfolio) and shift slightly as the data refreshes.*

**1. A data-quality bug initially made tail risk look far more extreme than it was.** Before cleaning, excess kurtosis measured 274.5 — implausible for a diversified 19-stock equity portfolio. Investigating rather than reporting this number traced it to three single-day cents/Rand unit errors in 2025. After correction, excess kurtosis fell to a realistic **5.72** and annualized volatility is **20.78%**. Verifying data quality before trusting a risk number is not optional in this field.

**2. Real fat tails persist after cleaning, and the Normal model's error changes sign with the confidence level.**

| Metric | Historical | Parametric (Normal) | Gap |
|---|---|---|---|
| 95% VaR | R1,929.27 | R2,116.46 | -8.8% |
| 95% ES | R2,951.59 | R2,663.40 | **+10.8%** |
| 99% VaR | R3,393.38 | R3,008.47 | +12.8% |
| 99% ES | R4,959.32 | R3,452.01 | **+43.7%** |

(Gap = Historical ÷ Parametric − 1, both as Rand losses. Positive means the Normal model understates the real loss.) Against a Normal curve with the same standard deviation, a fat-tailed distribution has fewer moderate bad days and more extreme ones, so the Normal model overstates the 95% VaR but understates the 99% VaR and the ES, which averages the most extreme days. Expected Shortfall exposes the fat tails more clearly than VaR does.

**3. The cross-checks validate the implementations.** There are two modelling families here (Normal-assumption and empirical), each implemented twice. Gaussian Monte Carlo matched Parametric within 0.12% on VaR (0.65% on 99% ES); Bootstrap Monte Carlo matched Historical within 0.73% on VaR (up to 2.3% on 99% ES, as expected from sampling noise in the extreme tail). The ~13%/~44% gap between the two families persisted, so the fat-tail effect is a property of the data, not a calculation quirk.

**4. Backtesting shows the Parametric model under-covers at 99%.** Using a rolling 250-day window across 2,735 out-of-sample observations, Kupiec's test could not reject Historical VaR at either confidence level (95%: 149 breaches vs 136.8 expected, p = 0.29; 99%: 32 vs 27.4, p = 0.38). Parametric VaR passed at 95% (127 breaches, p = 0.39) but was **rejected at 99% (p = 0.0091)**: it breached 42 times against 27.4 expected, about 54% more often than its stated confidence implies. 99% VaR is the long-standing Basel market-risk standard (the newer FRTB framework moved to 97.5% ES), so this is the tail that matters for capital. This is statistical evidence of under-coverage, not a proof; the result may depend on window length, which has not yet been tested.

## Stress Testing

Rather than asking "what could happen" or "what does the model predict," stress testing re-applies actual historical crisis returns to the current portfolio: "if this exact event happened again today, what would we lose?" Two scenarios were chosen deliberately — one global, one South Africa-specific — since a JSE portfolio faces both.

**COVID-19 Crash (Feb 20 – Mar 23, 2020):** -41.53% cumulative loss over 23 trading days (R41,534 on R100,000). Every one of the 19 holdings fell on the worst single day (Mar 23; portfolio -9.13%, R9,135), led by Sasol (-19.4%) and MTN (-17.9%). A textbook systemic shock — diversification within a single market offers little protection when the whole market moves together.

**Nenegate (Dec 8–11, 2015):** -11.17% cumulative loss over just 4 trading days (R11,171). On the worst day (Dec 10; portfolio -6.80%), every major bank was hit hard (FirstRand -14.8%, Absa -14.5%, Standard Bank -13.5%, Nedbank -10.6%) — but Sasol *gained* +8.5% and Naspers +2.4%. Both earn substantial revenue outside South Africa, so a rand collapse triggered by political crisis acts as a currency tailwind for them, even as rand-funded banks were crushed by the same event. A real, explainable natural hedge inside the portfolio.

**Key insight:** the two scenarios reveal fundamentally different risk structures. Global systemic shocks (COVID) move every holding in the same direction — little diversification benefit exists, consistent with tail risk surviving diversification. Local political/currency shocks (Nenegate) show genuine within-portfolio dispersion, where sector composition itself provides a hedge. Diversification's value is shock-dependent, not universal.

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
python src/diagnose_outliers.py # investigate extreme single-day returns (uncleaned data)
```

## Known limitations

- Assumes daily rebalancing to fixed equal weights, with no transaction costs, brokerage or taxes
- Survivorship bias: the 19 holdings are today's survivors with full history since 2015, so any return statistic from this history is flattered
- All 19 holdings are JSE equities — no cross-asset-class diversification (bonds, offshore exposure)
- Historical VaR is bounded by the 2015–present sample window
- Backtesting uses a fixed 250-day rolling window; sensitivity to window length is not yet tested
- Kupiec tests only how many breaches occurred, not whether they cluster in time, and does not test ES
- 99% ES rests on about 30 tail observations and is reported without confidence intervals
- Monte Carlo here is largely a validation exercise; it becomes essential once a portfolio includes non-linear instruments (options, etc.)
- Stress scenarios apply today's weights to past returns: a hypothetical "if held today" test, not a literal historical reconstruction
- Price data comes from Yahoo Finance via yfinance, an unofficial source

## Roadmap

- Time-varying volatility (EWMA / GARCH / filtered historical simulation) to test whether it closes the Parametric gap
- Christoffersen independence test and an ES backtest; window-length sensitivity; Basel traffic-light classification
- Bootstrap confidence intervals for VaR/ES
- Re-measure the 5-stock vs 19-stock diversification effect on cleaned data
- A diversifying asset (bond, money-market or offshore ETF) and risk-optimized weights
- Live-use layer: holdings input, share counts, weight drift, rebalance rule with costs, and a weekly VaR/ES monitoring report
- Additional stress scenarios (e.g. 2018 EM selloff, load-shedding-driven periods)