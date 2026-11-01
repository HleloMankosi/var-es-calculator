"""
backtest_extended.py

Extends the Kupiec backtest in backtest.py with:

1. WINDOW SENSITIVITY: does the Historical vs Parametric verdict depend on
   how many past days are used to estimate each day's VaR?
2. TIME-VARYING VOLATILITY: two models that react quickly to volatility
   regimes instead of treating the last N days equally:
     - EWMA-Normal: Normal VaR/ES with an exponentially weighted volatility
       (RiskMetrics, lambda = 0.94).
     - EWMA-FHS: Filtered Historical Simulation. Past returns are divided by
       the volatility forecast of their own day, the empirical quantile of
       those standardised returns is kept (so fat tails survive), and it is
       rescaled by today's volatility forecast.
3. CHRISTOFFERSEN TESTS: Kupiec only counts breaches. Independence tests
   whether breaches cluster in time (a sign the model reacts too slowly to
   volatility), and conditional coverage combines both tests.
4. ES BACKTEST: on breach days, is the realised loss larger than the model's
   predicted Expected Shortfall? (McNeil-Frey style exceedance test.)
5. BASEL TRAFFIC LIGHT: breach count over 250 days at 99% VaR classed as
   green (0-4), yellow (5-9) or red (10+).

All methods are compared on the SAME evaluation days, so the numbers are
directly comparable. That sample starts a little later than backtest.py
(EWMA needs a short burn-in), so counts differ slightly from backtest.py.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import xlogy
from scipy.stats import chi2, norm, ttest_1samp

from backtest import kupiec_pof_test

WINDOW = 250
EWMA_LAMBDA = 0.94
EWMA_BURN = 60  # days used to seed the EWMA before its forecasts are used
METHODS = ["Historical", "Parametric", "EWMA-Normal", "EWMA-FHS"]


def ewma_sigma(r, lam=EWMA_LAMBDA, burn=EWMA_BURN):
    """
    One-step-ahead EWMA volatility forecast. sigma[t] is the forecast for
    day t using only returns up to day t-1 (no look-ahead). NaN before `burn`.
    """
    n = len(r)
    sigma = np.full(n, np.nan)
    var = r[:burn].var(ddof=1)
    for t in range(burn, n):
        sigma[t] = np.sqrt(var)
        var = lam * var + (1 - lam) * r[t] ** 2  # update AFTER day t is observed
    return sigma


def rolling_forecasts(r, confidence, window=WINDOW, lam=EWMA_LAMBDA, burn=EWMA_BURN):
    """
    For every evaluation day t (from burn + window onward), forecasts VaR and
    ES for all four methods using only data before day t. VaR/ES are returned
    as RETURNS (negative numbers), not Rand losses.
    """
    alpha = 1 - confidence
    z_alpha = norm.ppf(alpha)
    tail_factor = norm.pdf(z_alpha) / alpha  # Normal ES = mu - sigma * tail_factor
    n = len(r)
    start = burn + window

    sigma_ewma = ewma_sigma(r, lam, burn)
    z_all = r / sigma_ewma  # standardised returns; NaN before `burn`

    var_ret = {m: [] for m in METHODS}
    es_ret = {m: [] for m in METHODS}
    scale = []  # predictable scale used to standardise ES exceedances

    for t in range(start, n):
        past = r[t - window:t]
        s_t = sigma_ewma[t]
        mu, sd = past.mean(), past.std(ddof=1)
        scale.append(sd)

        q = np.percentile(past, alpha * 100)
        var_ret["Historical"].append(q)
        es_ret["Historical"].append(past[past <= q].mean())

        var_ret["Parametric"].append(mu + z_alpha * sd)
        es_ret["Parametric"].append(mu - sd * tail_factor)

        var_ret["EWMA-Normal"].append(z_alpha * s_t)
        es_ret["EWMA-Normal"].append(-s_t * tail_factor)

        zw = z_all[t - window:t]
        qz = np.percentile(zw, alpha * 100)
        var_ret["EWMA-FHS"].append(s_t * qz)
        es_ret["EWMA-FHS"].append(s_t * zw[zw <= qz].mean())

    forecasts = {m: (np.array(var_ret[m]), np.array(es_ret[m])) for m in METHODS}
    return forecasts, r[start:], np.array(scale), start


def window_sensitivity(r, windows=(125, 250, 500, 750), confidences=(0.95, 0.99)):
    """Kupiec results for Historical and Parametric VaR across window lengths."""
    rows = []
    for confidence in confidences:
        alpha = 1 - confidence
        z_alpha = norm.ppf(alpha)
        for w in windows:
            # each row = the w days BEFORE a forecast day (no look-ahead)
            wins = sliding_window_view(r, w)[:-1]
            actual = r[w:]
            hist = np.percentile(wins, alpha * 100, axis=1)
            param = wins.mean(axis=1) + z_alpha * wins.std(axis=1, ddof=1)
            rows.append((confidence, w,
                         kupiec_pof_test(actual < hist, confidence),
                         kupiec_pof_test(actual < param, confidence)))
    return rows


def christoffersen_tests(breaches, confidence):
    """
    Kupiec (unconditional coverage), Christoffersen independence, and their
    sum (conditional coverage, chi-squared with 2 degrees of freedom).
    """
    b = np.asarray(breaches, dtype=int)
    prev, curr = b[:-1], b[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    n0, n1 = n00 + n01, n10 + n11

    kup = kupiec_pof_test(b.astype(bool), confidence)
    result = {"p_kupiec": kup["p_value"], "p_ind": np.nan, "p_cc": np.nan}

    if n0 == 0 or n1 == 0 or (n01 + n11) == 0:
        return result  # too few breaches to test independence

    pi01, pi11 = n01 / n0, n11 / n1
    pi = (n01 + n11) / (n0 + n1)
    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = (xlogy(n00, 1 - pi01) + xlogy(n01, pi01)
              + xlogy(n10, 1 - pi11) + xlogy(n11, pi11))
    lr_ind = max(0.0, -2 * (ll_null - ll_alt))
    result["p_ind"] = chi2.sf(lr_ind, 1)

    if not np.isnan(kup["lr_stat"]):
        result["p_cc"] = chi2.sf(kup["lr_stat"] + lr_ind, 2)
    return result


def es_exceedance_test(actual, var_ret, es_ret, scale):
    """
    On breach days, compares the realised loss with the model's predicted ES,
    in units of a predictable scale (rolling standard deviation). If ES is
    right, the average difference is zero; a significantly POSITIVE average
    means ES understates the real tail losses.
    """
    breach = actual < var_ret
    k = int(breach.sum())
    if k < 5:
        return {"n": k, "p_value": np.nan, "ratio": np.nan}
    loss = -actual[breach]
    predicted = -es_ret[breach]
    resid = (loss - predicted) / scale[breach]
    p_value = ttest_1samp(resid, 0.0, alternative="greater").pvalue
    return {"n": k, "p_value": p_value, "ratio": loss.mean() / predicted.mean()}


def basel_traffic_light(breaches, window=250):
    """Basel zones for 99% VaR over 250 days: green 0-4, yellow 5-9, red 10+."""
    b = np.asarray(breaches, dtype=int)
    counts = np.convolve(b, np.ones(window, dtype=int), mode="valid")

    def zone(c):
        return "green" if c <= 4 else ("yellow" if c <= 9 else "RED")

    return {
        "latest_count": int(counts[-1]),
        "latest_zone": zone(counts[-1]),
        "max_count": int(counts.max()),
        "pct_green": (counts <= 4).mean(),
        "pct_yellow": ((counts >= 5) & (counts <= 9)).mean(),
        "pct_red": (counts >= 10).mean(),
    }


def _p(p):
    """Format a p-value; a trailing * marks rejection at the 5% level."""
    if np.isnan(p):
        return "  n/a "
    return f"{p:.3f}{'*' if p < 0.05 else ' '}"


if __name__ == "__main__":
    from data_loader import load_portfolio_returns

    prices, asset_returns, portfolio_returns = load_portfolio_returns()
    r = portfolio_returns.values
    dates = portfolio_returns.index

    print(f"\n{'='*78}\n  1. WINDOW SENSITIVITY (Kupiec p-values; * = rejected at 5%)\n{'='*78}")
    print(f"{'Conf':<6}{'Window':<8}{'Obs':<7}{'Expected':<10}"
          f"{'Hist breaches (p)':<24}{'Param breaches (p)':<24}")
    for confidence, w, kh, kp in window_sensitivity(r):
        print(f"{confidence:<6.0%}{w:<8}{kh['n_obs']:<7}{kh['expected_breaches']:<10.1f}"
              f"{kh['n_breaches']:>4} ({_p(kh['p_value'])})        "
              f"{kp['n_breaches']:>4} ({_p(kp['p_value'])})")

    for confidence in (0.95, 0.99):
        forecasts, actual, scale, start = rolling_forecasts(r, confidence)
        print(f"\n{'='*78}\n  2. MODEL COMPARISON AT {confidence:.0%} CONFIDENCE "
              f"(window {WINDOW}, same {len(actual)} days: "
              f"{dates[start].date()} to {dates[-1].date()})\n{'='*78}")
        print(f"{'Method':<13}{'Breaches':<16}{'Kupiec':<9}{'Indep.':<9}{'CondCov':<9}"
              f"{'ES test':<9}{'Realised/Pred ES':<16}")
        for m in METHODS:
            var_ret, es_ret = forecasts[m]
            breaches = actual < var_ret
            tests = christoffersen_tests(breaches, confidence)
            es_test = es_exceedance_test(actual, var_ret, es_ret, scale)
            expected = len(actual) * (1 - confidence)
            print(f"{m:<13}{int(breaches.sum()):>4} / {expected:<8.1f}"
                  f"{_p(tests['p_kupiec']):<9}{_p(tests['p_ind']):<9}{_p(tests['p_cc']):<9}"
                  f"{_p(es_test['p_value']):<9}{es_test['ratio']:<16.2f}")

        if confidence == 0.99:
            print(f"\n  BASEL TRAFFIC LIGHT (99% VaR, rolling 250-day breach counts)")
            print(f"{'Method':<13}{'Latest 250d':<14}{'Worst 250d':<12}"
                  f"{'% green':<10}{'% yellow':<10}{'% red':<8}")
            for m in METHODS:
                tl = basel_traffic_light(actual < forecasts[m][0])
                print(f"{m:<13}{tl['latest_count']:>3} ({tl['latest_zone']:<6}) "
                      f"{tl['max_count']:>6}      "
                      f"{tl['pct_green']:<10.0%}{tl['pct_yellow']:<10.0%}{tl['pct_red']:<8.0%}")

    print("\nTests: Kupiec = right number of breaches; Indep. = breaches not clustered;")
    print("CondCov = both; ES test = realised tail loss not above predicted ES;")
    print("Realised/Pred ES > 1 means ES understated the average breach-day loss.")