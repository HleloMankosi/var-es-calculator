"""
backtest.py

Kupiec Proportion-of-Failures (POF) backtesting.

Every VaR method so far has only been checked against ITSELF (comparing
Historical vs Parametric vs Monte Carlo to each other). This is different:
it asks whether each method's VaR estimate would have actually held up
against what really happened, day by day -- using a proper ROLLING window
so that VaR on any given day is estimated ONLY from data available before
that day, never from the future. Testing VaR against the same full sample
it was fit on would be circular and make every method look artificially
good.

Only Historical and Parametric are backtested here. Gaussian Monte Carlo
was already shown to converge to Parametric, and Bootstrap Monte Carlo to
Historical (var_monte_carlo.py) -- so backtesting these two methods
effectively covers both "camps" without redundant computation.
"""

import numpy as np
from scipy.stats import chi2, norm


def rolling_historical_var(portfolio_returns, window=250, confidence=0.95):
    """
    For each day (after the first `window` days), estimates Historical VaR
    using only the PRECEDING `window` days of returns, then pairs that
    estimate with the actual return that followed -- exactly the
    information a risk desk would have had in real time, with no hindsight.
    """
    alpha = 1 - confidence
    var_estimates, actual_returns, dates = [], [], []

    for t in range(window, len(portfolio_returns)):
        past_window = portfolio_returns.iloc[t - window:t]
        var_return = np.percentile(past_window, alpha * 100)
        var_estimates.append(var_return)
        actual_returns.append(portfolio_returns.iloc[t])
        dates.append(portfolio_returns.index[t])

    return np.array(var_estimates), np.array(actual_returns), dates


def rolling_parametric_var(portfolio_returns, window=250, confidence=0.95):
    """
    Same rolling-window discipline as above, but using the Parametric
    (Normal distribution) formula instead of the empirical percentile.
    """
    alpha = 1 - confidence
    z_alpha = norm.ppf(alpha)
    var_estimates, actual_returns, dates = [], [], []

    for t in range(window, len(portfolio_returns)):
        past_window = portfolio_returns.iloc[t - window:t]
        mu = past_window.mean()
        sigma = past_window.std()
        var_return = mu + z_alpha * sigma
        var_estimates.append(var_return)
        actual_returns.append(portfolio_returns.iloc[t])
        dates.append(portfolio_returns.index[t])

    return np.array(var_estimates), np.array(actual_returns), dates


def count_breaches(var_estimates, actual_returns):
    """
    A breach occurs when the actual return is WORSE (more negative) than
    the VaR threshold predicted -- the model said "returns shouldn't fall
    below this level more than (1-confidence)% of the time," and reality
    did anyway.
    """
    return actual_returns < var_estimates


def kupiec_pof_test(breaches, confidence=0.95):
    """
    Kupiec's Proportion-of-Failures likelihood ratio test.

    Under a correctly calibrated VaR model, the breach rate should equal
    exactly (1 - confidence) -- e.g. 5% of days should breach a 95% VaR,
    no more, no less. This asks: given N observations and x actual
    breaches, is the observed breach rate statistically distinguishable
    from the expected rate, or within the range you'd see from chance
    alone?

    The test statistic follows a chi-squared distribution with 1 degree of
    freedom under the null hypothesis (model is correctly calibrated). A
    p-value below 0.05 means we REJECT that the model is well-calibrated
    -- at the 95% confidence level of the TEST itself, which is a
    different "95%" from the VaR confidence level being tested, worth not
    conflating the two.
    """
    n = len(breaches)
    x = int(breaches.sum())
    p_expected = 1 - confidence
    p_observed = x / n

    if x == 0 or x == n:
        # Avoid log(0). Zero breaches usually signals an overly
        # conservative model rather than a calculation failure -- flagged
        # explicitly rather than silently skipped.
        return {
            "n_obs": n, "n_breaches": x, "expected_breaches": n * p_expected,
            "breach_rate": p_observed, "expected_rate": p_expected,
            "lr_stat": np.nan, "p_value": np.nan, "reject_null": None,
        }

    log_likelihood_null = (n - x) * np.log(1 - p_expected) + x * np.log(p_expected)
    log_likelihood_observed = (n - x) * np.log(1 - p_observed) + x * np.log(p_observed)
    lr_stat = -2 * (log_likelihood_null - log_likelihood_observed)

    p_value = 1 - chi2.cdf(lr_stat, df=1)
    reject_null = p_value < 0.05

    return {
        "n_obs": n, "n_breaches": x, "expected_breaches": n * p_expected,
        "breach_rate": p_observed, "expected_rate": p_expected,
        "lr_stat": lr_stat, "p_value": p_value, "reject_null": reject_null,
    }


if __name__ == "__main__":
    from data_loader import load_portfolio_returns

    prices, log_returns, portfolio_returns = load_portfolio_returns()

    WINDOW = 250  # ~1 trading year of history used to estimate each day's VaR

    print(f"Rolling window: {WINDOW} trading days")
    print(f"Total observations after warm-up period: {len(portfolio_returns) - WINDOW}\n")

    for confidence in [0.95, 0.99]:
        print(f"{'='*70}")
        print(f"  {confidence:.0%} CONFIDENCE LEVEL")
        print(f"{'='*70}")

        for method_name, rolling_fn in [
            ("Historical", rolling_historical_var),
            ("Parametric", rolling_parametric_var),
        ]:
            var_est, actual, dates = rolling_fn(portfolio_returns, window=WINDOW, confidence=confidence)
            breaches = count_breaches(var_est, actual)
            result = kupiec_pof_test(breaches, confidence=confidence)

            print(f"\n{method_name} VaR:")
            print(f"  Observations:        {result['n_obs']}")
            print(f"  Expected breaches:   {result['expected_breaches']:.1f}  ({result['expected_rate']:.1%})")
            print(f"  Actual breaches:     {result['n_breaches']}  ({result['breach_rate']:.1%})")
            if not np.isnan(result['lr_stat']):
                print(f"  Kupiec LR statistic: {result['lr_stat']:.3f}")
                print(f"  p-value:             {result['p_value']:.4f}")
                verdict = "REJECTED (model miscalibrated)" if result['reject_null'] else "NOT rejected (model adequately calibrated)"
                print(f"  Verdict:             {verdict}")
            else:
                print("  Kupiec LR statistic: undefined (zero or all breaches)")
        print()