"""
var_parametric.py

Parametric (Variance-Covariance) VaR and Expected Shortfall.

Assumes portfolio returns follow a Normal distribution, described entirely
by mean (mu) and standard deviation (sigma). Comparing this against
Historical Simulation quantifies the fat-tail gap directly.
"""

import numpy as np
from scipy.stats import norm, kurtosis


def portfolio_mean_std(log_returns, weights):
    """
    Computes portfolio mean/std TWO ways as a cross-check: directly from
    the combined return series, and via the covariance matrix (w' Sigma w).
    These are mathematically equivalent, so they should match closely.
    """
    portfolio_returns = log_returns @ weights
    mu_direct = portfolio_returns.mean()
    sigma_direct = portfolio_returns.std()

    cov_matrix = log_returns.cov()
    mu_matrix = weights @ log_returns.mean()
    sigma_matrix = np.sqrt(weights @ cov_matrix @ weights)

    return {
        "mu_direct": mu_direct,
        "sigma_direct": sigma_direct,
        "mu_matrix": mu_matrix,
        "sigma_matrix": sigma_matrix,
        "portfolio_returns": portfolio_returns,
    }


def parametric_var(mu, sigma, confidence=0.95, portfolio_value=100_000):
    alpha = 1 - confidence
    z_alpha = norm.ppf(alpha)
    var_return = mu + z_alpha * sigma
    return -var_return * portfolio_value


def parametric_es(mu, sigma, confidence=0.95, portfolio_value=100_000):
    alpha = 1 - confidence
    z_alpha = norm.ppf(alpha)
    es_return = mu - sigma * norm.pdf(z_alpha) / alpha
    return -es_return * portfolio_value


if __name__ == "__main__":
    from data_loader import load_portfolio_returns, WEIGHTS, PORTFOLIO_VALUE
    from var_historical import historical_var, historical_es

    prices, log_returns, portfolio_returns = load_portfolio_returns()

    stats = portfolio_mean_std(log_returns, WEIGHTS)

    print("\nCross-check: direct calculation vs covariance-matrix calculation")
    print(f"  Mean    -> direct: {stats['mu_direct']:.6f}   matrix: {stats['mu_matrix']:.6f}")
    print(f"  Std Dev -> direct: {stats['sigma_direct']:.6f}   matrix: {stats['sigma_matrix']:.6f}")

    mu, sigma = stats["mu_direct"], stats["sigma_direct"]

    excess_kurtosis = kurtosis(portfolio_returns)
    print(f"\nExcess kurtosis of portfolio returns: {excess_kurtosis:.3f}")
    print("(0 = matches Normal distribution; positive = fatter tails than Normal)")

    print(f"\nPortfolio value: R{PORTFOLIO_VALUE:,.0f}\n")
    print(f"{'Confidence':<12}{'Method':<14}{'VaR (R)':<15}{'ES (R)':<15}")
    for conf in [0.95, 0.99]:
        hist_var = historical_var(portfolio_returns, confidence=conf, portfolio_value=PORTFOLIO_VALUE)
        hist_es = historical_es(portfolio_returns, confidence=conf, portfolio_value=PORTFOLIO_VALUE)
        param_var = parametric_var(mu, sigma, confidence=conf, portfolio_value=PORTFOLIO_VALUE)
        param_es = parametric_es(mu, sigma, confidence=conf, portfolio_value=PORTFOLIO_VALUE)

        print(f"{conf:.0%}         {'Historical':<14}R{hist_var:,.2f}      R{hist_es:,.2f}")
        print(f"{conf:.0%}         {'Parametric':<14}R{param_var:,.2f}      R{param_es:,.2f}")

        gap_var = (hist_var / param_var - 1) * 100
        gap_es = (hist_es / param_es - 1) * 100
        print(f"              -> Historical VaR is {gap_var:+.1f}% vs Parametric; ES is {gap_es:+.1f}%\n")