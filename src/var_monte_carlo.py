"""
var_monte_carlo.py

Monte Carlo VaR and Expected Shortfall.

Simulates many possible future daily portfolio returns rather than relying
solely on actual historical data (Historical Simulation) or a closed-form
formula (Parametric). Two variants are implemented:

1. GAUSSIAN Monte Carlo: simulates correlated asset returns by sampling
   from a multivariate Normal distribution parameterized by the historical
   mean vector and covariance matrix, via Cholesky decomposition. Since a
   linear combination of multivariate Normal variables is itself Normal,
   this should converge to the SAME answer as Parametric VaR as the number
   of simulations grows — a genuine implementation cross-check, not just a
   second opinion.

2. BOOTSTRAP Monte Carlo: simulates by resampling actual historical days
   WITH REPLACEMENT (sampling whole rows, so cross-asset correlations on
   any given day are preserved exactly as they occurred). This lets the
   simulation inherit the real empirical distribution's shape — including
   fat tails — rather than assuming Normality, so it should converge
   toward Historical Simulation's numbers instead of Parametric's.

Monte Carlo becomes NECESSARY rather than just illustrative once a
portfolio includes options or other non-linear instruments, where
Parametric's linear/Normal assumptions break down. This portfolio is all
equities (linear), so here Monte Carlo mainly serves as a cross-check on
the other two methods — worth being upfront about that in a writeup rather
than overselling why it was needed here specifically.
"""

import numpy as np

from var_historical import historical_var, historical_es


def simulate_gaussian(log_returns, weights, n_simulations=100_000, random_seed=42):
    """
    Simulates n_simulations days of correlated asset returns by drawing
    from a multivariate Normal distribution, via Cholesky decomposition.

    Cholesky decomposition: any covariance matrix Sigma can be factored as
    Sigma = L @ L.T, where L is lower-triangular. If Z is a vector of
    INDEPENDENT standard normal draws, then L @ Z has covariance Sigma —
    this is the mechanism that turns uncorrelated randomness into
    correlated randomness matching how the real assets actually co-move.
    """
    rng = np.random.default_rng(random_seed)

    mean_vector = log_returns.mean().values
    cov_matrix = log_returns.cov().values

    L = np.linalg.cholesky(cov_matrix)
    n_assets = len(mean_vector)

    z = rng.standard_normal((n_simulations, n_assets))
    simulated_asset_returns = mean_vector + z @ L.T

    simulated_portfolio_returns = simulated_asset_returns @ weights
    return simulated_portfolio_returns


def simulate_bootstrap(log_returns, weights, n_simulations=100_000, random_seed=42):
    """
    Simulates n_simulations days by resampling ACTUAL historical days with
    replacement. Sampling whole rows (not each ticker independently)
    preserves the real cross-sectional correlation structure — e.g. if
    banks and retailers tend to fall together on a bad day historically,
    that relationship survives in every resampled day too.

    This inherits the real empirical distribution's shape, unlike
    simulate_gaussian(), which assumes Normality by construction.
    """
    rng = np.random.default_rng(random_seed)

    n_days = len(log_returns)
    sampled_indices = rng.integers(0, n_days, size=n_simulations)
    simulated_asset_returns = log_returns.values[sampled_indices]

    simulated_portfolio_returns = simulated_asset_returns @ weights
    return simulated_portfolio_returns


if __name__ == "__main__":
    from data_loader import load_portfolio_returns, WEIGHTS, PORTFOLIO_VALUE
    from var_parametric import portfolio_mean_std, parametric_var, parametric_es

    prices, log_returns, portfolio_returns = load_portfolio_returns()

    gaussian_sim = simulate_gaussian(log_returns, WEIGHTS)
    bootstrap_sim = simulate_bootstrap(log_returns, WEIGHTS)

    stats = portfolio_mean_std(log_returns, WEIGHTS)
    mu, sigma = stats["mu_direct"], stats["sigma_direct"]

    print(f"Portfolio value: R{PORTFOLIO_VALUE:,.0f}")
    print(f"Simulations run: {len(gaussian_sim):,} per method\n")

    print(f"{'Conf':<6}{'Method':<26}{'VaR (R)':<15}{'ES (R)':<15}")
    for conf in [0.95, 0.99]:
        hist_var = historical_var(portfolio_returns, confidence=conf, portfolio_value=PORTFOLIO_VALUE)
        hist_es = historical_es(portfolio_returns, confidence=conf, portfolio_value=PORTFOLIO_VALUE)

        param_var = parametric_var(mu, sigma, confidence=conf, portfolio_value=PORTFOLIO_VALUE)
        param_es = parametric_es(mu, sigma, confidence=conf, portfolio_value=PORTFOLIO_VALUE)

        mc_gauss_var = historical_var(gaussian_sim, confidence=conf, portfolio_value=PORTFOLIO_VALUE)
        mc_gauss_es = historical_es(gaussian_sim, confidence=conf, portfolio_value=PORTFOLIO_VALUE)

        mc_boot_var = historical_var(bootstrap_sim, confidence=conf, portfolio_value=PORTFOLIO_VALUE)
        mc_boot_es = historical_es(bootstrap_sim, confidence=conf, portfolio_value=PORTFOLIO_VALUE)

        print(f"{conf:.0%}    {'Historical':<26}R{hist_var:,.2f}      R{hist_es:,.2f}")
        print(f"{conf:.0%}    {'Parametric':<26}R{param_var:,.2f}      R{param_es:,.2f}")
        print(f"{conf:.0%}    {'Monte Carlo (Gaussian)':<26}R{mc_gauss_var:,.2f}      R{mc_gauss_es:,.2f}")
        print(f"{conf:.0%}    {'Monte Carlo (Bootstrap)':<26}R{mc_boot_var:,.2f}      R{mc_boot_es:,.2f}")

        gauss_vs_param = (mc_gauss_var / param_var - 1) * 100
        boot_vs_hist = (mc_boot_var / hist_var - 1) * 100
        print(f"          -> Gaussian MC vs Parametric: {gauss_vs_param:+.2f}% (should be ~0%, validates simulation)")
        print(f"          -> Bootstrap MC vs Historical: {boot_vs_hist:+.2f}% (should be small, validates resampling)\n")