"""
stress_test.py

Historical Scenario Stress Testing.

Rather than asking "what could happen" (Monte Carlo) or "what does the
statistical model predict" (Parametric), stress testing asks a more direct
question: "if a specific real crisis happened again today, with today's
portfolio, how much would we lose?" This re-applies actual historical
return shocks to the CURRENT portfolio composition and weights, rather
than just reporting what happened at the time.

Two scenarios are used, chosen deliberately: one global (COVID-19 crash)
and one South Africa-specific (Nenegate) -- a JSE portfolio faces both
international and local political/currency risk, and a single global
stress scenario alone would miss the local dimension entirely.
"""

SCENARIOS = {
    "COVID-19 Crash (2020)": {
        "start": "2020-02-20",
        "end": "2020-03-23",
        "description": (
            "Global equity selloff triggered by the COVID-19 pandemic. "
            "Feb 20, 2020 is the widely-cited pre-crash market peak; "
            "Mar 23, 2020 is the widely-cited market bottom."
        ),
    },
    "Nenegate (Dec 2015)": {
        "start": "2015-12-08",
        "end": "2015-12-11",
        "description": (
            "South Africa-specific political/currency crisis: President "
            "Zuma's removal of Finance Minister Nhlanhla Nene on Dec 9, "
            "2015 triggered a rand collapse and a JSE banking-sector "
            "selloff (JSE banking index fell 13.5% on Dec 10 alone)."
        ),
    },
}


def run_scenario(portfolio_returns, asset_returns, start, end, portfolio_value=100_000):
    """
    Extracts actual historical returns within [start, end] and applies them
    to the CURRENT portfolio (same weights used throughout this project),
    answering: "if this exact sequence of returns happened again today,
    starting from today's portfolio value, what would happen?"
    """
    mask = (portfolio_returns.index >= start) & (portfolio_returns.index <= end)
    window_returns = portfolio_returns[mask]

    if len(window_returns) == 0:
        raise ValueError(f"No trading days found between {start} and {end}. Check the date range.")

    # The portfolio is rebalanced daily to its target weights, so its value
    # compounds the daily SIMPLE portfolio returns: prod(1 + r_t) - 1.
    cumulative_simple_return = (1 + window_returns).prod() - 1
    cumulative_loss = -cumulative_simple_return * portfolio_value

    worst_day_return = window_returns.min()
    worst_day_date = window_returns.idxmin()
    worst_day_loss = -worst_day_return * portfolio_value

    # Per-ticker breakdown for the single worst day, so we can see WHICH
    # holdings drove the loss, not just the aggregate number -- the same
    # diagnostic approach used in diagnose_outliers.py.
    worst_day_ticker_returns = asset_returns.loc[worst_day_date].sort_values()

    return {
        "start": start,
        "end": end,
        "n_trading_days": len(window_returns),
        "cumulative_return": cumulative_simple_return,
        "cumulative_loss": cumulative_loss,
        "worst_day_date": worst_day_date,
        "worst_day_return": worst_day_return,
        "worst_day_loss": worst_day_loss,
        "worst_day_ticker_returns": worst_day_ticker_returns,
    }


if __name__ == "__main__":
    from data_loader import load_portfolio_returns, PORTFOLIO_VALUE

    prices, asset_returns, portfolio_returns = load_portfolio_returns()

    print(f"Portfolio value: R{PORTFOLIO_VALUE:,.0f}\n")

    for scenario_name, scenario in SCENARIOS.items():
        print(f"{'='*70}")
        print(f"  SCENARIO: {scenario_name}")
        print(f"{'='*70}")
        print(f"{scenario['description']}\n")

        result = run_scenario(
            portfolio_returns, asset_returns,
            scenario["start"], scenario["end"],
            portfolio_value=PORTFOLIO_VALUE,
        )

        print(f"Trading days in window: {result['n_trading_days']}")
        print(f"Cumulative return over period: {result['cumulative_return']:+.2%}")
        print(f"Cumulative loss on R{PORTFOLIO_VALUE:,.0f}: R{result['cumulative_loss']:,.2f}")
        print(f"\nWorst single day: {result['worst_day_date'].date()} ({result['worst_day_return']:+.2%})")
        print(f"Worst single-day loss: R{result['worst_day_loss']:,.2f}")
        print(f"\nPer-stock returns on worst day (sorted, worst first):")
        print(result["worst_day_ticker_returns"].apply(lambda x: f"{x:+.2%}").to_string())
        print()