"""
visualize.py

Plots the portfolio's daily return distribution (in Rand P&L terms) with
VaR and ES thresholds marked at both 95% and 99% confidence.
"""

import matplotlib
matplotlib.use("Agg")  # renders to file without needing a display
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import load_portfolio_returns, PORTFOLIO_VALUE
from var_historical import historical_var, historical_es

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_return_distribution(returns, portfolio_value=PORTFOLIO_VALUE, save=True):
    """
    Histogram of daily portfolio P&L with 95% and 99% VaR/ES thresholds
    drawn as vertical lines.
    """
    var_95 = historical_var(returns, confidence=0.95, portfolio_value=portfolio_value)
    es_95 = historical_es(returns, confidence=0.95, portfolio_value=portfolio_value)
    var_99 = historical_var(returns, confidence=0.99, portfolio_value=portfolio_value)
    es_99 = historical_es(returns, confidence=0.99, portfolio_value=portfolio_value)

    pnl = returns * portfolio_value

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pnl, bins=80, color="#4C72B0", alpha=0.75, edgecolor="white", linewidth=0.3)

    ax.axvline(-var_95, color="#DD8452", linestyle="--", linewidth=2,
               label=f"95% VaR: R{var_95:,.0f}")
    ax.axvline(-es_95, color="#C44E52", linestyle="--", linewidth=2,
               label=f"95% ES: R{es_95:,.0f}")
    ax.axvline(-var_99, color="#937860", linestyle=":", linewidth=2,
               label=f"99% VaR: R{var_99:,.0f}")
    ax.axvline(-es_99, color="#8172B2", linestyle=":", linewidth=2,
               label=f"99% ES: R{es_99:,.0f}")

    ax.set_title("Daily Portfolio P&L Distribution — Historical Simulation (19-Stock JSE Portfolio, Cleaned Data)", fontsize=11)
    ax.set_xlabel("Daily P&L (R)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()

    if save:
        out_path = OUTPUT_DIR