"""
portfolio_monitor.py

The live-use layer. Turns the risk models into something you can act on:

1. HOLDINGS IN: reads what you own (private/holdings.csv) or, if that file
   does not exist, assumes you are investing a cash amount from scratch.
2. REBALANCE ORDERS OUT: compares your current weights with the target
   weights, and ONLY if drift is large enough (or it is your first
   investment) produces a buy/sell list with units, Rand amounts and
   estimated costs (brokerage + securities transfer tax).
3. RISK REPORT: 1-day VaR/ES for the portfolio you will hold after the
   trades (Historical and EWMA-FHS), checked against a VaR limit.

Run:   python src/portfolio_monitor.py --capital 5000
       python src/portfolio_monitor.py            (uses private/holdings.csv)

private/holdings.csv format (keep this folder OUT of git - it is your money):
       ticker,units
       NPN.JO,1.2345
       SBK.JO,3.5
       CASH,250.00        <- optional row: Rand cash not yet invested

This tool does NOT place orders. It prints a list for you to place manually,
using live prices from your broker. It is not financial advice.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_extended import EWMA_LAMBDA, WINDOW, ewma_sigma
from data_loader import TICKERS, WEIGHTS, load_portfolio_returns

# ---------------- Settings you can change ----------------
PRIVATE_DIR = Path(__file__).resolve().parent.parent / "private"
HOLDINGS_FILE = PRIVATE_DIR / "holdings.csv"
TRADES_FILE = PRIVATE_DIR / "trades.csv"

DEFAULT_CAPITAL = 5_000.0   # Rand to invest if there is no holdings file
ALLOW_FRACTIONAL = True     # EasyEquities offers fractional shares on JSE stocks
FRACTIONAL_DECIMALS = 4     # units are rounded DOWN to this many decimals
DRIFT_BAND = 0.02           # rebalance when any weight is >2 percentage points off target
MIN_TRADE_VALUE = 10.0      # ignore trades smaller than this many Rand

# Cost model. Verify against your broker's current cost profile.
BROKERAGE_RATE = 0.0025     # EasyEquities: 0.25% per trade
BROKERAGE_MIN = 0.01        # EasyEquities: 1c minimum
BROKERAGE_DISCOUNT = 0.0    # e.g. 0.20 if you qualify for a 20% brokerage discount
STT_RATE = 0.0025           # Securities Transfer Tax: 0.25%, paid on PURCHASES only
OTHER_LEVIES_RATE = 0.0     # JSE/Strate/investor-protection levies: set from your cost profile

VAR_LIMIT = 0.04            # max 1-day 99% VaR as a share of portfolio value (your risk appetite)
# ----------------------------------------------------------


def trade_costs(trade_value):
    """Estimated cost in Rand of each trade (positive = buy, negative = sell)."""
    v = np.abs(trade_value)
    brokerage = np.where(
        v > 0,
        np.maximum(v * BROKERAGE_RATE * (1 - BROKERAGE_DISCOUNT), BROKERAGE_MIN),
        0.0,
    )
    stt = np.where(trade_value > 0, trade_value * STT_RATE, 0.0)
    return brokerage + stt + v * OTHER_LEVIES_RATE


def load_holdings(path, tickers):
    """Returns (units array in TICKERS order, cash in Rand), or None if no file."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"ticker", "units"} <= set(df.columns):
        raise ValueError("holdings.csv needs the columns: ticker,units")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    cash = float(df.loc[df["ticker"] == "CASH", "units"].sum())
    stocks = df[df["ticker"] != "CASH"]
    unknown = set(stocks["ticker"]) - set(tickers)
    if unknown:
        raise ValueError(
            f"Tickers not in this portfolio: {sorted(unknown)}. Add them to TICKERS "
            f"in data_loader.py or remove them from holdings.csv."
        )
    units = stocks.groupby("ticker")["units"].sum().reindex(tickers).fillna(0.0)
    return units.values.astype(float), cash


def plan_trades(prices, weights, units, cash, allow_fractional=ALLOW_FRACTIONAL):
    """
    Decides whether to rebalance and, if so, the trades that bring the
    portfolio back to target weights AFTER paying estimated costs. Trades
    are rounded down so you never need more cash than you have.
    """
    step = 10.0 ** -FRACTIONAL_DECIMALS if allow_fractional else 1.0

    def to_units(x):
        return np.floor(x / step + 1e-9) * step

    current_value = units * prices
    total = current_value.sum() + cash
    drift = current_value / total - weights
    first_investment = units.sum() == 0
    triggered = bool(first_investment or np.abs(drift).max() > DRIFT_BAND)

    result = {
        "total": total, "drift": drift, "triggered": triggered,
        "first_investment": bool(first_investment), "step": step,
    }
    if not triggered:
        zeros = np.zeros_like(units)
        result.update(trade_units=zeros, trade_value=zeros, costs=zeros,
                      cash_after=cash, final_units=units.copy())
        return result

    # Costs reduce what can be invested, and costs depend on the trades, so iterate.
    cost_total = 0.0
    for _ in range(20):
        investable = total - cost_total
        target_units = to_units(weights * investable / prices)
        trade_units = target_units - units
        trade_value = trade_units * prices
        small = np.abs(trade_value) < MIN_TRADE_VALUE
        trade_units = np.where(small, 0.0, trade_units)
        trade_value = trade_units * prices
        costs = trade_costs(trade_value)
        converged = abs(costs.sum() - cost_total) < 0.005
        cost_total = costs.sum()
        if converged:
            break

    cash_after = cash - trade_value.sum() - costs.sum()

    # Whole shares leave unspent cash. Spend it, one unit at a time, on the
    # most under-weight stock that is still affordable.
    if not allow_fractional:
        ideal_value = weights * (total - costs.sum())
        unit_cost = prices * step * (
            1 + BROKERAGE_RATE * (1 - BROKERAGE_DISCOUNT) + STT_RATE + OTHER_LEVIES_RATE
        )
        for _ in range(10_000):
            gap = ideal_value - (units + trade_units) * prices
            affordable = (unit_cost <= cash_after) & (gap >= 0.5 * prices * step)
            if not affordable.any():
                break
            i = int(np.argmax(np.where(affordable, gap, -np.inf)))
            trade_units[i] += step
            trade_value = trade_units * prices
            costs = trade_costs(trade_value)
            cash_after = cash - trade_value.sum() - costs.sum()

    # Safety net: never plan to spend more cash than is available.
    guard = 0
    while cash_after < -1e-9 and guard < 100_000:
        i = int(np.argmax(trade_value))
        if trade_value[i] <= 0:
            break
        trade_units[i] = max(0.0, trade_units[i] - step)
        trade_value = trade_units * prices
        costs = trade_costs(trade_value)
        cash_after = cash - trade_value.sum() - costs.sum()
        guard += 1

    result.update(trade_units=trade_units, trade_value=trade_value, costs=costs,
                  cash_after=cash_after, final_units=units + trade_units)
    return result


def risk_summary(asset_returns, weights_vec, value):
    """
    1-day VaR/ES in Rand for a portfolio held at `weights_vec` (fractions of
    `value`; any remainder is cash earning zero). Historical uses the whole
    sample; EWMA-FHS scales the standardised past returns by tomorrow's
    volatility forecast, so it reflects the CURRENT volatility regime.
    """
    p = asset_returns.values @ weights_vec
    out = {}
    for c in (0.95, 0.99):
        q = np.percentile(p, (1 - c) * 100)
        out[f"hist_var_{round(c * 100)}"] = -q * value
        out[f"hist_es_{round(c * 100)}"] = -p[p <= q].mean() * value

    sigma = ewma_sigma(p)
    z = p / sigma
    zw = z[-WINDOW:]
    s_next = np.sqrt(EWMA_LAMBDA * sigma[-1] ** 2 + (1 - EWMA_LAMBDA) * p[-1] ** 2)
    for c in (0.95, 0.99):
        qz = np.percentile(zw, (1 - c) * 100)
        out[f"fhs_var_{round(c * 100)}"] = -s_next * qz * value
        out[f"fhs_es_{round(c * 100)}"] = -s_next * zw[zw <= qz].mean() * value
    out["vol_next_ann"] = s_next * np.sqrt(252)
    return out


def _r(x):
    return f"R{x:,.2f}"


def main():
    parser = argparse.ArgumentParser(description="JSE portfolio rebalance + risk monitor")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL,
                        help="Rand to invest when there is no private/holdings.csv")
    parser.add_argument("--whole-shares", action="store_true",
                        help="round to whole shares instead of fractions")
    args = parser.parse_args()
    allow_fractional = ALLOW_FRACTIONAL and not args.whole_shares

    prices_df, asset_returns, _ = load_portfolio_returns()
    price_date = prices_df.index[-1].date()
    prices = prices_df.iloc[-1][TICKERS].values.astype(float)

    holdings = load_holdings(HOLDINGS_FILE, TICKERS)
    if holdings is None:
        units, cash = np.zeros(len(TICKERS)), float(args.capital)
        print(f"\nNo {HOLDINGS_FILE.name} found: assuming a first investment of {_r(cash)}.")
    else:
        units, cash = holdings
        print(f"\nLoaded holdings from {HOLDINGS_FILE.name}.")

    plan = plan_trades(prices, WEIGHTS, units, cash, allow_fractional)
    total = plan["total"]

    print(f"\n{'=' * 78}\n  PORTFOLIO MONITOR  (prices = last close in data, {price_date})\n{'=' * 78}")
    print(f"Portfolio value now: {_r(total)}   (stocks {_r((units * prices).sum())}, cash {_r(cash)})")
    print(f"Largest drift from target weight: {np.abs(plan['drift']).max() * 100:.2f} "
          f"percentage points (rebalance band: {DRIFT_BAND * 100:.0f})")

    if not plan["triggered"]:
        print("\nNO REBALANCE NEEDED: every weight is within the band. Trading now would only cost money.")
        final_units, cash_after, costs_total = units, cash, 0.0
    else:
        reason = "first investment" if plan["first_investment"] else "drift beyond the band"
        print(f"\nREBALANCE TRIGGERED ({reason}).")
        final_units, cash_after = plan["final_units"], plan["cash_after"]
        costs_total = plan["costs"].sum()

        tv, tu = plan["trade_value"], plan["trade_units"]
        w_after = final_units * prices / (total - costs_total)
        table = pd.DataFrame({
            "Ticker": TICKERS,
            "Price (R)": [f"{p:,.2f}" for p in prices],
            "Units now": [f"{u:,.4f}" for u in units],
            "Trade units": [f"{u:+,.4f}" for u in tu],
            "Trade (R)": [f"{v:+,.2f}" for v in tv],
            "Cost (R)": [f"{c:,.2f}" for c in plan["costs"]],
            "Weight after": [f"{w * 100:.2f}%" for w in w_after],
            "Action": ["BUY" if v > 0 else ("SELL" if v < 0 else "-") for v in tv],
        })
        print("\n" + table.to_string(index=False))

        buys, sells = tv[tv > 0].sum(), abs(tv[tv < 0].sum())
        print(f"\nBuys {_r(buys)} | Sells {_r(sells)} | Estimated costs {_r(costs_total)} "
              f"({costs_total / total * 100:.2f}% of portfolio) | Cash left {_r(cash_after)}")
        dev = np.abs(w_after - WEIGHTS).max() * 100
        print(f"Largest gap between planned and target weight: {dev:.2f} percentage points"
              + ("  (whole-share rounding: consider fractional shares or more capital)"
                 if not allow_fractional and dev > 1 else ""))

        PRIVATE_DIR.mkdir(exist_ok=True)
        out = pd.DataFrame({
            "ticker": TICKERS, "price": prices, "units_now": units,
            "trade_units": tu, "trade_value": tv, "est_cost": plan["costs"],
            "action": table["Action"],
        })
        out.to_csv(TRADES_FILE, index=False)
        print(f"Saved the order list to {TRADES_FILE}")

    value_after = total - costs_total
    w_vec = final_units * prices / value_after
    risk = risk_summary(asset_returns, w_vec, value_after)

    print(f"\n{'-' * 78}\n  1-DAY RISK OF THE PORTFOLIO YOU WILL HOLD ({_r(value_after)})\n{'-' * 78}")
    print(f"{'':<22}{'Historical':>14}{'EWMA-FHS (now)':>18}")
    for c in (95, 99):
        print(f"{c}% VaR".ljust(22) + f"{_r(risk[f'hist_var_{c}']):>14}{_r(risk[f'fhs_var_{c}']):>18}")
        print(f"{c}% Expected Shortfall".ljust(22) + f"{_r(risk[f'hist_es_{c}']):>14}{_r(risk[f'fhs_es_{c}']):>18}")
    print(f"Forecast volatility tomorrow (annualised): {risk['vol_next_ann'] * 100:.1f}%")

    measure = max(risk["hist_var_99"], risk["fhs_var_99"])
    limit = VAR_LIMIT * value_after
    status = "WITHIN LIMIT" if measure <= limit else "OVER LIMIT: consider holding more cash or less risk"
    print(f"\nLimit check: larger of the two 99% VaRs = {_r(measure)} "
          f"({measure / value_after * 100:.2f}% of value) vs limit {_r(limit)} "
          f"({VAR_LIMIT * 100:.1f}%) -> {status}")

    print("\nNotes: prices are the last close, so check live quotes before ordering. Costs are")
    print("estimates (brokerage + securities transfer tax) and exclude platform fees and any")
    print("levies not set in OTHER_LEVIES_RATE. VaR says how bad a normal bad day can be;")
    print("it does not cap losses (see the stress tests). Not financial advice.")


if __name__ == "__main__":
    main()