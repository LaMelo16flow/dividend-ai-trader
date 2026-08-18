"""A local, no-signup paper-trading simulator.

Tracks a virtual cash balance and holdings in a JSON file on disk. Prices
come from yfinance (via dividend_ai.data), so P&L reflects real market
moves, but no broker, account, or API key is involved anywhere — it's
bookkeeping against a plain file.
"""

import datetime as dt
import json
import os

DEFAULT_ACCOUNT_PATH = "paper_account.json"
DEFAULT_STARTING_CASH = 100_000.0


def load_account(path: str = DEFAULT_ACCOUNT_PATH, starting_cash: float = DEFAULT_STARTING_CASH) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"cash": starting_cash, "starting_cash": starting_cash, "holdings": {}, "transactions": []}


def save_account(account: dict, path: str = DEFAULT_ACCOUNT_PATH) -> None:
    with open(path, "w") as f:
        json.dump(account, f, indent=2)


def reset_account(path: str = DEFAULT_ACCOUNT_PATH, starting_cash: float = DEFAULT_STARTING_CASH) -> dict:
    account = {"cash": starting_cash, "starting_cash": starting_cash, "holdings": {}, "transactions": []}
    save_account(account, path)
    return account


def buy(account: dict, ticker: str, dollar_amount: float, price: float | None) -> str | None:
    """Mutates `account` in place. Returns an error message, or None on success."""
    if price is None or price <= 0:
        return f"No valid price available for {ticker}."
    if dollar_amount <= 0:
        return "Dollar amount must be positive."
    if dollar_amount > account["cash"]:
        return f"Insufficient cash: have ${account['cash']:.2f}, need ${dollar_amount:.2f}."

    qty = dollar_amount / price
    holding = account["holdings"].get(ticker, {"qty": 0.0, "avg_entry_price": 0.0})
    new_qty = holding["qty"] + qty
    new_avg = (holding["qty"] * holding["avg_entry_price"] + qty * price) / new_qty
    account["holdings"][ticker] = {"qty": new_qty, "avg_entry_price": new_avg}
    account["cash"] -= dollar_amount
    account["transactions"].append({
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker, "side": "buy", "qty": round(qty, 6),
        "price": price, "amount": round(dollar_amount, 2),
    })
    return None


def sell(account: dict, ticker: str, price: float | None, qty: float | None = None) -> str | None:
    """Sells `qty` shares (or the whole position if qty is None). Mutates
    `account` in place. Returns an error message, or None on success."""
    holding = account["holdings"].get(ticker)
    if not holding or holding["qty"] <= 0:
        return f"No position in {ticker} to sell."
    if price is None or price <= 0:
        return f"No valid price available for {ticker}."

    sell_qty = holding["qty"] if qty is None else min(qty, holding["qty"])
    if sell_qty <= 0:
        return "Quantity to sell must be positive."

    proceeds = sell_qty * price
    remaining = holding["qty"] - sell_qty
    if remaining <= 1e-9:
        del account["holdings"][ticker]
    else:
        account["holdings"][ticker]["qty"] = remaining

    account["cash"] += proceeds
    account["transactions"].append({
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker, "side": "sell", "qty": round(sell_qty, 6),
        "price": price, "amount": round(proceeds, 2),
    })
    return None


def get_positions(account: dict, price_lookup: dict[str, float | None]) -> list[dict]:
    rows = []
    for ticker, h in account["holdings"].items():
        current = price_lookup.get(ticker)
        cost_basis = h["qty"] * h["avg_entry_price"]
        market_value = h["qty"] * current if current is not None else None
        unrealized_pl = (market_value - cost_basis) if market_value is not None else None
        unrealized_plpc = (unrealized_pl / cost_basis * 100) if unrealized_pl is not None and cost_basis else None
        rows.append({
            "ticker": ticker,
            "qty": h["qty"],
            "avg_entry_price": h["avg_entry_price"],
            "current_price": current,
            "market_value": market_value,
            "unrealized_pl": unrealized_pl,
            "unrealized_plpc": unrealized_plpc,
        })
    return rows


def get_summary(account: dict, price_lookup: dict[str, float | None]) -> dict:
    positions = get_positions(account, price_lookup)
    holdings_value = sum(p["market_value"] or 0.0 for p in positions)
    portfolio_value = account["cash"] + holdings_value
    starting_cash = account["starting_cash"]
    total_pl = portfolio_value - starting_cash
    total_pl_pct = (total_pl / starting_cash * 100) if starting_cash else 0.0
    return {
        "cash": account["cash"],
        "holdings_value": holdings_value,
        "portfolio_value": portfolio_value,
        "total_pl": total_pl,
        "total_pl_pct": total_pl_pct,
    }
