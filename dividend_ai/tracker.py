"""Tracks hypothetical picks over time (paper trading, no broker involved).

A screening run's top picks can be appended to a CSV log with `record_picks`.
Later, `review_picks` re-fetches current prices for everything in the log and
reports how each pick has performed since the day it was picked.
"""

import datetime as dt
import os

import pandas as pd

from dividend_ai.data import fetch_stock_data
from dividend_ai.scoring import ScoreResult

LOG_COLUMNS = ["pick_date", "ticker", "name", "price_at_pick", "composite", "grade"]


def record_picks(
    results: list[ScoreResult],
    log_path: str,
    top: int | None = None,
    pick_date: str | None = None,
) -> int:
    """Appends this run's top picks to the tracking log. Returns rows written."""
    ok = [r for r in results if not r.error]
    shown = ok[:top] if top is not None else ok
    if not shown:
        return 0

    date = pick_date or dt.date.today().isoformat()
    rows = [{
        "pick_date": date,
        "ticker": r.ticker,
        "name": r.name,
        "price_at_pick": r.raw.get("price"),
        "composite": r.composite,
        "grade": r.grade,
    } for r in shown]

    df = pd.DataFrame(rows, columns=LOG_COLUMNS)
    write_header = not os.path.exists(log_path)
    df.to_csv(log_path, mode="a", header=write_header, index=False)
    return len(rows)


def review_picks(log_path: str) -> pd.DataFrame:
    """Re-fetches current prices for every logged pick and computes return
    since pick date. Raises FileNotFoundError if the log doesn't exist yet."""
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"No tracking log found at {log_path}")

    log = pd.read_csv(log_path, parse_dates=["pick_date"])
    if log.empty:
        return log

    current_prices = {ticker: fetch_stock_data(ticker).price for ticker in log["ticker"].unique()}

    log = log.copy()
    log["current_price"] = log["ticker"].map(current_prices)
    log["return_pct"] = (
        (log["current_price"] - log["price_at_pick"]) / log["price_at_pick"] * 100
    ).round(2)
    today = pd.Timestamp.today().normalize()
    log["days_held"] = (today - log["pick_date"]).dt.days
    return log


def print_review(log: pd.DataFrame) -> None:
    if log.empty:
        print("No tracked picks yet. Run with --track to start logging picks.")
        return

    ranked = log.sort_values("return_pct", ascending=False, na_position="last")
    print(f"\n{'DATE':<12}{'TICKER':<8}{'DAYS':<7}{'PICK $':<10}{'NOW $':<10}{'RETURN':<9}{'GRADE'}")
    print("-" * 70)
    for _, row in ranked.iterrows():
        now = f"{row['current_price']:.2f}" if pd.notna(row["current_price"]) else "n/a"
        ret = f"{row['return_pct']:+.2f}%" if pd.notna(row["return_pct"]) else "n/a"
        print(f"{str(row['pick_date'].date()):<12}{row['ticker']:<8}{row['days_held']:<7}"
              f"{row['price_at_pick']:<10.2f}{now:<10}{ret:<9}{row['grade']}")

    valid = ranked["return_pct"].dropna()
    if not valid.empty:
        print(f"\nAverage return across {len(valid)} tracked pick(s): {valid.mean():+.2f}%")
