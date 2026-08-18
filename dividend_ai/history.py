"""Cached historical price + dividend fetching, for backtesting.

Backtests replay years of daily data across many tickers, which is slow
and unnecessary to re-fetch every run — yfinance's `history()` already
returns a `Dividends` column alongside `Close`, so one fetch per ticker
covers both. Results are cached to disk (`.cache/`) and reused within
`max_age_hours`, per the project's free-first / minimize-API-calls design.
"""

import os

import pandas as pd
import yfinance as yf

DEFAULT_CACHE_DIR = ".cache"
DEFAULT_MAX_AGE_HOURS = 24


def _cache_path(ticker: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{ticker.upper()}_history.csv")


def get_history(
    ticker: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> tuple[pd.DataFrame | None, str | None]:
    """Returns a DataFrame indexed by date with `Close` and `Dividends`
    columns covering all available history, or (None, error)."""
    path = _cache_path(ticker, cache_dir)

    if os.path.exists(path):
        age_hours = (pd.Timestamp.now().timestamp() - os.path.getmtime(path)) / 3600
        if age_hours < max_age_hours:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if not df.empty:
                return df, None

    try:
        raw = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    except Exception as exc:  # noqa: BLE001 - network call, many failure modes
        return None, str(exc)

    if raw is None or raw.empty:
        return None, f"No historical data returned for {ticker}."

    df = raw[["Close", "Dividends"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    os.makedirs(cache_dir, exist_ok=True)
    df.to_csv(path)
    return df, None


def monthly_series(history: pd.DataFrame) -> pd.DataFrame:
    """Collapses daily history into month-end close price + summed
    dividends paid during that month."""
    close = history["Close"].resample("ME").last()
    dividends = history["Dividends"].resample("ME").sum()
    df = pd.DataFrame({"close": close, "dividend": dividends}).dropna(subset=["close"])
    return df
