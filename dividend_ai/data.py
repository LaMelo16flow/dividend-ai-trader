"""Fetches fundamentals and dividend history for a ticker via yfinance."""

from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf


@dataclass
class StockData:
    ticker: str
    name: str | None = None
    sector: str | None = None
    price: float | None = None
    dividend_yield_pct: float | None = None
    payout_ratio_pct: float | None = None
    trailing_pe: float | None = None
    forward_pe: float | None = None
    debt_to_equity: float | None = None
    free_cashflow: float | None = None
    market_cap: float | None = None
    annual_dividends: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    error: str | None = None


def _annualize_dividends(dividends: pd.Series, years: int = 11) -> pd.Series:
    """Collapse a raw dividend-payment history into per-calendar-year totals."""
    if dividends is None or dividends.empty:
        return pd.Series(dtype=float)

    s = dividends.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    annual = s.resample("YE").sum()
    annual.index = annual.index.year

    # Drop the current, possibly-incomplete calendar year and keep the most
    # recent `years` complete years so growth/consistency math isn't skewed
    # by a partial final year.
    current_year = pd.Timestamp.now().year
    annual = annual[annual.index < current_year]
    return annual.tail(years)


def fetch_stock_data(ticker: str) -> StockData:
    """Pull fundamentals + dividend history for one ticker. Never raises —
    failures are captured in StockData.error so a screening run over many
    tickers doesn't die on one bad symbol."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            # yfinance sometimes returns a near-empty dict for delisted/typo'd tickers
            if not info.get("longName") and not info.get("shortName"):
                return StockData(ticker=ticker, error="No data returned (bad ticker or delisted?)")

        dividends = t.dividends
        annual = _annualize_dividends(dividends)

        # `dividendRate` (dollars/share) and price are both unambiguous, so
        # prefer computing yield from those over `dividendYield`, whose units
        # have changed across yfinance versions (sometimes a fraction like
        # 0.025, sometimes already a percent like 2.5) and can't be told apart
        # from a genuinely sub-1%-yield stock reported in percent form.
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        div_rate = info.get("dividendRate")
        if div_rate is not None and price:
            div_yield = round(div_rate / price * 100, 2)
        else:
            div_yield = info.get("dividendYield")
            if div_yield is not None and div_yield < 1:
                div_yield *= 100
        payout = info.get("payoutRatio")

        return StockData(
            ticker=ticker,
            name=info.get("longName") or info.get("shortName"),
            sector=info.get("sector"),
            price=price,
            dividend_yield_pct=div_yield,
            payout_ratio_pct=(payout * 100) if payout is not None else None,
            trailing_pe=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            debt_to_equity=info.get("debtToEquity"),
            free_cashflow=info.get("freeCashflow"),
            market_cap=info.get("marketCap"),
            annual_dividends=annual,
        )
    except Exception as exc:  # noqa: BLE001 - intentionally broad; see docstring
        return StockData(ticker=ticker, error=str(exc))
