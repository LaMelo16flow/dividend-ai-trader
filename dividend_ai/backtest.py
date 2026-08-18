"""Historical strategy backtesting and comparison.

Simulates a portfolio of tickers, month by month, using real historical
prices and dividend payments (via dividend_ai.history). Supports optional
recurring contributions and DRIP (dividend reinvestment), and separates
two return concepts that are easy to conflate:

- Money-weighted return (XIRR): the annualized rate that reconciles actual
  cash flows (contributions in, final value out) — what the investor
  actually experienced, sensitive to contribution timing.
- Time-weighted return: a synthetic NAV-per-unit index built from monthly
  investment returns with contribution cash flows stripped out — used for
  volatility, Sharpe, and max drawdown, since those shouldn't be distorted
  by how much cash happened to be added in a given month.
"""

import math
from dataclasses import dataclass, field

import pandas as pd

from dividend_ai.history import get_history, monthly_series

FREQUENCY_BANDS = [
    (10.0, "Monthly"),
    (3.5, "Quarterly"),
    (1.5, "Semi-Annual"),
    (0.5, "Annual"),
]


def classify_frequency(history: pd.DataFrame, lookback_years: float = 2.0) -> str:
    """Classifies a ticker's dividend frequency from its actual payment
    history over the trailing `lookback_years`, not from reputation."""
    if history is None or history.empty:
        return "None"
    end = history.index.max()
    start = end - pd.DateOffset(years=lookback_years)
    recent = history.loc[(history.index > start) & (history.index <= end), "Dividends"]
    payments = int((recent > 0).sum())
    if payments == 0:
        return "None"
    per_year = payments / lookback_years
    for threshold, label in FREQUENCY_BANDS:
        if per_year >= threshold:
            return label
    return "Irregular"


def xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    """Annualized money-weighted rate of return via bisection. Returns 0.0
    if no sign change is found in a wide search range (can't solve)."""
    if not cashflows:
        return 0.0
    t0 = min(d for d, _ in cashflows)

    def npv(rate: float) -> float:
        return sum(amt / (1 + rate) ** ((d - t0).days / 365.0) for d, amt in cashflows)

    lo, hi = -0.99, 10.0
    npv_lo, npv_hi = npv(lo), npv(hi)
    if npv_lo == 0:
        return lo
    if npv_hi == 0:
        return hi
    if npv_lo * npv_hi > 0:
        return 0.0

    for _ in range(200):
        mid = (lo + hi) / 2
        npv_mid = npv(mid)
        if abs(npv_mid) < 1e-6:
            return mid
        if npv_lo * npv_mid < 0:
            hi = mid
        else:
            lo, npv_lo = mid, npv_mid
    return (lo + hi) / 2


@dataclass
class BacktestResult:
    strategy_name: str
    tickers: list[str]
    years: float = 0.0
    initial_capital: float = 0.0
    monthly_contribution: float = 0.0
    drip: bool = True
    total_contributions: float = 0.0
    final_value: float = 0.0
    total_dividends: float = 0.0
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    volatility_pct: float = 0.0
    sharpe: float | None = None
    yield_on_cost_pct: float = 0.0
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    ticker_errors: list[str] = field(default_factory=list)
    error: str | None = None


def simulate_portfolio(
    tickers: list[str],
    years: float,
    initial_capital: float,
    monthly_contribution: float = 0.0,
    drip: bool = True,
    end_date: pd.Timestamp | None = None,
    cache_dir: str = ".cache",
    strategy_name: str = "",
) -> BacktestResult:
    """Equal-weight buy-and-hold simulation with monthly rebalancing only
    via new contributions (no periodic rebalancing of existing positions).
    """
    end_date = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=years)

    per_ticker: dict[str, pd.DataFrame] = {}
    ticker_errors: list[str] = []
    for t in tickers:
        hist, err = get_history(t, cache_dir=cache_dir)
        if err:
            ticker_errors.append(f"{t}: {err}")
            continue
        ms = monthly_series(hist)
        ms = ms[(ms.index >= start_date) & (ms.index <= end_date)]
        if ms.empty:
            ticker_errors.append(f"{t}: no price data in the requested window")
            continue
        per_ticker[t] = ms

    if not per_ticker:
        return BacktestResult(
            strategy_name=strategy_name, tickers=tickers, years=years,
            initial_capital=initial_capital, monthly_contribution=monthly_contribution,
            drip=drip, ticker_errors=ticker_errors,
            error="No usable price history for any ticker in this strategy.",
        )

    all_months = pd.DatetimeIndex(sorted(set().union(*[set(ms.index) for ms in per_ticker.values()])))
    active_tickers = list(per_ticker.keys())
    for t in active_tickers:
        per_ticker[t] = per_ticker[t].reindex(all_months)
        per_ticker[t]["close"] = per_ticker[t]["close"].ffill()
        per_ticker[t]["dividend"] = per_ticker[t]["dividend"].fillna(0.0)

    shares = {t: 0.0 for t in active_tickers}
    cash = 0.0
    total_contributions = 0.0
    total_dividends = 0.0
    monthly_dividend_cash: list[float] = []
    returns: list[float] = []
    equity_curve: list[tuple[pd.Timestamp, float]] = []
    synthetic_index = [1.0]
    cashflows: list[tuple[pd.Timestamp, float]] = []

    prev_value = 0.0
    for i, month_end in enumerate(all_months):
        value_start = cash
        for t in active_tickers:
            price = per_ticker[t].loc[month_end, "close"]
            if pd.notna(price):
                value_start += shares[t] * price

        dividend_cash = 0.0
        dividend_by_ticker: dict[str, float] = {}
        for t in active_tickers:
            div_ps = per_ticker[t].loc[month_end, "dividend"]
            if div_ps and shares[t] > 0:
                amt = shares[t] * div_ps
                dividend_by_ticker[t] = amt
                dividend_cash += amt
        total_dividends += dividend_cash
        monthly_dividend_cash.append(dividend_cash)

        value_after_income = value_start + dividend_cash

        if prev_value > 0:
            r = value_after_income / prev_value - 1
            returns.append(r)
            synthetic_index.append(synthetic_index[-1] * (1 + r))

        if drip:
            for t, amt in dividend_by_ticker.items():
                price = per_ticker[t].loc[month_end, "close"]
                if pd.notna(price) and price > 0:
                    shares[t] += amt / price
                else:
                    cash += amt
        else:
            cash += dividend_cash

        contribution = initial_capital if i == 0 else monthly_contribution
        if contribution > 0:
            available = [t for t in active_tickers if pd.notna(per_ticker[t].loc[month_end, "close"])]
            if available:
                per_ticker_amt = contribution / len(available)
                for t in available:
                    price = per_ticker[t].loc[month_end, "close"]
                    shares[t] += per_ticker_amt / price
            else:
                cash += contribution
            total_contributions += contribution
            cashflows.append((month_end, -contribution))

        value_end = cash
        for t in active_tickers:
            price = per_ticker[t].loc[month_end, "close"]
            if pd.notna(price):
                value_end += shares[t] * price
        equity_curve.append((month_end, value_end))
        prev_value = value_end

    final_value = prev_value
    if all_months.size:
        cashflows.append((all_months[-1], final_value))

    total_return_pct = ((final_value - total_contributions) / total_contributions * 100) if total_contributions else 0.0
    cagr_pct = xirr(cashflows) * 100

    if len(returns) >= 2:
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        monthly_std = math.sqrt(variance)
        volatility_pct = monthly_std * math.sqrt(12) * 100
        sharpe = (mean_r * 12) / (monthly_std * math.sqrt(12)) if monthly_std > 0 else None
    else:
        volatility_pct = 0.0
        sharpe = None

    running_max = synthetic_index[0]
    max_dd = 0.0
    for v in synthetic_index:
        running_max = max(running_max, v)
        dd = (v - running_max) / running_max
        max_dd = min(max_dd, dd)
    max_drawdown_pct = max_dd * 100

    trailing_12mo_dividends = sum(monthly_dividend_cash[-12:])
    yield_on_cost_pct = (trailing_12mo_dividends / total_contributions * 100) if total_contributions else 0.0

    return BacktestResult(
        strategy_name=strategy_name,
        tickers=active_tickers,
        years=years,
        initial_capital=initial_capital,
        monthly_contribution=monthly_contribution,
        drip=drip,
        total_contributions=round(total_contributions, 2),
        final_value=round(final_value, 2),
        total_dividends=round(total_dividends, 2),
        total_return_pct=round(total_return_pct, 2),
        cagr_pct=round(cagr_pct, 2),
        max_drawdown_pct=round(max_drawdown_pct, 2),
        volatility_pct=round(volatility_pct, 2),
        sharpe=round(sharpe, 2) if sharpe is not None else None,
        yield_on_cost_pct=round(yield_on_cost_pct, 2),
        equity_curve=equity_curve,
        ticker_errors=ticker_errors,
    )


def build_strategy_universes(
    candidate_tickers: list[str],
    top_n: int = 8,
    cache_dir: str = ".cache",
    on_progress=None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Classifies each candidate by observed dividend frequency and by the
    existing scoring engine, then buckets the top `top_n` per strategy."""
    from dividend_ai.data import fetch_stock_data
    from dividend_ai.scoring import score_stock

    freq_by_ticker: dict[str, str] = {}
    scored = []
    errors: list[str] = []
    total = len(candidate_tickers)

    for i, t in enumerate(candidate_tickers, 1):
        if on_progress:
            on_progress(i, total, t)

        hist, err = get_history(t, cache_dir=cache_dir)
        if err:
            errors.append(f"{t}: {err}")
        else:
            freq_by_ticker[t] = classify_frequency(hist)

        sd = fetch_stock_data(t)
        if sd.error:
            errors.append(f"{t}: {sd.error}")
            continue
        scored.append(score_stock(sd))

    monthly = [t for t, f in freq_by_ticker.items() if f == "Monthly"]
    quarterly = [t for t, f in freq_by_ticker.items() if f == "Quarterly"]
    lower_freq = [t for t, f in freq_by_ticker.items() if f in ("Semi-Annual", "Annual")]

    by_growth = sorted(scored, key=lambda r: r.sub_scores.get("growth", 0), reverse=True)
    by_yield = sorted(scored, key=lambda r: r.raw.get("dividend_yield_pct") or 0, reverse=True)
    by_composite = sorted(scored, key=lambda r: r.composite, reverse=True)
    by_total_return = sorted(
        scored, key=lambda r: r.sub_scores.get("growth", 0) + r.sub_scores.get("valuation", 0), reverse=True
    )

    universes = {
        "Monthly Income": monthly[:top_n],
        "Quarterly Income": quarterly[:top_n],
        "Annual / Lower-Frequency": lower_freq[:top_n],
        "Dividend Growth": [r.ticker for r in by_growth[:top_n]],
        "High Dividend Yield": [r.ticker for r in by_yield[:top_n]],
        "Total Return": [r.ticker for r in by_total_return[:top_n]],
        "Balanced Dividend": [r.ticker for r in by_composite[:top_n]],
    }
    return universes, errors


def compare_strategies(
    universes: dict[str, list[str]],
    years: float,
    initial_capital: float,
    monthly_contribution: float = 0.0,
    drip: bool = True,
    end_date: pd.Timestamp | None = None,
    cache_dir: str = ".cache",
) -> dict[str, BacktestResult]:
    results = {}
    for name, tickers in universes.items():
        if not tickers:
            results[name] = BacktestResult(
                strategy_name=name, tickers=[], years=years, initial_capital=initial_capital,
                monthly_contribution=monthly_contribution, drip=drip,
                error="No qualifying tickers found for this strategy in the candidate universe.",
            )
            continue
        results[name] = simulate_portfolio(
            tickers, years, initial_capital, monthly_contribution, drip,
            end_date=end_date, cache_dir=cache_dir, strategy_name=name,
        )
    return results
