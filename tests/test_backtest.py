import pandas as pd
import pytest

from dividend_ai.backtest import (
    build_strategy_universes,
    classify_frequency,
    compare_strategies,
    simulate_portfolio,
    xirr,
)
from dividend_ai.data import StockData
from dividend_ai.scoring import ScoreResult


HISTORY_END = pd.Timestamp("2024-12-31")


def _history(prices, dividends=None, periods=None):
    n = periods or len(prices)
    idx = pd.date_range(end=HISTORY_END, periods=n, freq="ME")
    div = dividends if dividends is not None else [0.0] * n
    return pd.DataFrame({"Close": prices, "Dividends": div}, index=idx)


def _patch_history(monkeypatch, histories: dict):
    def fake_get_history(ticker, cache_dir=".cache", max_age_hours=24):
        if ticker in histories:
            return histories[ticker], None
        return None, f"no data for {ticker}"

    monkeypatch.setattr("dividend_ai.backtest.get_history", fake_get_history)


# ---- simulate_portfolio ----

def test_lump_sum_final_value_matches_price_return(monkeypatch):
    prices = [100 * (1.01 ** i) for i in range(12)]
    _patch_history(monkeypatch, {"AAA": _history(prices)})

    result = simulate_portfolio(["AAA"], years=1, initial_capital=10_000, end_date=HISTORY_END)

    expected_final = 10_000 * prices[-1] / prices[0]
    assert result.final_value == pytest.approx(expected_final, rel=1e-6)
    assert result.total_return_pct == pytest.approx((prices[-1] / prices[0] - 1) * 100, abs=0.01)
    assert result.error is None


def test_monotonic_growth_has_no_drawdown(monkeypatch):
    prices = [100 * (1.01 ** i) for i in range(12)]
    _patch_history(monkeypatch, {"AAA": _history(prices)})

    result = simulate_portfolio(["AAA"], years=1, initial_capital=10_000, end_date=HISTORY_END)
    assert result.max_drawdown_pct == pytest.approx(0.0, abs=0.01)


def test_max_drawdown_detects_trough(monkeypatch):
    prices = [100, 120, 90, 150]
    _patch_history(monkeypatch, {"AAA": _history(prices)})

    result = simulate_portfolio(["AAA"], years=1, initial_capital=10_000, end_date=HISTORY_END)
    assert result.max_drawdown_pct == pytest.approx(-25.0, rel=1e-6)


def test_drip_beats_no_drip_with_flat_price(monkeypatch):
    prices = [100.0] * 12
    dividends = [1.0] * 12
    _patch_history(monkeypatch, {"AAA": _history(prices, dividends)})

    drip = simulate_portfolio(["AAA"], years=1, initial_capital=10_000, drip=True, end_date=HISTORY_END)
    no_drip = simulate_portfolio(["AAA"], years=1, initial_capital=10_000, drip=False, end_date=HISTORY_END)

    assert drip.final_value > no_drip.final_value
    assert drip.total_dividends > no_drip.total_dividends
    # flat price + no reinvestment => ending value is exactly cash-in plus dividends collected
    assert no_drip.final_value == pytest.approx(10_000 + no_drip.total_dividends, rel=1e-6)


def test_monthly_contributions_sum_correctly(monkeypatch):
    prices = [100.0] * 12
    _patch_history(monkeypatch, {"AAA": _history(prices)})

    result = simulate_portfolio(["AAA"], years=1, initial_capital=10_000, monthly_contribution=500, end_date=HISTORY_END)
    assert result.total_contributions == pytest.approx(10_000 + 500 * 11, rel=1e-9)


def test_missing_ticker_reports_error_without_raising(monkeypatch):
    _patch_history(monkeypatch, {})
    result = simulate_portfolio(["ZZZ"], years=1, initial_capital=10_000)
    assert result.error is not None
    assert "ZZZ" in result.ticker_errors[0]


# ---- xirr ----

def test_xirr_matches_simple_annual_return():
    t0 = pd.Timestamp("2023-01-01")
    t1 = t0 + pd.Timedelta(days=365)
    rate = xirr([(t0, -1000.0), (t1, 1100.0)])
    assert rate == pytest.approx(0.10, abs=1e-3)


def test_xirr_empty_returns_zero():
    assert xirr([]) == 0.0


# ---- classify_frequency ----

def test_classify_frequency_monthly():
    idx = pd.date_range(end=pd.Timestamp.today(), periods=24, freq="MS")
    hist = pd.DataFrame({"Close": [100.0] * 24, "Dividends": [1.0] * 24}, index=idx)
    assert classify_frequency(hist) == "Monthly"


def test_classify_frequency_quarterly():
    idx = pd.date_range(end=pd.Timestamp.today(), periods=8, freq="QS")
    hist = pd.DataFrame({"Close": [100.0] * 8, "Dividends": [1.0] * 8}, index=idx)
    assert classify_frequency(hist) == "Quarterly"


def test_classify_frequency_annual():
    idx = pd.date_range(end=pd.Timestamp.today(), periods=2, freq="YS")
    hist = pd.DataFrame({"Close": [100.0] * 2, "Dividends": [1.0] * 2}, index=idx)
    assert classify_frequency(hist) == "Annual"


def test_classify_frequency_none_when_no_dividends():
    idx = pd.date_range(end=pd.Timestamp.today(), periods=12, freq="MS")
    hist = pd.DataFrame({"Close": [100.0] * 12, "Dividends": [0.0] * 12}, index=idx)
    assert classify_frequency(hist) == "None"


def test_classify_frequency_empty_history():
    assert classify_frequency(pd.DataFrame(columns=["Close", "Dividends"])) == "None"


# ---- compare_strategies ----

def test_compare_strategies_runs_each_universe(monkeypatch):
    prices = [100 * (1.01 ** i) for i in range(12)]
    _patch_history(monkeypatch, {"AAA": _history(prices), "BBB": _history(prices)})

    universes = {"Strategy One": ["AAA"], "Strategy Two": ["BBB"], "Empty Strategy": []}
    results = compare_strategies(universes, years=1, initial_capital=10_000, end_date=HISTORY_END)

    assert set(results.keys()) == set(universes.keys())
    assert results["Strategy One"].error is None
    assert results["Empty Strategy"].error is not None


# ---- build_strategy_universes ----

def test_build_strategy_universes_buckets_by_frequency_and_score(monkeypatch):
    monthly_hist = pd.DataFrame(
        {"Close": [100.0] * 24, "Dividends": [1.0] * 24},
        index=pd.date_range(end=pd.Timestamp.today(), periods=24, freq="MS"),
    )
    quarterly_hist = pd.DataFrame(
        {"Close": [100.0] * 8, "Dividends": [1.0] * 8},
        index=pd.date_range(end=pd.Timestamp.today(), periods=8, freq="QS"),
    )

    def fake_get_history(ticker, cache_dir=".cache", max_age_hours=24):
        return (monthly_hist, None) if ticker == "MMM_PAYER" else (quarterly_hist, None)

    def fake_fetch_stock_data(ticker):
        return StockData(ticker=ticker, name=ticker, price=100.0, dividend_yield_pct=3.0,
                          payout_ratio_pct=50.0, trailing_pe=15.0, debt_to_equity=80.0)

    monkeypatch.setattr("dividend_ai.backtest.get_history", fake_get_history)
    monkeypatch.setattr("dividend_ai.data.fetch_stock_data", fake_fetch_stock_data)

    universes, errors = build_strategy_universes(["MMM_PAYER", "QQQ_PAYER"], top_n=5)

    assert "MMM_PAYER" in universes["Monthly Income"]
    assert "QQQ_PAYER" in universes["Quarterly Income"]
    assert len(universes["Balanced Dividend"]) == 2
