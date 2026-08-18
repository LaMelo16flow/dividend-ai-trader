import pandas as pd

from dividend_ai.data import StockData
from dividend_ai.screener import filter_results
from dividend_ai.scoring import (
    consecutive_years_no_cut,
    dividend_growth_cagr,
    score_consistency,
    score_financial_health,
    score_payout,
    score_stock,
    score_valuation,
    score_yield,
)


def test_score_yield_prefers_moderate_range():
    assert score_yield(4) > score_yield(0.2)
    assert score_yield(4) > score_yield(20)  # yield trap penalized
    assert score_yield(None) == 50.0


def test_score_payout_penalizes_over_100():
    assert score_payout(45) > score_payout(120)
    assert score_payout(0) < score_payout(45)


def test_score_financial_health_prefers_low_debt():
    assert score_financial_health(20) > score_financial_health(300)


def test_score_valuation_prefers_cheap_but_not_free():
    assert score_valuation(15) > score_valuation(70)
    assert score_valuation(None) == 50.0
    assert score_valuation(-5) == 50.0  # negative PE treated as neutral


def test_dividend_growth_cagr_computes_positive_growth():
    annual = pd.Series([1.0, 1.05, 1.10, 1.15, 1.20, 1.25], index=range(2018, 2024))
    cagr = dividend_growth_cagr(annual, years=5)
    assert cagr is not None
    assert 0.03 < cagr < 0.06


def test_dividend_growth_cagr_needs_two_points():
    assert dividend_growth_cagr(pd.Series([1.0]), years=5) is None
    assert dividend_growth_cagr(pd.Series(dtype=float), years=5) is None


def test_consecutive_years_no_cut_stops_at_first_cut():
    annual = pd.Series([1.0, 1.1, 0.9, 1.0, 1.1, 1.2], index=range(2018, 2024))
    # growth sequence: +10%, -18%(cut), +11%, +10%, +9% -> streak of 3 since the cut
    assert consecutive_years_no_cut(annual) == 3


def test_score_consistency_caps_at_100():
    assert score_consistency(20) == 100.0
    assert score_consistency(0) == 0.0
    assert score_consistency(5) == 50.0


def test_score_stock_surfaces_error_without_crashing():
    sd = StockData(ticker="BADTICKER", error="No data returned")
    result = score_stock(sd)
    assert result.error == "No data returned"
    assert result.composite == 0.0
    assert result.grade == "F"


def test_score_stock_end_to_end_happy_path():
    annual = pd.Series([1.0, 1.05, 1.10, 1.15, 1.20], index=range(2019, 2024))
    sd = StockData(
        ticker="TEST",
        name="Test Corp",
        price=100.0,
        dividend_yield_pct=3.5,
        payout_ratio_pct=50.0,
        trailing_pe=16.0,
        debt_to_equity=80.0,
        annual_dividends=annual,
    )
    result = score_stock(sd)
    assert result.error is None
    assert 0 <= result.composite <= 100
    assert result.grade in {"A", "B", "C", "D", "F"}
    assert set(result.sub_scores) == {
        "yield", "growth", "payout", "consistency", "financial_health", "valuation",
    }


def _make_result(ticker, composite, grade, **raw_overrides):
    from dividend_ai.scoring import ScoreResult

    raw = {"dividend_yield_pct": 3.0, "payout_ratio_pct": 50.0, "trailing_pe": 15.0,
           "debt_to_equity": 60.0}
    raw.update(raw_overrides)
    return ScoreResult(ticker=ticker, name=ticker, composite=composite, grade=grade, raw=raw)


def test_filter_results_by_min_score_and_grade():
    results = [
        _make_result("AAA", 90.0, "A"),
        _make_result("BBB", 60.0, "C"),
        _make_result("CCC", 30.0, "F"),
    ]
    assert [r.ticker for r in filter_results(results, min_score=50)] == ["AAA", "BBB"]
    assert [r.ticker for r in filter_results(results, min_grade="B")] == ["AAA"]


def test_filter_results_by_yield_payout_pe_de():
    results = [
        _make_result("LOWY", 70.0, "B", dividend_yield_pct=0.5),
        _make_result("HIPO", 70.0, "B", payout_ratio_pct=120.0),
        _make_result("GOOD", 70.0, "B"),
    ]
    kept = filter_results(results, min_yield=2.0, max_payout=80.0)
    assert [r.ticker for r in kept] == ["GOOD"]


def test_filter_results_keeps_missing_data_and_errors():
    from dividend_ai.scoring import ScoreResult

    results = [
        _make_result("NODATA", 70.0, "B", dividend_yield_pct=None),
        ScoreResult(ticker="ERR", name=None, composite=0.0, grade="F", error="fetch failed"),
    ]
    kept = filter_results(results, min_yield=5.0)
    assert [r.ticker for r in kept] == ["NODATA", "ERR"]
