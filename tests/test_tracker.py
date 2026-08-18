import pandas as pd

from dividend_ai.data import StockData
from dividend_ai.scoring import ScoreResult
from dividend_ai.tracker import record_picks, review_picks


def _result(ticker, price, composite=70.0, grade="B"):
    return ScoreResult(ticker=ticker, name=ticker, composite=composite, grade=grade,
                        raw={"price": price})


def test_record_picks_writes_top_n_and_skips_errors(tmp_path):
    log_path = str(tmp_path / "picks_log.csv")
    results = [
        _result("AAA", 100.0),
        _result("BBB", 50.0),
        ScoreResult(ticker="ERR", name=None, composite=0.0, grade="F", error="fetch failed"),
    ]
    n = record_picks(results, log_path, top=1, pick_date="2026-01-01")
    assert n == 1

    df = pd.read_csv(log_path)
    assert list(df["ticker"]) == ["AAA"]
    assert df.iloc[0]["price_at_pick"] == 100.0


def test_record_picks_appends_across_calls(tmp_path):
    log_path = str(tmp_path / "picks_log.csv")
    record_picks([_result("AAA", 100.0)], log_path, pick_date="2026-01-01")
    record_picks([_result("BBB", 50.0)], log_path, pick_date="2026-01-02")

    df = pd.read_csv(log_path)
    assert list(df["ticker"]) == ["AAA", "BBB"]


def test_review_picks_computes_return_since_pick(tmp_path, monkeypatch):
    log_path = str(tmp_path / "picks_log.csv")
    record_picks([_result("AAA", 100.0)], log_path, pick_date="2026-01-01")

    def fake_fetch(ticker):
        return StockData(ticker=ticker, price=110.0)

    monkeypatch.setattr("dividend_ai.tracker.fetch_stock_data", fake_fetch)

    log = review_picks(log_path)
    assert log.iloc[0]["current_price"] == 110.0
    assert log.iloc[0]["return_pct"] == 10.0
