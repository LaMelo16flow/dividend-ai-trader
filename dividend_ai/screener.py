"""Runs the full screen over a list of tickers and returns a ranked table."""

from typing import Callable

import pandas as pd

from dividend_ai.config import GRADE_BANDS
from dividend_ai.data import fetch_stock_data
from dividend_ai.scoring import ScoreResult, score_stock


def screen(
    tickers: list[str],
    progress: bool = True,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[ScoreResult]:
    """Fetches and scores every ticker. `on_progress(i, total, ticker)` is
    called before each fetch, letting callers (e.g. a GUI) drive their own
    progress indicator instead of the console progress printed by `progress`.
    """
    results = []
    for i, tkr in enumerate(tickers, 1):
        if progress:
            print(f"[{i}/{len(tickers)}] fetching {tkr}...")
        if on_progress:
            on_progress(i, len(tickers), tkr)
        sd = fetch_stock_data(tkr)
        results.append(score_stock(sd))
    results.sort(key=lambda r: r.composite, reverse=True)
    return results


def _grade_rank(grade: str) -> int:
    """Position in GRADE_BANDS (best-to-worst order); lower is better."""
    letters = [letter for _, letter in GRADE_BANDS]
    return letters.index(grade) if grade in letters else len(letters)


def filter_results(
    results: list[ScoreResult],
    *,
    min_yield: float | None = None,
    max_payout: float | None = None,
    min_score: float | None = None,
    min_grade: str | None = None,
    max_pe: float | None = None,
    max_de: float | None = None,
) -> list[ScoreResult]:
    """Keeps stocks meeting all given thresholds; order is preserved.

    A stock missing the underlying data for a threshold is kept rather than
    excluded, matching the "unknown is neutral, not punished" philosophy
    used throughout scoring.py. Errored results are always kept as-is since
    they have no raw data to filter on.
    """
    min_grade_rank = _grade_rank(min_grade) if min_grade else None
    kept = []
    for r in results:
        if r.error:
            kept.append(r)
            continue

        if min_score is not None and r.composite < min_score:
            continue
        if min_grade_rank is not None and _grade_rank(r.grade) > min_grade_rank:
            continue

        yld = r.raw.get("dividend_yield_pct")
        if min_yield is not None and yld is not None and yld < min_yield:
            continue

        payout = r.raw.get("payout_ratio_pct")
        if max_payout is not None and payout is not None and payout > max_payout:
            continue

        pe = r.raw.get("trailing_pe")
        if max_pe is not None and pe is not None and pe > max_pe:
            continue

        de = r.raw.get("debt_to_equity")
        if max_de is not None and de is not None and de > max_de:
            continue

        kept.append(r)
    return kept


def to_dataframe(results: list[ScoreResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        if r.error:
            rows.append({"ticker": r.ticker, "name": r.name, "composite": None,
                         "grade": None, "error": r.error})
            continue
        row = {"ticker": r.ticker, "name": r.name, "composite": r.composite, "grade": r.grade}
        row.update({f"score_{k}": v for k, v in r.sub_scores.items()})
        row.update(r.raw)
        row["notes"] = " | ".join(r.notes)
        rows.append(row)
    return pd.DataFrame(rows)


def print_report(results: list[ScoreResult], top: int | None = None) -> None:
    ok = [r for r in results if not r.error]
    failed = [r for r in results if r.error]

    shown = ok[:top] if top is not None else ok
    print(f"\n{'RANK':<5}{'TICKER':<8}{'GRADE':<7}{'SCORE':<8}{'NAME'}")
    print("-" * 70)
    for i, r in enumerate(shown, 1):
        print(f"{i:<5}{r.ticker:<8}{r.grade:<7}{r.composite:<8}{r.name or ''}")

    print("\nDetail:")
    for i, r in enumerate(shown, 1):
        print(f"\n{i}. {r.ticker} — {r.name}  [{r.grade}, {r.composite}/100]")
        sub = "  ".join(f"{k}={v}" for k, v in r.sub_scores.items())
        print(f"   sub-scores: {sub}")
        raw = r.raw
        print(f"   yield={raw.get('dividend_yield_pct')}%  payout={raw.get('payout_ratio_pct')}%  "
              f"5y div CAGR={raw.get('dividend_cagr_5y_pct')}%  no-cut streak={raw.get('no_cut_streak_years')}yr  "
              f"P/E={raw.get('trailing_pe')}  D/E={raw.get('debt_to_equity')}")
        for note in r.notes:
            print(f"   - {note}")

    if failed:
        print(f"\n{len(failed)} ticker(s) failed to fetch:")
        for r in failed:
            print(f"   {r.ticker}: {r.error}")

    print(
        "\nThis is a rule-based educational screening tool, not financial advice. "
        "Scores are derived from public data (via yfinance) which can be incomplete, "
        "delayed, or wrong — verify anything before acting on it."
    )
