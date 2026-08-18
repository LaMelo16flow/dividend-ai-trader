"""Command-line entry point: python -m dividend_ai.cli [options]"""

import argparse

from dividend_ai.config import DEFAULT_UNIVERSE, GRADE_BANDS
from dividend_ai.screener import filter_results, print_report, screen, to_dataframe
from dividend_ai.tracker import print_review, record_picks, review_picks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen and rank dividend-paying stocks by yield quality, "
                     "growth, payout sustainability, consistency, financial health, "
                     "and valuation."
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Space-separated tickers to screen (default: built-in dividend universe).",
    )
    parser.add_argument(
        "--top", type=int, default=15,
        help="Number of top-ranked stocks to show in detail (default: 15).",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Optional path to write the full ranked results as CSV.",
    )
    parser.add_argument(
        "--min-yield", type=float, default=None,
        help="Exclude stocks with dividend yield below this percent (e.g. 2.5).",
    )
    parser.add_argument(
        "--max-payout", type=float, default=None,
        help="Exclude stocks with payout ratio above this percent (e.g. 80).",
    )
    parser.add_argument(
        "--min-score", type=float, default=None,
        help="Exclude stocks with composite score below this value (0-100).",
    )
    parser.add_argument(
        "--min-grade", type=str, default=None, choices=[letter for _, letter in GRADE_BANDS],
        help="Exclude stocks graded below this letter (e.g. B keeps A and B only).",
    )
    parser.add_argument(
        "--max-pe", type=float, default=None,
        help="Exclude stocks with trailing P/E above this value.",
    )
    parser.add_argument(
        "--max-de", type=float, default=None,
        help="Exclude stocks with debt-to-equity above this value.",
    )
    parser.add_argument(
        "--track", action="store_true",
        help="Append this run's shown top picks to the tracking log (--track-file) "
             "as hypothetical (paper) picks, so their performance can be checked later.",
    )
    parser.add_argument(
        "--track-file", type=str, default="picks_log.csv",
        help="Path to the tracking log CSV used by --track and --review (default: picks_log.csv).",
    )
    parser.add_argument(
        "--review", action="store_true",
        help="Skip screening; instead re-fetch current prices for every previously "
             "tracked pick in --track-file and report performance since each was picked.",
    )
    args = parser.parse_args()

    if args.review:
        try:
            log = review_picks(args.track_file)
        except FileNotFoundError as exc:
            print(exc)
            return
        print_review(log)
        return

    tickers = args.tickers or DEFAULT_UNIVERSE
    results = screen(tickers)

    filtered = filter_results(
        results,
        min_yield=args.min_yield,
        max_payout=args.max_payout,
        min_score=args.min_score,
        min_grade=args.min_grade,
        max_pe=args.max_pe,
        max_de=args.max_de,
    )
    if len(filtered) != len(results):
        dropped = len(results) - len(filtered)
        print(f"\nFiltered out {dropped} of {len(results)} stocks that didn't meet the given thresholds.")
    results = filtered

    print_report(results, top=args.top)

    if args.csv:
        df = to_dataframe(results)
        df.to_csv(args.csv, index=False)
        print(f"\nWrote full results to {args.csv}")

    if args.track:
        n = record_picks(results, args.track_file, top=args.top)
        print(f"\nLogged {n} pick(s) to {args.track_file}. Run with --review later to see how they performed.")


if __name__ == "__main__":
    main()
