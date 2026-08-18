"""Turns raw fundamentals into 0-100 sub-scores and a weighted composite.

Every score function is pure (no I/O) so it can be unit tested without
network access. `None` inputs mean "unknown" and map to a neutral score
rather than punishing a stock for missing data.
"""

from dataclasses import dataclass, field

from dividend_ai.config import GRADE_BANDS, WEIGHTS
from dividend_ai.data import StockData

NEUTRAL = 50.0


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _lerp_score(x: float, points: list[tuple[float, float]]) -> float:
    """Piecewise-linear interpolation through (x, score) control points,
    clamped to the first/last score outside the range."""
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            frac = (x - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    return points[-1][1]


def score_yield(yield_pct: float | None) -> float:
    """Rewards a healthy 2-6% yield; penalizes near-zero (not much income)
    and double-digit "yield trap" territory (often a sign of a falling
    stock price or an unsustainable payout about to be cut)."""
    if yield_pct is None:
        return NEUTRAL
    return _lerp_score(yield_pct, [
        (0, 20), (1, 40), (2.5, 80), (4, 100), (6, 85), (8, 55), (12, 20), (20, 0),
    ])


def score_growth(cagr: float | None) -> float:
    """Rewards a growing dividend (5yr CAGR). Flat-to-shrinking dividends
    score low; double-digit growth scores near 100."""
    if cagr is None:
        return NEUTRAL
    pct = cagr * 100
    return _lerp_score(pct, [
        (-20, 0), (0, 30), (3, 55), (5, 70), (8, 88), (12, 100), (25, 100),
    ])


def score_payout(payout_pct: float | None) -> float:
    """Rewards a sustainable payout ratio (~30-60% of earnings). Very low
    can mean the dividend isn't a priority; over ~80% or negative (paying
    out more than the company earns) is a cut risk."""
    if payout_pct is None:
        return NEUTRAL
    return _lerp_score(payout_pct, [
        (0, 50), (20, 65), (40, 90), (60, 90), (80, 60), (100, 25), (150, 0),
    ])


def score_consistency(streak_years: int | None) -> float:
    """Rewards years of no dividend cuts (a proxy for reliability). Caps
    out at a 10-year streak."""
    if streak_years is None:
        return NEUTRAL
    return _clamp(streak_years / 10 * 100)


def score_financial_health(debt_to_equity: float | None) -> float:
    """Lower debt-to-equity means more room to keep paying the dividend
    through a downturn. D/E is reported by yfinance as a raw ratio *100
    (e.g. 150 means 1.5x)."""
    if debt_to_equity is None:
        return NEUTRAL
    return _lerp_score(debt_to_equity, [
        (0, 100), (50, 90), (100, 70), (150, 50), (250, 25), (400, 0),
    ])


def score_valuation(trailing_pe: float | None) -> float:
    """Cheaper (lower P/E) scores higher, within a sane band. Negative or
    absent P/E (no earnings) is treated as neutral rather than punished,
    since some solid dividend payers have noisy GAAP earnings."""
    if trailing_pe is None or trailing_pe <= 0:
        return NEUTRAL
    return _lerp_score(trailing_pe, [
        (5, 90), (12, 100), (18, 80), (25, 55), (35, 30), (50, 10), (80, 0),
    ])


def dividend_growth_cagr(annual_dividends, years: int = 5) -> float | None:
    a = annual_dividends.tail(years + 1)
    if len(a) < 2 or a.iloc[0] <= 0:
        return None
    n = len(a) - 1
    return (a.iloc[-1] / a.iloc[0]) ** (1 / n) - 1


def consecutive_years_no_cut(annual_dividends, threshold: float = -0.01) -> int | None:
    growth = annual_dividends.pct_change().dropna()
    if growth.empty:
        return None
    streak = 0
    for g in reversed(growth.tolist()):
        if g >= threshold:
            streak += 1
        else:
            break
    return streak


def grade_for(score: float) -> str:
    for cutoff, letter in GRADE_BANDS:
        if score >= cutoff:
            return letter
    return "F"


@dataclass
class ScoreResult:
    ticker: str
    name: str | None
    composite: float
    grade: str
    sub_scores: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str | None = None


def score_stock(sd: StockData) -> ScoreResult:
    if sd.error:
        return ScoreResult(
            ticker=sd.ticker, name=sd.name, composite=0.0, grade="F", error=sd.error
        )

    cagr = dividend_growth_cagr(sd.annual_dividends)
    streak = consecutive_years_no_cut(sd.annual_dividends)

    sub = {
        "yield": score_yield(sd.dividend_yield_pct),
        "growth": score_growth(cagr),
        "payout": score_payout(sd.payout_ratio_pct),
        "consistency": score_consistency(streak),
        "financial_health": score_financial_health(sd.debt_to_equity),
        "valuation": score_valuation(sd.trailing_pe),
    }
    composite = sum(sub[k] * WEIGHTS[k] for k in WEIGHTS)

    notes = []
    if sd.dividend_yield_pct is not None and sd.dividend_yield_pct >= 8:
        notes.append("Very high yield — check for a yield trap (falling price / cut risk).")
    if sd.payout_ratio_pct is not None and sd.payout_ratio_pct > 100:
        notes.append("Payout ratio over 100% — paying out more than earnings.")
    if streak is not None and streak == 0:
        notes.append("Dividend was cut or flat in the most recent year.")
    if streak is not None and streak >= 10:
        notes.append(f"{streak}+ years without a cut — strong reliability track record.")
    if cagr is not None and cagr * 100 >= 8:
        notes.append(f"Dividend growing fast (~{cagr*100:.1f}%/yr over 5yr).")
    if sd.annual_dividends.empty:
        notes.append("No dividend history found — may not currently pay a dividend.")

    return ScoreResult(
        ticker=sd.ticker,
        name=sd.name,
        composite=round(composite, 1),
        grade=grade_for(composite),
        sub_scores={k: round(v, 1) for k, v in sub.items()},
        raw={
            "price": sd.price,
            "dividend_yield_pct": sd.dividend_yield_pct,
            "payout_ratio_pct": sd.payout_ratio_pct,
            "dividend_cagr_5y_pct": round(cagr * 100, 2) if cagr is not None else None,
            "no_cut_streak_years": streak,
            "trailing_pe": sd.trailing_pe,
            "debt_to_equity": sd.debt_to_equity,
            "sector": sd.sector,
        },
        notes=notes,
    )
