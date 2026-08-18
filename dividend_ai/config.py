"""Default universes and scoring weights."""

# A sample of well-known long-running dividend payers (Aristocrats/Kings and
# a few large, stable dividend blue chips). This is a starting point, not
# a recommendation — pass your own tickers via --tickers to screen anything.
DEFAULT_UNIVERSE = [
    "JNJ", "PG", "KO", "PEP", "MMM", "MCD", "WMT", "XOM", "CVX", "ABBV",
    "PFE", "T", "VZ", "IBM", "CAT", "HD", "LOW", "MO", "PM", "CL",
    "KMB", "GD", "ADP", "SYY", "SHW", "ITW", "EMR", "APD", "O", "MDT",
    "ABT", "TGT", "CVS", "DUK", "SO", "NEE", "AFL", "BEN", "CB", "TROW",
]

# Composite score weights (must sum to 1.0). Adjust to taste — e.g. raise
# "growth" if you care more about future income than current yield.
WEIGHTS = {
    "yield": 0.20,
    "growth": 0.20,
    "payout": 0.20,
    "consistency": 0.20,
    "financial_health": 0.15,
    "valuation": 0.05,
}

GRADE_BANDS = [
    (85, "A"),
    (70, "B"),
    (55, "C"),
    (40, "D"),
    (0, "F"),
]

# Well-known monthly dividend payers. DEFAULT_UNIVERSE is almost entirely
# quarterly payers (O is the one exception), so a monthly-vs-quarterly
# strategy comparison needs a separate candidate pool to draw from.
MONTHLY_PAYER_UNIVERSE = [
    "O", "MAIN", "STAG", "AGNC", "ARR", "GAIN", "LTC", "PFLT", "APLE", "EPR",
]

# The combined pool the backtesting engine classifies by actual observed
# dividend frequency (see dividend_ai.backtest.classify_frequency) rather
# than trusting these lists' names — a payer can change frequency over time.
BACKTEST_UNIVERSE = sorted(set(DEFAULT_UNIVERSE) | set(MONTHLY_PAYER_UNIVERSE))

# Companies with a documented dividend cut or suspension in their yfinance
# dividend history. DEFAULT_UNIVERSE was deliberately curated to *avoid*
# cutters, so training a cut-risk classifier on it alone would see almost no
# positive examples — this list exists purely to give that model something
# to learn a cut actually looks like.
KNOWN_CUTTER_UNIVERSE = [
    "GE", "F", "KHC", "OXY", "KMI", "WBA", "VFC", "T", "DIS", "MRO",
    "COTY", "C", "BAC", "WFC", "AIG", "PBI", "M", "IPG",
]

# Combined training pool for the dividend cut-risk model — needs both
# stable payers (negative examples) and past cutters (positive examples).
CUT_RISK_TRAINING_UNIVERSE = sorted(set(BACKTEST_UNIVERSE) | set(KNOWN_CUTTER_UNIVERSE))
