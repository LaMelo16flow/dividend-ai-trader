"""A small, honest dividend cut-risk classifier.

This SUPPLEMENTS the deterministic sustainability checks in scoring.py —
it never replaces them, and its output is a statistical estimate from a
small, imbalanced dataset, not a fact. Every prediction should be shown
next to the model's cross-validated metrics so the reader can judge how
much to trust it.

Data reality check: free sources don't give point-in-time historical
fundamentals (payout ratio, debt, FCF) far enough back to build a
meaningful training set. What IS available for decades of history, for
free, is daily price + dividend payments (dividend_ai.history). So every
feature here is derived only from price and dividend history — dividend
growth trend, yield level and "spikiness", price momentum, payment
consistency, and prior cut history. That's a real limitation: it can't
see balance-sheet deterioration directly, only its usual side effects
(falling price, decelerating dividend growth, elevated yield).
"""

import os
import statistics
from dataclasses import dataclass, field

import pandas as pd

from dividend_ai.history import get_history

FEATURE_COLUMNS = [
    "div_growth_1y",
    "div_growth_3y",
    "div_growth_deceleration",
    "yield_now",
    "yield_spike_vs_3y_avg",
    "price_return_1y",
    "payment_consistency_std",
    "had_cut_in_past_4y",
]

CUT_THRESHOLD = -0.05  # a >5% YoY drop in annual dividend counts as a cut
DEFAULT_MODEL_PATH = "cut_risk_model.joblib"


def _annual_dividends(history: pd.DataFrame) -> pd.Series:
    s = history["Dividends"].resample("YE").sum()
    s.index = s.index.year
    current_year = pd.Timestamp.now().year
    return s[s.index < current_year]


def _annual_price(history: pd.DataFrame) -> pd.Series:
    s = history["Close"].resample("YE").last()
    s.index = s.index.year
    current_year = pd.Timestamp.now().year
    return s[s.index < current_year]


def _compute_features(annual_div: pd.Series, annual_price: pd.Series, year: int) -> dict | None:
    if year not in annual_div.index or year not in annual_price.index:
        return None
    div_y = annual_div.get(year)
    price_y = annual_price.get(year)
    if div_y is None or div_y <= 0 or price_y is None or price_y <= 0:
        return None

    div_y1 = annual_div.get(year - 1)
    div_y3 = annual_div.get(year - 3)
    price_y1 = annual_price.get(year - 1)

    div_growth_1y = (div_y / div_y1 - 1) if div_y1 and div_y1 > 0 else 0.0
    div_growth_3y = ((div_y / div_y3) ** (1 / 3) - 1) if div_y3 and div_y3 > 0 else 0.0
    price_return_1y = (price_y / price_y1 - 1) if price_y1 and price_y1 > 0 else 0.0

    yield_now = div_y / price_y * 100
    trailing_yields = []
    for yr in range(year - 2, year + 1):
        if yr in annual_div.index and yr in annual_price.index and annual_price[yr] > 0:
            trailing_yields.append(annual_div[yr] / annual_price[yr] * 100)
    avg_yield_3y = sum(trailing_yields) / len(trailing_yields) if trailing_yields else yield_now
    yield_spike = (yield_now / avg_yield_3y - 1) if avg_yield_3y > 0 else 0.0

    yoy_growths = []
    for yr in range(year - 2, year + 1):
        if yr in annual_div.index and (yr - 1) in annual_div.index and annual_div[yr - 1] > 0:
            yoy_growths.append(annual_div[yr] / annual_div[yr - 1] - 1)
    payment_consistency_std = statistics.pstdev(yoy_growths) if len(yoy_growths) >= 2 else 0.0

    had_past_cut = any(
        yr in annual_div.index and (yr - 1) in annual_div.index and annual_div[yr - 1] > 0
        and annual_div[yr] / annual_div[yr - 1] - 1 <= CUT_THRESHOLD
        for yr in range(year - 4, year)
    )

    return {
        "div_growth_1y": div_growth_1y,
        "div_growth_3y": div_growth_3y,
        "div_growth_deceleration": div_growth_1y - div_growth_3y,
        "yield_now": yield_now,
        "yield_spike_vs_3y_avg": yield_spike,
        "price_return_1y": price_return_1y,
        "payment_consistency_std": payment_consistency_std,
        "had_cut_in_past_4y": 1.0 if had_past_cut else 0.0,
    }


def _compute_label(annual_div: pd.Series, year: int) -> int | None:
    div_y = annual_div.get(year)
    div_y1 = annual_div.get(year + 1)
    if div_y is None or div_y <= 0 or div_y1 is None:
        return None
    return 1 if (div_y1 / div_y - 1) <= CUT_THRESHOLD else 0


def build_training_dataset(
    tickers: list[str], cache_dir: str = ".cache", on_progress=None
) -> tuple[pd.DataFrame, list[str]]:
    """One row per (ticker, year) with features known as of that year-end
    and a label of whether the *following* year's dividend was cut."""
    rows = []
    errors = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        if on_progress:
            on_progress(i, total, ticker)

        hist, err = get_history(ticker, cache_dir=cache_dir)
        if err:
            errors.append(f"{ticker}: {err}")
            continue

        annual_div = _annual_dividends(hist)
        annual_price = _annual_price(hist)
        if annual_div.empty:
            continue

        for year in annual_div.index:
            features = _compute_features(annual_div, annual_price, int(year))
            label = _compute_label(annual_div, int(year))
            if features is None or label is None:
                continue
            rows.append({"ticker": ticker, "year": int(year), "label": label, **features})

    return pd.DataFrame(rows), errors


def train_model(dataset: pd.DataFrame, min_samples: int = 20):
    """Returns (fitted_pipeline_or_None, metrics_dict)."""
    if len(dataset) < min_samples:
        return None, {"error": f"Not enough samples to train ({len(dataset)} < {min_samples})."}

    y = dataset["label"].to_numpy()
    X = dataset[FEATURE_COLUMNS].to_numpy()
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return None, {
            "error": "Training data has only one class (no cuts or no non-cuts found) — "
                     "can't train a classifier. Broaden the ticker universe.",
            "n_samples": len(dataset), "n_positive": n_pos, "n_negative": n_neg,
        }

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))

    n_splits = min(5, n_pos, n_neg)
    metrics: dict = {"n_samples": len(dataset), "n_positive": n_pos, "n_negative": n_neg}
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics.update({
            "cv_folds": n_splits,
            "roc_auc": round(roc_auc_score(y, proba), 3),
            "accuracy": round(accuracy_score(y, preds), 3),
            "precision": round(precision_score(y, preds, zero_division=0), 3),
            "recall": round(recall_score(y, preds, zero_division=0), 3),
        })
    else:
        metrics["cv_folds"] = 0
        metrics["note"] = "Too few samples in the minority class for cross-validation; metrics unavailable."

    model.fit(X, y)
    return model, metrics


def save_model(model, path: str = DEFAULT_MODEL_PATH) -> None:
    import joblib
    joblib.dump(model, path)


def load_model(path: str = DEFAULT_MODEL_PATH):
    import joblib
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def predict_cut_risk(
    model, ticker: str, cache_dir: str = ".cache"
) -> tuple[float | None, dict | None, str | None]:
    """Returns (probability_of_cut, info_dict, error)."""
    hist, err = get_history(ticker, cache_dir=cache_dir)
    if err:
        return None, None, err

    annual_div = _annual_dividends(hist)
    annual_price = _annual_price(hist)
    if annual_div.empty or len(annual_div) < 4:
        return None, None, "Not enough dividend history to estimate cut risk (need several years of payments)."

    target_year = int(annual_div.index.max())
    features = _compute_features(annual_div, annual_price, target_year)
    if features is None:
        return None, None, "Could not compute cut-risk features from available price/dividend data."

    X = pd.DataFrame([features])[FEATURE_COLUMNS].to_numpy()
    proba = float(model.predict_proba(X)[0][1])
    return proba, {"as_of_year": target_year, **features}, None
