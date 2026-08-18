import pandas as pd
import pytest

from dividend_ai.cut_risk import (
    FEATURE_COLUMNS,
    _annual_dividends,
    _annual_price,
    _compute_features,
    _compute_label,
    build_training_dataset,
    predict_cut_risk,
    train_model,
)


def _daily_history(year_prices: dict[int, float], year_dividends: dict[int, float]) -> pd.DataFrame:
    """One row per Dec-31 for each year — resample('YE') collapses to the same thing."""
    years = sorted(set(year_prices) | set(year_dividends))
    idx = pd.DatetimeIndex([pd.Timestamp(f"{y}-12-31") for y in years])
    close = [year_prices.get(y, list(year_prices.values())[0]) for y in years]
    div = [year_dividends.get(y, 0.0) for y in years]
    return pd.DataFrame({"Close": close, "Dividends": div}, index=idx)


def test_compute_features_and_label_no_cut():
    # steady 5% dividend growth every year, label should be "no cut"
    prices = {y: 100.0 for y in range(2015, 2024)}
    divs = {y: 1.0 * (1.05 ** (y - 2015)) for y in range(2015, 2024)}
    annual_div = pd.Series(divs)
    annual_price = pd.Series(prices)

    features = _compute_features(annual_div, annual_price, 2020)
    assert features is not None
    assert features["div_growth_1y"] == pytest.approx(0.05, abs=1e-6)
    assert features["had_cut_in_past_4y"] == 0.0

    label = _compute_label(annual_div, 2020)
    assert label == 0


def test_compute_label_detects_cut():
    annual_div = pd.Series({2019: 1.0, 2020: 1.0, 2021: 0.5})  # 50% cut in 2021
    label = _compute_label(annual_div, 2020)
    assert label == 1


def test_compute_label_ignores_small_decrease():
    annual_div = pd.Series({2019: 1.0, 2020: 1.0, 2021: 0.97})  # 3% dip, below threshold
    label = _compute_label(annual_div, 2020)
    assert label == 0


def test_compute_features_flags_past_cut():
    annual_div = pd.Series({2016: 1.0, 2017: 1.0, 2018: 0.4, 2019: 0.4, 2020: 0.42})
    annual_price = pd.Series({y: 50.0 for y in range(2016, 2021)})
    features = _compute_features(annual_div, annual_price, 2020)
    assert features["had_cut_in_past_4y"] == 1.0


def test_compute_features_returns_none_without_price_or_dividend():
    annual_div = pd.Series({2020: 1.0})
    annual_price = pd.Series({2020: 50.0})
    assert _compute_features(annual_div, annual_price, 2021) is None


def test_annual_dividends_excludes_current_incomplete_year():
    current_year = pd.Timestamp.now().year
    hist = _daily_history({current_year - 1: 100.0, current_year: 100.0},
                           {current_year - 1: 2.0, current_year: 1.0})
    annual = _annual_dividends(hist)
    assert current_year not in annual.index
    assert (current_year - 1) in annual.index


def test_annual_price_matches_year_end_close():
    hist = _daily_history({2020: 42.0}, {2020: 1.0})
    annual = _annual_price(hist)
    assert annual[2020] == 42.0


def test_build_training_dataset_labels_engineered_cut(monkeypatch):
    stable_hist = _daily_history(
        {y: 100.0 for y in range(2010, 2024)},
        {y: 1.0 * (1.05 ** (y - 2010)) for y in range(2010, 2024)},
    )
    cutter_divs = {y: 2.0 for y in range(2010, 2018)}
    cutter_divs.update({y: 0.5 for y in range(2018, 2024)})
    cutter_hist = _daily_history({y: 30.0 for y in range(2010, 2024)}, cutter_divs)

    def fake_get_history(ticker, cache_dir=".cache", max_age_hours=24):
        return (cutter_hist, None) if ticker == "CUTTER" else (stable_hist, None)

    monkeypatch.setattr("dividend_ai.cut_risk.get_history", fake_get_history)

    dataset, errors = build_training_dataset(["STABLE", "CUTTER"])
    assert errors == []
    assert set(FEATURE_COLUMNS).issubset(dataset.columns)

    cutter_rows = dataset[(dataset["ticker"] == "CUTTER") & (dataset["year"] == 2017)]
    assert len(cutter_rows) == 1
    assert cutter_rows.iloc[0]["label"] == 1

    stable_rows = dataset[(dataset["ticker"] == "STABLE") & (dataset["year"] == 2015)]
    assert len(stable_rows) == 1
    assert stable_rows.iloc[0]["label"] == 0


def test_build_training_dataset_reports_fetch_errors(monkeypatch):
    def fake_get_history(ticker, cache_dir=".cache", max_age_hours=24):
        return None, "not found"

    monkeypatch.setattr("dividend_ai.cut_risk.get_history", fake_get_history)
    dataset, errors = build_training_dataset(["ZZZ"])
    assert dataset.empty
    assert "ZZZ" in errors[0]


# ---- train_model ----

def _synthetic_dataset(n_per_class=15):
    import random
    random.seed(0)
    rows = []
    for i in range(n_per_class):
        rows.append({"ticker": f"GOOD{i}", "year": 2020, "label": 0,
                     "div_growth_1y": 0.05 + random.uniform(-0.01, 0.01),
                     "div_growth_3y": 0.05, "div_growth_deceleration": 0.0,
                     "yield_now": 2.5, "yield_spike_vs_3y_avg": 0.0,
                     "price_return_1y": 0.08, "payment_consistency_std": 0.01,
                     "had_cut_in_past_4y": 0.0})
        rows.append({"ticker": f"BAD{i}", "year": 2020, "label": 1,
                     "div_growth_1y": -0.10 + random.uniform(-0.01, 0.01),
                     "div_growth_3y": -0.05, "div_growth_deceleration": -0.15,
                     "yield_now": 9.0, "yield_spike_vs_3y_avg": 0.8,
                     "price_return_1y": -0.30, "payment_consistency_std": 0.2,
                     "had_cut_in_past_4y": 1.0})
    return pd.DataFrame(rows)


def test_train_model_too_few_samples():
    dataset = _synthetic_dataset(n_per_class=2)
    model, metrics = train_model(dataset, min_samples=20)
    assert model is None
    assert "error" in metrics


def test_train_model_single_class_reports_error():
    dataset = _synthetic_dataset(n_per_class=15)
    dataset = dataset[dataset["label"] == 0]
    model, metrics = train_model(dataset, min_samples=5)
    assert model is None
    assert "error" in metrics


def test_train_model_fits_and_reports_cv_metrics():
    dataset = _synthetic_dataset(n_per_class=15)
    model, metrics = train_model(dataset, min_samples=20)
    assert model is not None
    assert metrics["n_samples"] == 30
    assert metrics["cv_folds"] >= 2
    assert 0.0 <= metrics["roc_auc"] <= 1.0

    # a clearly "good" and clearly "bad" profile should predict sensibly
    good = pd.DataFrame([{
        "div_growth_1y": 0.05, "div_growth_3y": 0.05, "div_growth_deceleration": 0.0,
        "yield_now": 2.5, "yield_spike_vs_3y_avg": 0.0, "price_return_1y": 0.08,
        "payment_consistency_std": 0.01, "had_cut_in_past_4y": 0.0,
    }])[FEATURE_COLUMNS]
    bad = pd.DataFrame([{
        "div_growth_1y": -0.10, "div_growth_3y": -0.05, "div_growth_deceleration": -0.15,
        "yield_now": 9.0, "yield_spike_vs_3y_avg": 0.8, "price_return_1y": -0.30,
        "payment_consistency_std": 0.2, "had_cut_in_past_4y": 1.0,
    }])[FEATURE_COLUMNS]
    assert model.predict_proba(bad.to_numpy())[0][1] > model.predict_proba(good.to_numpy())[0][1]


def test_predict_cut_risk_end_to_end(monkeypatch):
    dataset = _synthetic_dataset(n_per_class=15)
    model, _ = train_model(dataset, min_samples=20)
    assert model is not None

    hist = _daily_history({y: 100.0 for y in range(2015, 2024)},
                           {y: 1.0 * (1.05 ** (y - 2015)) for y in range(2015, 2024)})

    def fake_get_history(ticker, cache_dir=".cache", max_age_hours=24):
        return hist, None

    monkeypatch.setattr("dividend_ai.cut_risk.get_history", fake_get_history)

    proba, info, err = predict_cut_risk(model, "GOODCO")
    assert err is None
    assert 0.0 <= proba <= 1.0
    assert info["as_of_year"] == 2023


def test_predict_cut_risk_reports_insufficient_history(monkeypatch):
    hist = _daily_history({2023: 100.0}, {2023: 1.0})

    def fake_get_history(ticker, cache_dir=".cache", max_age_hours=24):
        return hist, None

    monkeypatch.setattr("dividend_ai.cut_risk.get_history", fake_get_history)
    proba, info, err = predict_cut_risk(object(), "TOOFEW")
    assert proba is None
    assert err is not None
