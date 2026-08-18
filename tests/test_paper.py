from dividend_ai.paper import buy, get_positions, get_summary, load_account, save_account, sell


def _new_account(cash=1000.0):
    return {"cash": cash, "starting_cash": cash, "holdings": {}, "transactions": []}


def test_buy_deducts_cash_and_opens_position():
    account = _new_account()
    err = buy(account, "AAA", 200.0, price=50.0)
    assert err is None
    assert account["cash"] == 800.0
    assert account["holdings"]["AAA"]["qty"] == 4.0
    assert account["holdings"]["AAA"]["avg_entry_price"] == 50.0
    assert len(account["transactions"]) == 1


def test_buy_averages_entry_price_across_two_buys():
    account = _new_account()
    buy(account, "AAA", 100.0, price=50.0)   # 2 shares @ 50
    buy(account, "AAA", 100.0, price=100.0)  # 1 share @ 100
    holding = account["holdings"]["AAA"]
    assert holding["qty"] == 3.0
    assert round(holding["avg_entry_price"], 2) == 66.67


def test_buy_rejects_insufficient_cash():
    account = _new_account(cash=50.0)
    err = buy(account, "AAA", 200.0, price=50.0)
    assert err is not None
    assert account["cash"] == 50.0
    assert "AAA" not in account["holdings"]


def test_buy_rejects_missing_price():
    account = _new_account()
    err = buy(account, "AAA", 100.0, price=None)
    assert err is not None
    assert account["cash"] == 1000.0


def test_sell_all_closes_position_and_credits_cash():
    account = _new_account()
    buy(account, "AAA", 200.0, price=50.0)  # 4 shares
    err = sell(account, "AAA", price=60.0)
    assert err is None
    assert "AAA" not in account["holdings"]
    assert account["cash"] == 800.0 + 4 * 60.0


def test_sell_partial_reduces_qty():
    account = _new_account()
    buy(account, "AAA", 200.0, price=50.0)  # 4 shares
    err = sell(account, "AAA", price=50.0, qty=1.0)
    assert err is None
    assert account["holdings"]["AAA"]["qty"] == 3.0


def test_sell_rejects_no_position():
    account = _new_account()
    err = sell(account, "AAA", price=50.0)
    assert err is not None


def test_get_positions_computes_unrealized_pl():
    account = _new_account()
    buy(account, "AAA", 100.0, price=50.0)  # 2 shares @ 50
    positions = get_positions(account, {"AAA": 75.0})
    assert positions[0]["market_value"] == 150.0
    assert positions[0]["unrealized_pl"] == 50.0
    assert positions[0]["unrealized_plpc"] == 50.0


def test_get_positions_handles_missing_price():
    account = _new_account()
    buy(account, "AAA", 100.0, price=50.0)
    positions = get_positions(account, {})
    assert positions[0]["current_price"] is None
    assert positions[0]["market_value"] is None
    assert positions[0]["unrealized_pl"] is None


def test_get_summary_matches_cash_plus_holdings():
    account = _new_account()
    buy(account, "AAA", 100.0, price=50.0)  # cash 900, 2 shares @ 50
    summary = get_summary(account, {"AAA": 75.0})
    assert summary["cash"] == 900.0
    assert summary["holdings_value"] == 150.0
    assert summary["portfolio_value"] == 1050.0
    assert summary["total_pl"] == 50.0
    assert summary["total_pl_pct"] == 5.0


def test_save_and_load_account_roundtrip(tmp_path):
    path = str(tmp_path / "account.json")
    account = _new_account()
    buy(account, "AAA", 100.0, price=50.0)
    save_account(account, path)

    loaded = load_account(path)
    assert loaded["cash"] == account["cash"]
    assert loaded["holdings"] == account["holdings"]


def test_load_account_creates_fresh_when_missing(tmp_path):
    path = str(tmp_path / "does_not_exist.json")
    account = load_account(path, starting_cash=500.0)
    assert account["cash"] == 500.0
    assert account["holdings"] == {}
