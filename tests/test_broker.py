from types import SimpleNamespace

from dividend_ai.broker import (
    close_position,
    fetch_account,
    fetch_positions,
    fetch_recent_orders,
    submit_buy,
)


class _RaisingClient:
    def get_account(self):
        raise RuntimeError("invalid API key")

    def get_all_positions(self):
        raise RuntimeError("invalid API key")

    def submit_order(self, order):
        raise RuntimeError("insufficient buying power")

    def close_position(self, ticker):
        raise RuntimeError("no position to close")

    def get_orders(self, request):
        raise RuntimeError("invalid API key")


class _OkClient:
    def get_account(self):
        return SimpleNamespace(cash="1000.50", buying_power="2000.00",
                                portfolio_value="3000.25", equity="3000.25")

    def get_all_positions(self):
        return [SimpleNamespace(symbol="JNJ", qty="5", avg_entry_price="150.00",
                                 current_price="160.00", market_value="800.00",
                                 unrealized_pl="50.00", unrealized_plpc="0.0667")]

    def submit_order(self, order):
        return SimpleNamespace(status="accepted")

    def close_position(self, ticker):
        return SimpleNamespace(status="accepted")


def test_fetch_account_reports_error_without_raising():
    result = fetch_account(_RaisingClient())
    assert result.error == "invalid API key"
    assert result.cash is None


def test_fetch_account_parses_values():
    result = fetch_account(_OkClient())
    assert result.error is None
    assert result.cash == 1000.50
    assert result.buying_power == 2000.00


def test_fetch_positions_reports_error_without_raising():
    positions, error = fetch_positions(_RaisingClient())
    assert positions == []
    assert error == "invalid API key"


def test_fetch_positions_parses_values():
    positions, error = fetch_positions(_OkClient())
    assert error is None
    assert len(positions) == 1
    assert positions[0].ticker == "JNJ"
    assert positions[0].qty == 5.0
    assert round(positions[0].unrealized_plpc, 2) == 6.67


def test_submit_buy_reports_error_without_raising():
    result = submit_buy(_RaisingClient(), "AAPL", 100.0)
    assert result.error == "insufficient buying power"
    assert result.ticker == "AAPL"


def test_submit_buy_succeeds():
    result = submit_buy(_OkClient(), "AAPL", 100.0)
    assert result.error is None
    assert result.status == "accepted"


def test_close_position_reports_error_without_raising():
    result = close_position(_RaisingClient(), "AAPL")
    assert result.error == "no position to close"


def test_fetch_recent_orders_reports_error_without_raising():
    orders, error = fetch_recent_orders(_RaisingClient())
    assert orders == []
    assert error == "invalid API key"
