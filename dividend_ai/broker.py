"""Paper-trading broker integration via Alpaca (https://alpaca.markets).

Always talks to Alpaca's *paper* trading endpoint (simulated money, real
market mechanics) — never live. Requires a free Alpaca account and API
keys; nothing here works without them. Every function catches broker/API
errors and reports them on the returned object instead of raising, so a
GUI can display a clean message instead of crashing.
"""

from dataclasses import dataclass, field

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest


def get_client(api_key: str, secret_key: str) -> TradingClient:
    return TradingClient(api_key, secret_key, paper=True)


@dataclass
class AccountInfo:
    cash: float | None = None
    buying_power: float | None = None
    portfolio_value: float | None = None
    equity: float | None = None
    error: str | None = None


@dataclass
class Position:
    ticker: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float


@dataclass
class OrderResult:
    ticker: str
    side: str
    status: str | None = None
    filled_qty: float | None = None
    error: str | None = None


def fetch_account(client: TradingClient) -> AccountInfo:
    try:
        a = client.get_account()
        return AccountInfo(
            cash=float(a.cash),
            buying_power=float(a.buying_power),
            portfolio_value=float(a.portfolio_value),
            equity=float(a.equity),
        )
    except Exception as exc:  # noqa: BLE001 - broker call, many failure modes (auth, network)
        return AccountInfo(error=str(exc))


def fetch_positions(client: TradingClient) -> tuple[list[Position], str | None]:
    try:
        raw = client.get_all_positions()
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)

    positions = [
        Position(
            ticker=p.symbol,
            qty=float(p.qty),
            avg_entry_price=float(p.avg_entry_price),
            current_price=float(p.current_price),
            market_value=float(p.market_value),
            unrealized_pl=float(p.unrealized_pl),
            unrealized_plpc=float(p.unrealized_plpc) * 100,
        )
        for p in raw
    ]
    return positions, None


def submit_buy(client: TradingClient, ticker: str, dollar_amount: float) -> OrderResult:
    try:
        order = client.submit_order(MarketOrderRequest(
            symbol=ticker,
            notional=round(dollar_amount, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        ))
        return OrderResult(ticker=ticker, side="buy", status=str(order.status))
    except Exception as exc:  # noqa: BLE001
        return OrderResult(ticker=ticker, side="buy", error=str(exc))


def close_position(client: TradingClient, ticker: str) -> OrderResult:
    try:
        order = client.close_position(ticker)
        return OrderResult(ticker=ticker, side="sell", status=str(order.status))
    except Exception as exc:  # noqa: BLE001
        return OrderResult(ticker=ticker, side="sell", error=str(exc))


def fetch_recent_orders(client: TradingClient, limit: int = 20) -> tuple[list[dict], str | None]:
    try:
        orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit))
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)

    rows = [{
        "ticker": o.symbol,
        "side": o.side.value if o.side else None,
        "qty": float(o.qty) if o.qty else None,
        "notional": float(o.notional) if o.notional else None,
        "status": o.status.value if o.status else None,
        "submitted_at": o.submitted_at,
        "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
    } for o in orders]
    return rows, None
