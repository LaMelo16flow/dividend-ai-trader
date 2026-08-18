"""Questrade REST API client for practice (or live) accounts.

Questrade auth is a manual OAuth2 refresh-token flow: you generate a
refresh token yourself from Questrade's App Hub (my.questrade.com, or
practicelogin.questrade.com for a practice account) and paste it in here.
Every token exchange returns a *new* refresh token and invalidates the old
one, so the current token is persisted to a local JSON file after each
exchange — losing that file means generating a fresh token from Questrade.

None of this has been exercised against a real Questrade account while
building it; the shapes below follow Questrade's published API docs
(questrade.com/api/documentation) as of this writing, but Questrade can
change endpoint/field details without notice, and the practice-vs-live
token-exchange domain in particular is worth double-checking against your
own account if authentication fails. Every function catches request
errors and reports them on its return value instead of raising, so the
GUI can surface Questrade's own error message instead of crashing.
"""

import json
import os
import time
from dataclasses import dataclass

import requests

DEFAULT_TOKEN_PATH = "questrade_token.json"
TOKEN_ENDPOINT = "https://login.questrade.com/oauth2/token"
REQUEST_TIMEOUT = 15


@dataclass
class Session:
    access_token: str
    api_server: str
    refresh_token: str
    expires_at: float


def _load_token_state(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_token_state(state: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def authenticate(refresh_token: str | None, token_path: str = DEFAULT_TOKEN_PATH) -> tuple[Session | None, str | None]:
    """Returns a valid Session, refreshing/exchanging tokens as needed.

    Reuses a still-valid persisted access token when possible. If
    `refresh_token` is given and differs from what's persisted, it's used
    for a fresh exchange (e.g. the user pasted a new one). Every exchange's
    resulting refresh token is saved immediately, since Questrade
    invalidates the old one on use.
    """
    state = _load_token_state(token_path)

    if state and (not refresh_token or refresh_token == state.get("refresh_token")):
        if state.get("expires_at", 0) > time.time() + 30:
            return Session(
                access_token=state["access_token"],
                api_server=state["api_server"],
                refresh_token=state["refresh_token"],
                expires_at=state["expires_at"],
            ), None
        refresh_token = state.get("refresh_token")

    if not refresh_token:
        return None, (
            "No refresh token provided or saved. Generate one from Questrade's "
            "App Hub (my.questrade.com → API Centre → Personal Apps)."
        )

    try:
        resp = requests.get(
            TOKEN_ENDPOINT,
            params={"grant_type": "refresh_token", "refresh_token": refresh_token},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 - broker call, many failure modes
        return None, f"Questrade authentication failed: {exc}"

    new_state = {
        "access_token": data["access_token"],
        "api_server": data["api_server"],
        "refresh_token": data["refresh_token"],
        "expires_at": time.time() + data.get("expires_in", 1800) - 60,
    }
    _save_token_state(new_state, token_path)
    return Session(**new_state), None


def _get(session: Session, path: str, params: dict | None = None) -> tuple[dict | None, str | None]:
    try:
        resp = requests.get(
            f"{session.api_server.rstrip('/')}/{path.lstrip('/')}",
            headers={"Authorization": f"Bearer {session.access_token}"},
            params=params, timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json(), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def get_accounts(session: Session) -> tuple[list[dict], str | None]:
    data, err = _get(session, "v1/accounts")
    return (data.get("accounts", []) if data else []), err


def get_balances(session: Session, account_id: str) -> tuple[dict | None, str | None]:
    data, err = _get(session, f"v1/accounts/{account_id}/balances")
    if err or not data:
        return None, err
    combined = (data.get("combinedBalances") or [{}])[0]
    return combined, None


def get_positions(session: Session, account_id: str) -> tuple[list[dict], str | None]:
    data, err = _get(session, f"v1/accounts/{account_id}/positions")
    return (data.get("positions", []) if data else []), err


def search_symbol(session: Session, prefix: str) -> tuple[list[dict], str | None]:
    data, err = _get(session, "v1/symbols/search", params={"prefix": prefix})
    return (data.get("symbols", []) if data else []), err


def get_quote(session: Session, symbol_id: int) -> tuple[dict | None, str | None]:
    data, err = _get(session, f"v1/markets/quotes/{symbol_id}")
    if err or not data:
        return None, err
    quotes = data.get("quotes") or []
    return (quotes[0] if quotes else None), None


def place_market_order(
    session: Session, account_id: str, symbol_id: int, quantity: int, action: str = "Buy",
) -> tuple[dict | None, str | None]:
    """Submits a Day market order. `action` is "Buy" or "Sell"."""
    body = {
        "symbolId": symbol_id,
        "quantity": quantity,
        "action": action,
        "orderType": "Market",
        "timeInForce": "Day",
        "primaryRoute": "AUTO",
        "secondaryRoute": "AUTO",
    }
    try:
        resp = requests.post(
            f"{session.api_server.rstrip('/')}/v1/accounts/{account_id}/orders",
            headers={"Authorization": f"Bearer {session.access_token}"},
            json=body, timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json(), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
