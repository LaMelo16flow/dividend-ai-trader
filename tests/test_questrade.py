from types import SimpleNamespace

from dividend_ai.questrade import Session, authenticate, get_accounts, get_balances, place_market_order


def _fake_response(json_data, status=200):
    def raise_for_status():
        if status >= 400:
            raise RuntimeError(f"HTTP {status}")

    return SimpleNamespace(status_code=status, json=lambda: json_data, raise_for_status=raise_for_status)


def test_authenticate_saves_rotated_refresh_token(tmp_path, monkeypatch):
    token_path = str(tmp_path / "token.json")

    def fake_get(url, params=None, timeout=None, **kw):
        assert params["refresh_token"] == "old-token"
        return _fake_response({
            "access_token": "abc123", "api_server": "https://api01.iq.questrade.com/",
            "refresh_token": "new-token-rotated", "expires_in": 1800,
        })

    monkeypatch.setattr("dividend_ai.questrade.requests.get", fake_get)

    session, err = authenticate("old-token", token_path)
    assert err is None
    assert session.access_token == "abc123"
    assert session.refresh_token == "new-token-rotated"

    import json
    with open(token_path) as f:
        saved = json.load(f)
    assert saved["refresh_token"] == "new-token-rotated"


def test_authenticate_reuses_unexpired_saved_session(tmp_path, monkeypatch):
    token_path = str(tmp_path / "token.json")
    import json, time
    with open(token_path, "w") as f:
        json.dump({
            "access_token": "cached-token", "api_server": "https://api01.iq.questrade.com/",
            "refresh_token": "cached-refresh", "expires_at": time.time() + 3600,
        }, f)

    def fake_get(*a, **kw):
        raise AssertionError("should not hit the network when a valid session is cached")

    monkeypatch.setattr("dividend_ai.questrade.requests.get", fake_get)

    session, err = authenticate(None, token_path)
    assert err is None
    assert session.access_token == "cached-token"


def test_authenticate_reports_error_without_raising(tmp_path, monkeypatch):
    token_path = str(tmp_path / "token.json")

    def fake_get(*a, **kw):
        raise RuntimeError("invalid_grant")

    monkeypatch.setattr("dividend_ai.questrade.requests.get", fake_get)

    session, err = authenticate("bad-token", token_path)
    assert session is None
    assert "invalid_grant" in err


def test_authenticate_missing_token_reports_error(tmp_path):
    token_path = str(tmp_path / "token.json")
    session, err = authenticate(None, token_path)
    assert session is None
    assert "No refresh token" in err


def _session():
    return Session(access_token="tok", api_server="https://api01.iq.questrade.com/",
                    refresh_token="r", expires_at=0)


def test_get_accounts_parses_list(monkeypatch):
    def fake_get(url, headers=None, params=None, timeout=None, **kw):
        assert url == "https://api01.iq.questrade.com/v1/accounts"
        return _fake_response({"accounts": [{"number": "123", "type": "Margin"}]})

    monkeypatch.setattr("dividend_ai.questrade.requests.get", fake_get)
    accounts, err = get_accounts(_session())
    assert err is None
    assert accounts[0]["number"] == "123"


def test_get_accounts_reports_error_without_raising(monkeypatch):
    def fake_get(*a, **kw):
        raise RuntimeError("unauthorized")

    monkeypatch.setattr("dividend_ai.questrade.requests.get", fake_get)
    accounts, err = get_accounts(_session())
    assert accounts == []
    assert err == "unauthorized"


def test_get_balances_returns_combined_balance(monkeypatch):
    def fake_get(*a, **kw):
        return _fake_response({"combinedBalances": [{"cash": 5000.0, "totalEquity": 6000.0}]})

    monkeypatch.setattr("dividend_ai.questrade.requests.get", fake_get)
    balance, err = get_balances(_session(), "123")
    assert err is None
    assert balance["cash"] == 5000.0


def test_place_market_order_posts_expected_body(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, **kw):
        captured["url"] = url
        captured["body"] = json
        return _fake_response({"orders": [{"id": 1, "state": "Accepted"}]})

    monkeypatch.setattr("dividend_ai.questrade.requests.post", fake_post)
    result, err = place_market_order(_session(), "123", symbol_id=8049, quantity=5, action="Buy")
    assert err is None
    assert captured["url"] == "https://api01.iq.questrade.com/v1/accounts/123/orders"
    assert captured["body"]["symbolId"] == 8049
    assert captured["body"]["quantity"] == 5
    assert captured["body"]["action"] == "Buy"
    assert captured["body"]["orderType"] == "Market"
