"""Streamlit GUI for the dividend screener. Run with: streamlit run app.py"""

import json
import os

import pandas as pd
import streamlit as st

from dividend_ai.backtest import build_strategy_universes, compare_strategies
from dividend_ai.broker import (
    close_position,
    fetch_account,
    fetch_positions,
    fetch_recent_orders,
    get_client,
    submit_buy,
)
from dividend_ai.config import BACKTEST_UNIVERSE, CUT_RISK_TRAINING_UNIVERSE, DEFAULT_UNIVERSE, GRADE_BANDS, WEIGHTS
from dividend_ai.cut_risk import (
    DEFAULT_MODEL_PATH as CUT_RISK_MODEL_PATH,
    build_training_dataset as cr_build_training_dataset,
    load_model as cr_load_model,
    predict_cut_risk as cr_predict_cut_risk,
    save_model as cr_save_model,
    train_model as cr_train_model,
)
from dividend_ai.data import fetch_stock_data
from dividend_ai.paper import (
    DEFAULT_STARTING_CASH,
    buy as paper_buy,
    get_positions as paper_get_positions,
    get_summary as paper_get_summary,
    load_account as paper_load_account,
    reset_account as paper_reset_account,
    save_account as paper_save_account,
    sell as paper_sell,
)
from dividend_ai.questrade import (
    authenticate as qt_authenticate,
    get_accounts as qt_get_accounts,
    get_balances as qt_get_balances,
    get_positions as qt_get_positions,
    place_market_order as qt_place_market_order,
    search_symbol as qt_search_symbol,
)
from dividend_ai.screener import filter_results, screen, to_dataframe
from dividend_ai.tracker import record_picks, review_picks

GRADE_LETTERS = [letter for _, letter in GRADE_BANDS]
GRADE_EMOJI = {"A": "🟢 A", "B": "🟢 B", "C": "🟡 C", "D": "🟠 D", "F": "🔴 F"}

st.set_page_config(page_title="Dividend AI Screener", page_icon="💰", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    h1 { background: linear-gradient(90deg, #22c55e, #0ea5e9);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         font-weight: 800; }
    div[data-testid="stMetric"] {
        background: rgba(125,125,125,0.08); border-radius: 12px;
        padding: 0.8rem 1rem; border: 1px solid rgba(125,125,125,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💰 Dividend AI Screener")
st.caption(
    "Rule-based ranking by yield quality, growth, payout sustainability, consistency, "
    "financial health, and valuation. Not financial advice — data via yfinance can be "
    "incomplete, delayed, or wrong."
)

st.session_state.setdefault("results", None)
st.session_state.setdefault("shown", None)

with st.sidebar:
    st.header("Universe")
    mode = st.radio("Tickers", ["Default dividend universe", "Custom"], label_visibility="collapsed")
    if mode == "Custom":
        raw = st.text_area("Enter tickers (space or comma separated)", "JNJ KO PG")
        tickers = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
    else:
        tickers = DEFAULT_UNIVERSE
        st.caption(f"{len(tickers)} built-in tickers.")

    top_n = st.slider("Show top N", 1, max(len(tickers), 1), min(15, len(tickers)) or 1)

    with st.expander("Filters (optional)"):
        min_yield = st.slider("Min dividend yield %", 0.0, 15.0, 0.0, 0.5)
        max_payout = st.slider("Max payout ratio %", 0, 200, 200, 5)
        min_score = st.slider("Min composite score", 0, 100, 0, 5)
        min_grade = st.selectbox("Min grade", ["Any"] + GRADE_LETTERS)
        max_pe = st.slider("Max trailing P/E", 0, 200, 200, 5)
        max_de = st.slider("Max debt-to-equity", 0, 500, 500, 10)

    run = st.button("🔍 Run Screen", type="primary", use_container_width=True, disabled=not tickers)

    st.divider()
    st.header("Pick Tracking")
    track_file = st.text_input("Tracking log file", "picks_log.csv")

    st.divider()
    st.header("Paper Account (no signup)")
    paper_file = st.text_input("Paper account file", "paper_account.json")
    paper_starting_cash = st.number_input(
        "Starting cash (new accounts only)", min_value=100.0,
        value=DEFAULT_STARTING_CASH, step=1000.0,
    )
    if st.button("♻️ Reset paper account", use_container_width=True):
        paper_reset_account(paper_file, paper_starting_cash)
        st.success(f"Reset {paper_file} to ${paper_starting_cash:,.2f} cash.")

    st.divider()
    st.header("Broker (Alpaca Paper) — optional")
    alpaca_key = st.text_input(
        "API Key ID", value=os.environ.get("ALPACA_API_KEY_ID", ""), type="password"
    )
    alpaca_secret = st.text_input(
        "Secret Key", value=os.environ.get("ALPACA_SECRET_KEY", ""), type="password"
    )
    st.caption(
        "Only needed for the Alpaca tab. Free keys at alpaca.markets → Paper Trading. "
        "Simulated money only — never live. Keys are used for this session only, never "
        "written to disk. The Paper Trade tab above needs no signup or keys at all."
    )

    st.divider()
    st.header("Questrade — optional")
    qt_refresh_token = st.text_input(
        "Refresh token", value="", type="password",
        help="Generate from my.questrade.com (or practicelogin.questrade.com for a "
             "practice account) → API Centre → Personal Apps → Generate new token.",
    )
    qt_token_file = st.text_input("Token cache file", "questrade_token.json")
    st.caption(
        "Only needed for the Questrade tab. Each token exchange rotates the refresh "
        "token, so the new one is cached in the file above — reuse it across runs "
        "instead of re-pasting. Whether this hits a practice or live account depends "
        "entirely on which token you generated; this app never chooses that for you."
    )

tab_screen, tab_track, tab_paper, tab_alpaca, tab_questrade, tab_backtest, tab_cutrisk = st.tabs(
    ["📊 Screener", "📈 Track Picks", "🧪 Paper Trade", "💼 Alpaca (Paper API)", "🇨🇦 Questrade",
     "🔬 Backtest", "🎲 Cut Risk (ML)"]
)

with tab_screen:
    if run:
        progress = st.progress(0.0, text="Starting...")

        def _update(i, total, tkr):
            progress.progress(i / total, text=f"Fetching {tkr} ({i}/{total})")

        results = screen(tickers, progress=False, on_progress=_update)
        progress.empty()

        filtered = filter_results(
            results,
            min_yield=min_yield or None,
            max_payout=None if max_payout == 200 else max_payout,
            min_score=min_score or None,
            min_grade=None if min_grade == "Any" else min_grade,
            max_pe=None if max_pe == 200 else max_pe,
            max_de=None if max_de == 500 else max_de,
        )
        st.session_state["results"] = results
        st.session_state["shown"] = filtered[:top_n]

        dropped = len(results) - len(filtered)
        if dropped:
            st.info(f"Filtered out {dropped} of {len(results)} stocks that didn't meet the thresholds.")

    results = st.session_state["results"]
    shown = st.session_state["shown"]

    if not results:
        st.info("Configure your universe and filters in the sidebar, then click **Run Screen**.")
    else:
        failed = [r for r in results if r.error]
        ok_shown = [r for r in shown if not r.error]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tickers screened", len(results))
        c2.metric("Shown (after filters)", len(ok_shown))
        avg_score = sum(r.composite for r in ok_shown) / len(ok_shown) if ok_shown else 0
        c3.metric("Avg score (shown)", f"{avg_score:.1f}")
        ab_count = sum(1 for r in ok_shown if r.grade in ("A", "B"))
        c4.metric("A/B grades (shown)", ab_count)

        if ok_shown:
            table = pd.DataFrame([{
                "Rank": i,
                "Ticker": r.ticker,
                "Name": r.name or "",
                "Grade": GRADE_EMOJI.get(r.grade, r.grade),
                "Score": r.composite,
                "Yield %": r.raw.get("dividend_yield_pct"),
                "Payout %": r.raw.get("payout_ratio_pct"),
                "5y Div CAGR %": r.raw.get("dividend_cagr_5y_pct"),
                "No-cut streak (yr)": r.raw.get("no_cut_streak_years"),
                "P/E": r.raw.get("trailing_pe"),
                "D/E": r.raw.get("debt_to_equity"),
                "Sector": r.raw.get("sector") or "",
            } for i, r in enumerate(ok_shown, 1)])

            st.dataframe(
                table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", min_value=0, max_value=100, format="%.1f"
                    ),
                },
            )

            st.bar_chart(table.set_index("Ticker")["Score"], height=280)

            st.subheader("Detail")
            for r in ok_shown:
                with st.expander(f"{GRADE_EMOJI.get(r.grade, r.grade)}  {r.ticker} — {r.name}  ({r.composite}/100)"):
                    left, right = st.columns([2, 1])
                    with left:
                        st.bar_chart(pd.Series(r.sub_scores, name="sub-score"))
                        for note in r.notes:
                            st.write(f"- {note}")
                    with right:
                        st.write(f"**Price:** {r.raw.get('price')}")
                        st.write(f"**Yield:** {r.raw.get('dividend_yield_pct')}%")
                        st.write(f"**Payout:** {r.raw.get('payout_ratio_pct')}%")
                        st.write(f"**5y Div CAGR:** {r.raw.get('dividend_cagr_5y_pct')}%")
                        st.write(f"**No-cut streak:** {r.raw.get('no_cut_streak_years')} yr")
                        st.write(f"**P/E:** {r.raw.get('trailing_pe')}")
                        st.write(f"**D/E:** {r.raw.get('debt_to_equity')}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "⬇ Download full results (CSV)",
                    to_dataframe(results).to_csv(index=False),
                    file_name="dividend_screen_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_b:
                if st.button("📌 Log shown picks to tracking file", use_container_width=True):
                    n = record_picks(ok_shown, track_file, top=None)
                    st.success(f"Logged {n} pick(s) to {track_file}.")
        else:
            st.warning("No stocks passed the current filters.")

        if failed:
            with st.expander(f"⚠ {len(failed)} ticker(s) failed to fetch"):
                for r in failed:
                    st.write(f"**{r.ticker}**: {r.error}")

with tab_track:
    st.subheader("Tracked pick performance")
    st.caption(
        "Paper-tracking only — no broker or real money involved. Log picks from the "
        "Screener tab, then come back here to see how they've performed since."
    )
    if st.button("🔄 Refresh performance", type="primary"):
        try:
            log = review_picks(track_file)
        except FileNotFoundError:
            st.warning(f"No tracking log found at `{track_file}` yet. Log some picks from the Screener tab first.")
            log = None

        if log is not None:
            if log.empty:
                st.info("The tracking log is empty.")
            else:
                log = log.sort_values("return_pct", ascending=False, na_position="last")
                display = log.copy()
                display["pick_date"] = display["pick_date"].dt.date.astype(str)

                st.dataframe(
                    display,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "return_pct": st.column_config.NumberColumn("Return %", format="%.2f%%"),
                        "price_at_pick": st.column_config.NumberColumn("Pick $", format="$%.2f"),
                        "current_price": st.column_config.NumberColumn("Now $", format="$%.2f"),
                    },
                )

                st.bar_chart(log.set_index("ticker")["return_pct"], height=280)

                valid = log["return_pct"].dropna()
                if not valid.empty:
                    st.metric("Average return across tracked picks", f"{valid.mean():+.2f}%")

with tab_paper:
    st.subheader("Paper Trade — Local Simulator")
    st.caption(
        "No signup, no API keys, no real broker. A virtual cash balance tracked in a "
        "local file, priced with live yfinance data — same source the screener uses."
    )

    account = paper_load_account(paper_file, paper_starting_cash)
    held_tickers = list(account["holdings"].keys())

    price_lookup = {}
    if held_tickers:
        with st.spinner(f"Fetching current prices for {len(held_tickers)} holding(s)..."):
            price_lookup = {t: fetch_stock_data(t).price for t in held_tickers}

    summary = paper_get_summary(account, price_lookup)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cash", f"${summary['cash']:,.2f}")
    c2.metric("Holdings value", f"${summary['holdings_value']:,.2f}")
    c3.metric("Portfolio value", f"${summary['portfolio_value']:,.2f}")
    c4.metric(
        "Total P/L",
        f"${summary['total_pl']:,.2f}",
        delta=f"{summary['total_pl_pct']:+.2f}%",
    )

    st.subheader("Positions")
    positions = paper_get_positions(account, price_lookup)
    if positions:
        pos_df = pd.DataFrame([{
            "Ticker": p["ticker"],
            "Qty": p["qty"],
            "Avg Entry": p["avg_entry_price"],
            "Current": p["current_price"],
            "Market Value": p["market_value"],
            "Unrealized P/L": p["unrealized_pl"],
            "Unrealized P/L %": p["unrealized_plpc"],
        } for p in positions])
        st.dataframe(
            pos_df, hide_index=True, use_container_width=True,
            column_config={
                "Unrealized P/L %": st.column_config.NumberColumn(format="%.2f%%"),
                "Avg Entry": st.column_config.NumberColumn(format="$%.2f"),
                "Current": st.column_config.NumberColumn(format="$%.2f"),
                "Market Value": st.column_config.NumberColumn(format="$%.2f"),
                "Unrealized P/L": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
        sell_col1, sell_col2 = st.columns([2, 1])
        with sell_col1:
            sell_ticker = st.selectbox("Sell a position", ["—"] + [p["ticker"] for p in positions])
        with sell_col2:
            if sell_ticker != "—" and st.button(f"🔴 Sell all {sell_ticker}"):
                err = paper_sell(account, sell_ticker, price=price_lookup.get(sell_ticker))
                if err:
                    st.error(err)
                else:
                    paper_save_account(account, paper_file)
                    st.success(f"Sold entire {sell_ticker} position.")
                    st.rerun()
    else:
        st.info("No open positions.")

    st.subheader("Buy From Screener Picks")
    paper_shown = st.session_state.get("shown")
    paper_picks = [r for r in paper_shown if not r.error] if paper_shown else []
    if not paper_picks:
        st.info("Run a screen in the **Screener** tab first to get picks to trade.")
    else:
        paper_dollar_amount = st.number_input(
            "Dollar amount per pick", min_value=1.0, value=1000.0, step=100.0, key="paper_dollar_amount"
        )
        paper_picks_to_buy = st.multiselect(
            "Picks to buy", [r.ticker for r in paper_picks],
            default=[r.ticker for r in paper_picks], key="paper_picks_to_buy",
        )
        buy_clicked = st.button(
            f"🛒 Buy {len(paper_picks_to_buy)} pick(s) — ${paper_dollar_amount:,.0f} each",
            type="primary", disabled=not paper_picks_to_buy, key="paper_buy_button",
        )
        if buy_clicked:
            price_by_ticker = {r.ticker: r.raw.get("price") for r in paper_picks}
            for ticker in paper_picks_to_buy:
                err = paper_buy(account, ticker, paper_dollar_amount, price_by_ticker.get(ticker))
                if err:
                    st.error(f"{ticker}: {err}")
                else:
                    st.success(f"{ticker}: bought ${paper_dollar_amount:,.0f} at ${price_by_ticker.get(ticker):.2f}.")
            paper_save_account(account, paper_file)
            st.rerun()

    st.subheader("Transaction History")
    if account["transactions"]:
        st.dataframe(
            pd.DataFrame(list(reversed(account["transactions"]))),
            hide_index=True, use_container_width=True,
        )
    else:
        st.info("No transactions yet.")

with tab_alpaca:
    st.subheader("Paper Trading (Alpaca)")
    st.caption(
        "Simulated money, real market mechanics via Alpaca's paper-trading API. "
        "This never places live/real-money trades."
    )

    if not alpaca_key or not alpaca_secret:
        st.info("Enter your Alpaca **paper trading** API keys in the sidebar to connect.")
    else:
        client = get_client(alpaca_key, alpaca_secret)
        account = fetch_account(client)

        if account.error:
            st.error(f"Couldn't connect to Alpaca: {account.error}")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cash", f"${account.cash:,.2f}")
            c2.metric("Buying power", f"${account.buying_power:,.2f}")
            c3.metric("Portfolio value", f"${account.portfolio_value:,.2f}")
            c4.metric("Equity", f"${account.equity:,.2f}")

            st.subheader("Positions")
            positions, pos_err = fetch_positions(client)
            if pos_err:
                st.error(pos_err)
            elif positions:
                pos_df = pd.DataFrame([{
                    "Ticker": p.ticker,
                    "Qty": p.qty,
                    "Avg Entry": p.avg_entry_price,
                    "Current": p.current_price,
                    "Market Value": p.market_value,
                    "Unrealized P/L": p.unrealized_pl,
                    "Unrealized P/L %": p.unrealized_plpc,
                } for p in positions])
                st.dataframe(
                    pos_df, hide_index=True, use_container_width=True,
                    column_config={
                        "Unrealized P/L %": st.column_config.NumberColumn(format="%.2f%%"),
                        "Avg Entry": st.column_config.NumberColumn(format="$%.2f"),
                        "Current": st.column_config.NumberColumn(format="$%.2f"),
                        "Market Value": st.column_config.NumberColumn(format="$%.2f"),
                        "Unrealized P/L": st.column_config.NumberColumn(format="$%.2f"),
                    },
                )
                close_ticker = st.selectbox("Close a position", ["—"] + [p.ticker for p in positions])
                if close_ticker != "—" and st.button(f"🔴 Close {close_ticker} position"):
                    result = close_position(client, close_ticker)
                    if result.error:
                        st.error(result.error)
                    else:
                        st.success(f"Submitted close order for {close_ticker} ({result.status}).")
            else:
                st.info("No open positions.")

            st.subheader("Buy From Screener Picks")
            trade_shown = st.session_state.get("shown")
            trade_picks = [r for r in trade_shown if not r.error] if trade_shown else []
            if not trade_picks:
                st.info("Run a screen in the **Screener** tab first to get picks to trade.")
            else:
                dollar_amount = st.number_input(
                    "Dollar amount per pick", min_value=1.0, value=100.0, step=25.0
                )
                picks_to_buy = st.multiselect(
                    "Picks to buy", [r.ticker for r in trade_picks],
                    default=[r.ticker for r in trade_picks],
                )
                buy_clicked = st.button(
                    f"🛒 Buy {len(picks_to_buy)} pick(s) — paper, ${dollar_amount:,.0f} each",
                    type="primary", disabled=not picks_to_buy,
                )
                if buy_clicked:
                    for ticker in picks_to_buy:
                        result = submit_buy(client, ticker, dollar_amount)
                        if result.error:
                            st.error(f"{ticker}: {result.error}")
                        else:
                            st.success(f"{ticker}: order {result.status}")

            st.subheader("Recent Orders")
            orders, order_err = fetch_recent_orders(client)
            if order_err:
                st.error(order_err)
            elif orders:
                st.dataframe(pd.DataFrame(orders), hide_index=True, use_container_width=True)
            else:
                st.info("No orders yet.")

with tab_questrade:
    st.subheader("Questrade")
    st.caption(
        "Connects to whichever Questrade account issued your refresh token — practice "
        "or live, this app can't tell the difference and doesn't choose for you. "
        "Double-check you generated a **practice** token if that's what you want."
    )

    qt_session, qt_err = qt_authenticate(qt_refresh_token or None, qt_token_file)
    if qt_err:
        st.info(qt_err) if "No refresh token" in qt_err else st.error(qt_err)
    else:
        accounts, acct_err = qt_get_accounts(qt_session)
        if acct_err:
            st.error(f"Couldn't load accounts: {acct_err}")
        elif not accounts:
            st.warning("Connected, but no accounts were returned for this token.")
        else:
            account_labels = {f"{a.get('number')} ({a.get('type')})": a.get("number") for a in accounts}
            selected_label = st.selectbox("Account", list(account_labels.keys()))
            qt_account_id = account_labels[selected_label]

            balance, bal_err = qt_get_balances(qt_session, qt_account_id)
            if bal_err:
                st.error(bal_err)
            elif balance:
                cur = balance.get("currency", "")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Cash", f"{balance.get('cash', 0):,.2f} {cur}")
                c2.metric("Market value", f"{balance.get('marketValue', 0):,.2f} {cur}")
                c3.metric("Total equity", f"{balance.get('totalEquity', 0):,.2f} {cur}")
                c4.metric("Buying power", f"{balance.get('buyingPower', 0):,.2f} {cur}")

            st.subheader("Positions")
            qt_positions, pos_err = qt_get_positions(qt_session, qt_account_id)
            if pos_err:
                st.error(pos_err)
            elif qt_positions:
                qt_pos_df = pd.DataFrame([{
                    "Symbol": p.get("symbol"),
                    "Qty": p.get("openQuantity"),
                    "Avg Entry": p.get("averageEntryPrice"),
                    "Current": p.get("currentPrice"),
                    "Market Value": p.get("currentMarketValue"),
                    "Open P/L": p.get("openPnl"),
                } for p in qt_positions])
                st.dataframe(
                    qt_pos_df, hide_index=True, use_container_width=True,
                    column_config={
                        "Avg Entry": st.column_config.NumberColumn(format="%.2f"),
                        "Current": st.column_config.NumberColumn(format="%.2f"),
                        "Market Value": st.column_config.NumberColumn(format="%.2f"),
                        "Open P/L": st.column_config.NumberColumn(format="%.2f"),
                    },
                )
                qt_close_symbol = st.selectbox("Close a position", ["—"] + [p.get("symbol") for p in qt_positions])
                if qt_close_symbol != "—" and st.button(f"🔴 Close {qt_close_symbol} position"):
                    pos = next(p for p in qt_positions if p.get("symbol") == qt_close_symbol)
                    order, order_err = qt_place_market_order(
                        qt_session, qt_account_id, symbol_id=pos["symbolId"],
                        quantity=int(abs(pos["openQuantity"])), action="Sell",
                    )
                    if order_err:
                        st.error(order_err)
                    else:
                        st.success(f"Submitted sell order for {qt_close_symbol}.")
            else:
                st.info("No open positions.")

            st.subheader("Buy From Screener Picks")
            qt_shown = st.session_state.get("shown")
            qt_picks = [r for r in qt_shown if not r.error] if qt_shown else []
            if not qt_picks:
                st.info("Run a screen in the **Screener** tab first to get picks to trade.")
            else:
                qt_shares = st.number_input("Shares per pick", min_value=1, value=1, step=1)
                qt_picks_to_buy = st.multiselect(
                    "Picks to buy", [r.ticker for r in qt_picks],
                    default=[r.ticker for r in qt_picks], key="qt_picks_to_buy",
                )
                qt_buy_clicked = st.button(
                    f"🛒 Buy {len(qt_picks_to_buy)} pick(s) — {qt_shares} share(s) each",
                    type="primary", disabled=not qt_picks_to_buy,
                )
                if qt_buy_clicked:
                    for ticker in qt_picks_to_buy:
                        matches, search_err = qt_search_symbol(qt_session, ticker)
                        exact = next(
                            (m for m in matches if m.get("symbol", "").upper() == ticker.upper()),
                            matches[0] if matches else None,
                        )
                        if search_err or not exact:
                            st.error(f"{ticker}: symbol lookup failed ({search_err or 'no match'}).")
                            continue
                        order, order_err = qt_place_market_order(
                            qt_session, qt_account_id, symbol_id=exact["symbolId"],
                            quantity=qt_shares, action="Buy",
                        )
                        if order_err:
                            st.error(f"{ticker}: {order_err}")
                        else:
                            st.success(f"{ticker}: buy order submitted.")

with tab_backtest:
    st.subheader("Strategy Backtest")
    st.caption(
        "Simulates each strategy month by month using real historical prices and "
        "dividend payments. **Historical performance does not guarantee future "
        "results** — this is a research tool, not a forecast."
    )

    st.session_state.setdefault("backtest_universes", None)
    st.session_state.setdefault("backtest_results", None)

    bt_col1, bt_col2, bt_col3 = st.columns(3)
    with bt_col1:
        bt_initial = st.number_input("Initial capital", min_value=100.0, value=25_000.0, step=1000.0)
    with bt_col2:
        bt_monthly = st.number_input("Monthly contribution", min_value=0.0, value=500.0, step=50.0)
    with bt_col3:
        bt_top_n = st.slider("Tickers per strategy", 2, 15, 8)

    bt_horizons = st.multiselect("Horizons (years)", [5, 10, 15, 20], default=[10, 15, 20])
    bt_drip = st.checkbox("Reinvest dividends (DRIP)", value=True)

    if st.button("🔬 Build strategy universes & run backtest", type="primary", disabled=not bt_horizons):
        bt_progress = st.progress(0.0, text="Classifying candidate universe...")

        def _bt_update(i, total, tkr):
            bt_progress.progress(i / total, text=f"Classifying {tkr} ({i}/{total})")

        universes, universe_errors = build_strategy_universes(
            BACKTEST_UNIVERSE, top_n=bt_top_n, on_progress=_bt_update
        )
        bt_progress.empty()
        st.session_state["backtest_universes"] = universes

        results_by_horizon = {}
        for years in bt_horizons:
            with st.spinner(f"Simulating {years}-year backtests..."):
                results_by_horizon[years] = compare_strategies(
                    universes, years=years, initial_capital=bt_initial,
                    monthly_contribution=bt_monthly, drip=bt_drip,
                )
        st.session_state["backtest_results"] = results_by_horizon

        if universe_errors:
            with st.expander(f"⚠ {len(universe_errors)} ticker(s) skipped while building universes"):
                for e in universe_errors:
                    st.write(e)

    if st.session_state["backtest_universes"]:
        with st.expander("Strategy universes used"):
            for name, tickers in st.session_state["backtest_universes"].items():
                st.write(f"**{name}**: {', '.join(tickers) if tickers else '_no qualifying tickers_'}")

    if st.session_state["backtest_results"]:
        for years, results in st.session_state["backtest_results"].items():
            st.subheader(f"{years}-Year Horizon")
            rows = []
            for name, r in results.items():
                if r.error:
                    rows.append({"Strategy": name, "Error": r.error})
                else:
                    rows.append({
                        "Strategy": name,
                        "Final Value": r.final_value,
                        "Contributions": r.total_contributions,
                        "Dividends": r.total_dividends,
                        "Total Return %": r.total_return_pct,
                        "CAGR (IRR) %": r.cagr_pct,
                        "Max Drawdown %": r.max_drawdown_pct,
                        "Volatility %": r.volatility_pct,
                        "Sharpe": r.sharpe,
                        "Yield on Cost %": r.yield_on_cost_pct,
                    })
            bt_df = pd.DataFrame(rows)
            st.dataframe(bt_df, hide_index=True, use_container_width=True)

            valid = {n: r for n, r in results.items() if not r.error}
            if valid:
                winner = max(valid.items(), key=lambda kv: kv[1].cagr_pct)
                st.success(f"Highest CAGR over {years}yr: **{winner[0]}** ({winner[1].cagr_pct:+.2f}%/yr)")

                chart_data = {}
                for name, r in valid.items():
                    if r.equity_curve:
                        s = pd.Series([v for _, v in r.equity_curve], index=[d for d, _ in r.equity_curve])
                        chart_data[name] = s
                if chart_data:
                    st.line_chart(pd.DataFrame(chart_data), height=300)
    else:
        st.info("Configure the backtest above and click **Build strategy universes & run backtest**.")

with tab_cutrisk:
    st.subheader("Dividend Cut-Risk Model")
    st.caption(
        "A small logistic-regression classifier trained on real historical dividend "
        "cuts/non-cuts. It **supplements** the rule-based sustainability checks in "
        "the Screener tab — it does not replace them. Trained on price- and "
        "dividend-history-derived features only (no historical fundamentals were "
        "freely available to train on), so it can't see balance-sheet deterioration "
        "directly, only its usual side effects. Always read the metrics below before "
        "trusting a prediction."
    )

    st.session_state.setdefault("cut_risk_model", None)
    st.session_state.setdefault("cut_risk_metrics", None)

    cr_metrics_path = CUT_RISK_MODEL_PATH.replace(".joblib", "_metrics.json")
    if st.session_state["cut_risk_model"] is None:
        loaded = cr_load_model(CUT_RISK_MODEL_PATH)
        if loaded is not None:
            st.session_state["cut_risk_model"] = loaded
            if os.path.exists(cr_metrics_path):
                with open(cr_metrics_path) as f:
                    st.session_state["cut_risk_metrics"] = json.load(f)

    if st.button("🧠 Train / retrain model", type="primary"):
        cr_progress = st.progress(0.0, text="Fetching dividend history...")

        def _cr_update(i, total, tkr):
            cr_progress.progress(i / total, text=f"{tkr} ({i}/{total})")

        dataset, train_errors = cr_build_training_dataset(CUT_RISK_TRAINING_UNIVERSE, on_progress=_cr_update)
        cr_progress.empty()

        model, metrics = cr_train_model(dataset)
        if model is None:
            st.error(metrics.get("error", "Training failed."))
        else:
            cr_save_model(model, CUT_RISK_MODEL_PATH)
            with open(cr_metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            st.session_state["cut_risk_model"] = model
            st.session_state["cut_risk_metrics"] = metrics
            st.success(
                f"Trained on {metrics['n_samples']} samples "
                f"({metrics['n_positive']} cuts, {metrics['n_negative']} non-cuts)."
            )
        if train_errors:
            with st.expander(f"⚠ {len(train_errors)} ticker(s) skipped while building training data"):
                for e in train_errors:
                    st.write(e)

    cr_metrics = st.session_state["cut_risk_metrics"]
    if cr_metrics and "roc_auc" in cr_metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CV ROC-AUC", cr_metrics["roc_auc"])
        m2.metric("Precision", cr_metrics["precision"])
        m3.metric("Recall", cr_metrics["recall"])
        m4.metric("Training samples", f"{cr_metrics['n_samples']} ({cr_metrics['n_positive']} cuts)")
        if cr_metrics["precision"] < 0.5:
            st.warning(
                f"Precision is {cr_metrics['precision']:.0%} — most tickers this model flags as "
                "high-risk will turn out fine. Treat a high score as 'worth a second look', not a verdict."
            )

    if st.session_state["cut_risk_model"] is None:
        st.info("No model trained yet — click **Train / retrain model** above (takes a minute or two).")
    else:
        cr_default_tickers = ", ".join(r.ticker for r in (st.session_state.get("shown") or []) if not r.error)
        cr_raw = st.text_area("Tickers to check (space or comma separated)", cr_default_tickers or "JNJ KO O")
        cr_tickers = [t.strip().upper() for t in cr_raw.replace(",", " ").split() if t.strip()]

        if st.button("🎲 Check cut risk", disabled=not cr_tickers):
            cr_rows = []
            for t in cr_tickers:
                proba, info, err = cr_predict_cut_risk(st.session_state["cut_risk_model"], t)
                if err:
                    cr_rows.append({"Ticker": t, "Cut Risk %": None, "Note": err})
                else:
                    band = (
                        "Low" if proba < 0.15 else
                        "Moderate" if proba < 0.35 else
                        "Elevated" if proba < 0.60 else
                        "High"
                    )
                    cr_rows.append({
                        "Ticker": t, "Cut Risk %": round(proba * 100, 1), "Band": band,
                        "As of FY": info["as_of_year"], "Note": "",
                    })
            st.dataframe(pd.DataFrame(cr_rows), hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "This is a rule-based educational screening tool, not financial advice. "
    f"Scoring weights: {', '.join(f'{k}={v}' for k, v in WEIGHTS.items())}."
)
