# ============================================================
# STOCK TRADING DASHBOARD PRO
# Robust Fundamentals + Manual Ratio Fallbacks
# Streamlit App
# ============================================================

import math
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from datetime import datetime, timedelta


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Stock Trading Dashboard Pro",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Trading Dashboard Pro")
st.caption(
    "Hybrid FMP + Finnhub + Yahoo Backup engine with technicals, "
    "valuation, entry zones, and options ideas."
)


# ============================================================
# API KEYS
# ============================================================

def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


FMP_API_KEY = get_secret("FMP_API_KEY", "")
FINNHUB_API_KEY = get_secret("FINNHUB_API_KEY", "")


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value):
    try:
        if value is None:
            return None

        if isinstance(value, str):
            value = value.replace(",", "").replace("$", "").strip()
            if value in ["", "None", "nan", "NaN", "N/A"]:
                return None

        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    except Exception:
        return None


def safe_int(value):
    try:
        value = safe_float(value)
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def first_valid(*values):
    for value in values:
        value = safe_float(value)
        if value is not None:
            return value
    return None


def pick_number(dictionary, keys):
    if not isinstance(dictionary, dict):
        return None

    for key in keys:
        value = safe_float(dictionary.get(key))
        if value is not None:
            return value

    return None


def fmt_number(value, decimals=2):
    value = safe_float(value)

    if value is None:
        return "N/A"

    return f"{value:,.{decimals}f}"


def fmt_large(value):
    value = safe_float(value)

    if value is None:
        return "N/A"

    abs_value = abs(value)

    if abs_value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:,.2f}T"
    elif abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    elif abs_value >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    else:
        return f"${value:,.0f}"


def metric_value(value, decimals=2):
    value = safe_float(value)

    if value is None:
        return "N/A"

    return f"{value:,.{decimals}f}"


def ratio_status(value):
    value = safe_float(value)
    if value is None:
        return "Missing"
    return "OK"


def fetch_json(url, timeout=12):
    try:
        response = requests.get(url, timeout=timeout)

        if response.status_code != 200:
            return None

        data = response.json()
        return data

    except Exception:
        return None


# ============================================================
# FMP DATA
# ============================================================

@st.cache_data(ttl=3600)
def get_fmp_data(ticker, api_key):
    ticker = ticker.upper().strip()

    empty = {
        "profile": {},
        "quote": {},
        "ratios_ttm": {},
        "key_metrics_ttm": {},
        "income": {},
        "balance": {},
        "cashflow": {},
        "analyst_estimates": [],
        "earnings_calendar": [],
    }

    if not api_key:
        return empty

    base = "https://financialmodelingprep.com/api/v3"

    urls = {
        "profile": f"{base}/profile/{ticker}?apikey={api_key}",
        "quote": f"{base}/quote/{ticker}?apikey={api_key}",
        "ratios_ttm": f"{base}/ratios-ttm/{ticker}?apikey={api_key}",
        "key_metrics_ttm": f"{base}/key-metrics-ttm/{ticker}?apikey={api_key}",
        "income": f"{base}/income-statement/{ticker}?period=annual&limit=1&apikey={api_key}",
        "balance": f"{base}/balance-sheet-statement/{ticker}?period=annual&limit=1&apikey={api_key}",
        "cashflow": f"{base}/cash-flow-statement/{ticker}?period=annual&limit=1&apikey={api_key}",
        "analyst_estimates": f"{base}/analyst-estimates/{ticker}?period=annual&limit=3&apikey={api_key}",
        "earnings_calendar": f"{base}/earning_calendar?symbol={ticker}&limit=10&apikey={api_key}",
    }

    result = empty.copy()

    for key, url in urls.items():
        data = fetch_json(url)

        if isinstance(data, list):
            if key in ["analyst_estimates", "earnings_calendar"]:
                result[key] = data
            elif len(data) > 0:
                result[key] = data[0]
        elif isinstance(data, dict):
            result[key] = data

    return result


# ============================================================
# YAHOO DATA
# ============================================================

@st.cache_data(ttl=1800)
def get_yahoo_data(ticker):
    ticker = ticker.upper().strip()

    result = {
        "info": {},
        "history": pd.DataFrame(),
        "calendar": None,
        "options_dates": [],
    }

    try:
        tk = yf.Ticker(ticker)

        try:
            result["info"] = tk.get_info()
        except Exception:
            result["info"] = {}

        try:
            result["history"] = tk.history(period="1y", auto_adjust=False)
        except Exception:
            result["history"] = pd.DataFrame()

        try:
            result["calendar"] = tk.calendar
        except Exception:
            result["calendar"] = None

        try:
            result["options_dates"] = list(tk.options)
        except Exception:
            result["options_dates"] = []

    except Exception:
        pass

    return result


# ============================================================
# FINNHUB DATA
# ============================================================

@st.cache_data(ttl=3600)
def get_finnhub_earnings(ticker, api_key):
    ticker = ticker.upper().strip()

    if not api_key:
        return []

    today = datetime.today().date()
    future = today + timedelta(days=365)

    url = (
        "https://finnhub.io/api/v1/calendar/earnings"
        f"?symbol={ticker}"
        f"&from={today}"
        f"&to={future}"
        f"&token={api_key}"
    )

    data = fetch_json(url)

    if isinstance(data, dict):
        return data.get("earningsCalendar", [])

    return []


# ============================================================
# EARNINGS DATE FALLBACK
# ============================================================

def extract_yahoo_earnings_date(calendar):
    try:
        if calendar is None:
            return None

        if isinstance(calendar, pd.DataFrame):
            if calendar.empty:
                return None

            for col in calendar.columns:
                values = calendar[col].dropna()
                if len(values) > 0:
                    return str(values.iloc[0])[:10]

        if isinstance(calendar, dict):
            for key in ["Earnings Date", "EarningsDate", "earningsDate"]:
                value = calendar.get(key)
                if value is None:
                    continue

                if isinstance(value, list) and len(value) > 0:
                    return str(value[0])[:10]

                return str(value)[:10]

    except Exception:
        return None

    return None


def extract_fmp_earnings_date(earnings_calendar):
    try:
        today = datetime.today().date()

        dates = []

        for item in earnings_calendar:
            date_text = item.get("date")
            if not date_text:
                continue

            date_obj = pd.to_datetime(date_text).date()

            if date_obj >= today:
                dates.append(date_obj)

        if dates:
            return str(min(dates))

    except Exception:
        return None

    return None


def extract_finnhub_earnings_date(earnings_calendar):
    try:
        today = datetime.today().date()

        dates = []

        for item in earnings_calendar:
            date_text = item.get("date")
            if not date_text:
                continue

            date_obj = pd.to_datetime(date_text).date()

            if date_obj >= today:
                dates.append(date_obj)

        if dates:
            return str(min(dates))

    except Exception:
        return None

    return None


# ============================================================
# MANUAL CALCULATIONS
# ============================================================

def calculate_enterprise_value(market_cap, total_debt, cash):
    market_cap = safe_float(market_cap)

    if market_cap is None:
        return None

    total_debt = safe_float(total_debt) or 0
    cash = safe_float(cash) or 0

    return market_cap + total_debt - cash


def calculate_ebitda(operating_income, depreciation_and_amortization):
    operating_income = safe_float(operating_income)
    depreciation_and_amortization = safe_float(depreciation_and_amortization)

    if operating_income is None or depreciation_and_amortization is None:
        return None

    return operating_income + depreciation_and_amortization


def calculate_trailing_pe(market_cap, net_income):
    market_cap = safe_float(market_cap)
    net_income = safe_float(net_income)

    if market_cap is None or net_income is None:
        return None

    if net_income <= 0:
        return None

    return market_cap / net_income


def calculate_forward_pe(last_price, forward_eps):
    last_price = safe_float(last_price)
    forward_eps = safe_float(forward_eps)

    if last_price is None or forward_eps is None:
        return None

    if forward_eps <= 0:
        return None

    return last_price / forward_eps


def calculate_ev_ebitda(enterprise_value, ebitda):
    enterprise_value = safe_float(enterprise_value)
    ebitda = safe_float(ebitda)

    if enterprise_value is None or ebitda is None:
        return None

    if ebitda <= 0:
        return None

    return enterprise_value / ebitda


# ============================================================
# MAIN FUNDAMENTAL ENGINE
# ============================================================

@st.cache_data(ttl=3600)
def get_fundamentals(ticker, fmp_api_key, finnhub_api_key):
    ticker = ticker.upper().strip()

    notes = []
    debug_rows = []

    fmp = get_fmp_data(ticker, fmp_api_key)
    yahoo = get_yahoo_data(ticker)
    finnhub_earnings = get_finnhub_earnings(ticker, finnhub_api_key)

    yinfo = yahoo.get("info", {}) or {}
    hist = yahoo.get("history", pd.DataFrame())

    profile = fmp.get("profile", {}) or {}
    quote = fmp.get("quote", {}) or {}
    ratios = fmp.get("ratios_ttm", {}) or {}
    key_metrics = fmp.get("key_metrics_ttm", {}) or {}
    income = fmp.get("income", {}) or {}
    balance = fmp.get("balance", {}) or {}
    cashflow = fmp.get("cashflow", {}) or {}

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    last_close = first_valid(
        quote.get("previousClose"),
        quote.get("price"),
        yinfo.get("regularMarketPreviousClose"),
        yinfo.get("previousClose"),
        yinfo.get("regularMarketPrice"),
        yinfo.get("currentPrice"),
    )

    if last_close is None and isinstance(hist, pd.DataFrame) and not hist.empty:
        last_close = safe_float(hist["Close"].dropna().iloc[-1])

    source_last_close = "FMP/Yahoo/History" if last_close is not None else "Missing"

    # --------------------------------------------------------
    # MARKET CAP
    # --------------------------------------------------------

    market_cap = first_valid(
        profile.get("mktCap"),
        quote.get("marketCap"),
        key_metrics.get("marketCapTTM"),
        yinfo.get("marketCap"),
    )

    shares_outstanding = first_valid(
        yinfo.get("sharesOutstanding"),
        profile.get("sharesOutstanding"),
    )

    if market_cap is None and last_close is not None and shares_outstanding is not None:
        market_cap = last_close * shares_outstanding
        notes.append("Market Cap calculated manually from Last Price × Shares Outstanding.")

    # --------------------------------------------------------
    # DEBT AND CASH
    # --------------------------------------------------------

    total_debt = first_valid(
        balance.get("totalDebt"),
        yinfo.get("totalDebt"),
    )

    if total_debt is None:
        short_debt = safe_float(balance.get("shortTermDebt")) or 0
        long_debt = safe_float(balance.get("longTermDebt")) or 0

        if short_debt != 0 or long_debt != 0:
            total_debt = short_debt + long_debt
            notes.append("Total Debt calculated manually from Short-Term Debt + Long-Term Debt.")

    cash = first_valid(
        balance.get("cashAndCashEquivalents"),
        balance.get("cashAndShortTermInvestments"),
        yinfo.get("totalCash"),
    )

    # --------------------------------------------------------
    # ENTERPRISE VALUE
    # --------------------------------------------------------

    enterprise_value = first_valid(
        key_metrics.get("enterpriseValueTTM"),
        key_metrics.get("enterpriseValue"),
        yinfo.get("enterpriseValue"),
    )

    if enterprise_value is None:
        enterprise_value = calculate_enterprise_value(
            market_cap=market_cap,
            total_debt=total_debt,
            cash=cash,
        )

        if enterprise_value is not None:
            notes.append("Enterprise Value calculated manually from Market Cap + Total Debt - Cash.")

    # --------------------------------------------------------
    # EBITDA
    # --------------------------------------------------------

    ebitda = first_valid(
        income.get("ebitda"),
        key_metrics.get("ebitdaTTM"),
        yinfo.get("ebitda"),
    )

    if ebitda is None:
        operating_income = first_valid(
            income.get("operatingIncome"),
            income.get("operatingIncomeLoss"),
        )

        depreciation_and_amortization = first_valid(
            cashflow.get("depreciationAndAmortization"),
            cashflow.get("depreciationAndAmortizationExpense"),
        )

        ebitda = calculate_ebitda(
            operating_income=operating_income,
            depreciation_and_amortization=depreciation_and_amortization,
        )

        if ebitda is not None:
            notes.append("EBITDA calculated manually from Operating Income + Depreciation & Amortization.")

    # --------------------------------------------------------
    # NET INCOME
    # --------------------------------------------------------

    net_income = first_valid(
        income.get("netIncome"),
        key_metrics.get("netIncomePerShareTTM"),
    )

    # --------------------------------------------------------
    # TRAILING P/E
    # --------------------------------------------------------

    trailing_pe = first_valid(
        ratios.get("peRatioTTM"),
        ratios.get("priceEarningsRatioTTM"),
        yinfo.get("trailingPE"),
    )

    if trailing_pe is None:
        trailing_eps = first_valid(
            yinfo.get("trailingEps"),
            income.get("eps"),
            income.get("epsdiluted"),
        )

        if last_close is not None and trailing_eps is not None and trailing_eps > 0:
            trailing_pe = last_close / trailing_eps
            notes.append("Trailing P/E calculated manually from Last Price / EPS TTM.")

    if trailing_pe is None:
        trailing_pe = calculate_trailing_pe(
            market_cap=market_cap,
            net_income=net_income,
        )

        if trailing_pe is not None:
            notes.append("Trailing P/E calculated manually from Market Cap / Net Income.")

    if trailing_pe is None and net_income is not None and net_income <= 0:
        notes.append("Trailing P/E is N/A because Net Income is negative or zero.")

    # --------------------------------------------------------
    # FORWARD P/E
    # --------------------------------------------------------

    forward_pe = first_valid(
        yinfo.get("forwardPE"),
    )

    forward_eps = first_valid(
        yinfo.get("forwardEps"),
    )

    if forward_eps is None:
        estimates = fmp.get("analyst_estimates", []) or []

        for estimate in estimates:
            possible_eps = first_valid(
                estimate.get("estimatedEpsAvg"),
                estimate.get("estimatedEpsHigh"),
                estimate.get("estimatedEpsLow"),
            )

            if possible_eps is not None and possible_eps > 0:
                forward_eps = possible_eps
                notes.append("Forward EPS obtained from FMP analyst estimates.")
                break

    if forward_pe is None:
        forward_pe = calculate_forward_pe(
            last_price=last_close,
            forward_eps=forward_eps,
        )

        if forward_pe is not None:
            notes.append("Forward P/E calculated manually from Last Price / Forward EPS.")

    if forward_pe is None:
        notes.append("Forward P/E is N/A because Forward EPS estimate is unavailable.")

    # --------------------------------------------------------
    # EV / EBITDA
    # --------------------------------------------------------

    ev_ebitda = first_valid(
        key_metrics.get("enterpriseValueOverEBITDATTM"),
        key_metrics.get("evToEBITDATTM"),
        ratios.get("enterpriseValueMultipleTTM"),
        yinfo.get("enterpriseToEbitda"),
    )

    if ev_ebitda is None:
        ev_ebitda = calculate_ev_ebitda(
            enterprise_value=enterprise_value,
            ebitda=ebitda,
        )

        if ev_ebitda is not None:
            notes.append("EV / EBITDA calculated manually from Enterprise Value / EBITDA.")

    if ev_ebitda is None:
        if ebitda is not None and ebitda <= 0:
            notes.append("EV / EBITDA is N/A because EBITDA is negative or zero.")
        else:
            notes.append("EV / EBITDA is N/A because Enterprise Value or EBITDA is unavailable.")

    # --------------------------------------------------------
    # EARNINGS DATE
    # --------------------------------------------------------

    next_earnings_date = None

    yahoo_earnings = extract_yahoo_earnings_date(yahoo.get("calendar"))
    fmp_earnings = extract_fmp_earnings_date(fmp.get("earnings_calendar", []))
    finnhub_date = extract_finnhub_earnings_date(finnhub_earnings)

    next_earnings_date = yahoo_earnings or fmp_earnings or finnhub_date

    if next_earnings_date is None:
        next_earnings_date = "N/A"
        notes.append("Next Earnings Date unavailable from Yahoo, FMP, and Finnhub.")

    # --------------------------------------------------------
    # DEBUG TABLE
    # --------------------------------------------------------

    debug_rows = [
        {
            "Metric": "Last Close",
            "Value": fmt_number(last_close),
            "Status": ratio_status(last_close),
            "Comment": source_last_close,
        },
        {
            "Metric": "Market Cap",
            "Value": fmt_large(market_cap),
            "Status": ratio_status(market_cap),
            "Comment": "Used for manual P/E and EV calculations.",
        },
        {
            "Metric": "Total Debt",
            "Value": fmt_large(total_debt),
            "Status": ratio_status(total_debt),
            "Comment": "Used for Enterprise Value.",
        },
        {
            "Metric": "Cash",
            "Value": fmt_large(cash),
            "Status": ratio_status(cash),
            "Comment": "Used for Enterprise Value.",
        },
        {
            "Metric": "Enterprise Value",
            "Value": fmt_large(enterprise_value),
            "Status": ratio_status(enterprise_value),
            "Comment": "Direct value or Market Cap + Debt - Cash.",
        },
        {
            "Metric": "EBITDA",
            "Value": fmt_large(ebitda),
            "Status": ratio_status(ebitda),
            "Comment": "Direct value or Operating Income + D&A.",
        },
        {
            "Metric": "Net Income",
            "Value": fmt_large(net_income),
            "Status": ratio_status(net_income),
            "Comment": "Used for manual Trailing P/E.",
        },
        {
            "Metric": "Forward EPS",
            "Value": fmt_number(forward_eps),
            "Status": ratio_status(forward_eps),
            "Comment": "Required for manual Forward P/E.",
        },
    ]

    return {
        "ticker": ticker,
        "last_close": last_close,
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "total_debt": total_debt,
        "cash": cash,
        "ebitda": ebitda,
        "net_income": net_income,
        "trailing_pe": trailing_pe,
        "forward_eps": forward_eps,
        "forward_pe": forward_pe,
        "ev_ebitda": ev_ebitda,
        "next_earnings_date": next_earnings_date,
        "notes": notes,
        "debug_table": pd.DataFrame(debug_rows),
        "history": hist,
    }


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_technicals(df):
    if df is None or df.empty:
        return df

    df = df.copy()

    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["RSI_14"] = calculate_rsi(df["Close"], 14)

    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


def technical_summary(df):
    if df is None or df.empty:
        return {
            "trend": "N/A",
            "rsi": None,
            "sma_20": None,
            "sma_50": None,
            "sma_200": None,
            "comment": "No price history available.",
        }

    df = add_technicals(df)
    last = df.dropna().iloc[-1]

    close = safe_float(last.get("Close"))
    sma_20 = safe_float(last.get("SMA_20"))
    sma_50 = safe_float(last.get("SMA_50"))
    sma_200 = safe_float(last.get("SMA_200"))
    rsi = safe_float(last.get("RSI_14"))

    trend = "Neutral"

    if close and sma_50 and sma_200:
        if close > sma_50 > sma_200:
            trend = "Bullish"
        elif close < sma_50 < sma_200:
            trend = "Bearish"

    if rsi is None:
        comment = "RSI unavailable."
    elif rsi >= 70:
        comment = "RSI suggests the stock may be overbought."
    elif rsi <= 30:
        comment = "RSI suggests the stock may be oversold."
    else:
        comment = "RSI is in a neutral range."

    return {
        "trend": trend,
        "rsi": rsi,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "comment": comment,
    }


# ============================================================
# OPTIONS DATA
# ============================================================

@st.cache_data(ttl=1800)
def get_options_snapshot(ticker):
    ticker = ticker.upper().strip()

    result = {
        "expiration": None,
        "atm_call_iv": None,
        "atm_put_iv": None,
        "atm_strike": None,
        "notes": [],
    }

    try:
        tk = yf.Ticker(ticker)
        dates = list(tk.options)

        if not dates:
            result["notes"].append("No options chain available from Yahoo.")
            return result

        expiration = dates[0]
        chain = tk.option_chain(expiration)

        calls = chain.calls
        puts = chain.puts

        hist = tk.history(period="5d")
        if hist.empty:
            result["notes"].append("No recent price available for ATM option selection.")
            return result

        price = safe_float(hist["Close"].iloc[-1])

        if price is None:
            result["notes"].append("No price available for ATM option selection.")
            return result

        calls["distance"] = abs(calls["strike"] - price)
        puts["distance"] = abs(puts["strike"] - price)

        atm_call = calls.sort_values("distance").iloc[0]
        atm_put = puts.sort_values("distance").iloc[0]

        result["expiration"] = expiration
        result["atm_strike"] = safe_float(atm_call["strike"])
        result["atm_call_iv"] = safe_float(atm_call.get("impliedVolatility"))
        result["atm_put_iv"] = safe_float(atm_put.get("impliedVolatility"))

    except Exception as e:
        result["notes"].append(f"Options data unavailable: {e}")

    return result


# ============================================================
# VALUATION INTERPRETATION
# ============================================================

def valuation_comment(trailing_pe, forward_pe, ev_ebitda):
    comments = []

    if trailing_pe is not None:
        if trailing_pe < 15:
            comments.append("Trailing P/E looks relatively low.")
        elif trailing_pe <= 30:
            comments.append("Trailing P/E is moderate.")
        else:
            comments.append("Trailing P/E is elevated.")

    if forward_pe is not None:
        if forward_pe < 15:
            comments.append("Forward P/E looks relatively low.")
        elif forward_pe <= 30:
            comments.append("Forward P/E is moderate.")
        else:
            comments.append("Forward P/E is elevated.")

    if ev_ebitda is not None:
        if ev_ebitda < 10:
            comments.append("EV / EBITDA looks relatively low.")
        elif ev_ebitda <= 20:
            comments.append("EV / EBITDA is moderate.")
        else:
            comments.append("EV / EBITDA is elevated.")

    if not comments:
        return "Valuation interpretation unavailable because the required ratios are missing."

    return " ".join(comments)


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Inputs")

ticker = st.sidebar.text_input("Ticker", value="MOD").upper().strip()

st.sidebar.caption("Optional API keys can be added in Streamlit secrets:")
st.sidebar.code(
    """
FMP_API_KEY = "your_key"
FINNHUB_API_KEY = "your_key"
    """,
    language="toml",
)

refresh = st.sidebar.button("Refresh Data")

if refresh:
    st.cache_data.clear()
    st.rerun()


# ============================================================
# LOAD DATA
# ============================================================

if not ticker:
    st.warning("Please enter a ticker.")
    st.stop()

data = get_fundamentals(ticker, FMP_API_KEY, FINNHUB_API_KEY)
history = data["history"]

tabs = st.tabs(["Overview", "Technical", "Valuation", "Options", "Scanner", "Data Debug"])


# ============================================================
# OVERVIEW TAB
# ============================================================

with tabs[0]:
    st.header(data["ticker"])

    st.write(f"**Next Earnings Date:** {data['next_earnings_date']}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Last Close", metric_value(data["last_close"]))

    with col2:
        st.metric("Trailing P/E", metric_value(data["trailing_pe"]))

    with col3:
        st.metric("Forward P/E", metric_value(data["forward_pe"]))

    with col4:
        st.metric("EV / EBITDA", metric_value(data["ev_ebitda"]))

    st.divider()

    st.subheader("Financial Snapshot")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Market Cap", fmt_large(data["market_cap"]))

    with c2:
        st.metric("Enterprise Value", fmt_large(data["enterprise_value"]))

    with c3:
        st.metric("EBITDA", fmt_large(data["ebitda"]))

    with c4:
        st.metric("Net Income", fmt_large(data["net_income"]))

    if data["notes"]:
        with st.expander("Data Notes"):
            for note in data["notes"]:
                st.write(f"- {note}")


# ============================================================
# TECHNICAL TAB
# ============================================================

with tabs[1]:
    st.header("Technical Analysis")

    if history is None or history.empty:
        st.warning("No price history available.")
    else:
        tech_df = add_technicals(history)
        summary = technical_summary(history)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Trend", summary["trend"])

        with c2:
            st.metric("RSI 14", metric_value(summary["rsi"]))

        with c3:
            st.metric("SMA 50", metric_value(summary["sma_50"]))

        with c4:
            st.metric("SMA 200", metric_value(summary["sma_200"]))

        st.write(summary["comment"])

        chart_df = tech_df[["Close", "SMA_20", "SMA_50", "SMA_200"]].dropna()
        st.line_chart(chart_df)

        st.subheader("Recent Price Data")
        st.dataframe(tech_df.tail(20), use_container_width=True)


# ============================================================
# VALUATION TAB
# ============================================================

with tabs[2]:
    st.header("Valuation")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Trailing P/E", metric_value(data["trailing_pe"]))

    with c2:
        st.metric("Forward P/E", metric_value(data["forward_pe"]))

    with c3:
        st.metric("EV / EBITDA", metric_value(data["ev_ebitda"]))

    st.subheader("Interpretation")
    st.write(
        valuation_comment(
            data["trailing_pe"],
            data["forward_pe"],
            data["ev_ebitda"],
        )
    )

    st.subheader("Manual Calculation Logic")

    st.code(
        """
Enterprise Value = Market Cap + Total Debt - Cash

EV / EBITDA = Enterprise Value / EBITDA

Trailing P/E = Market Cap / Net Income

Forward P/E = Last Price / Forward EPS
        """,
        language="text",
    )

    st.subheader("Raw Inputs Used")

    raw_inputs = pd.DataFrame(
        [
            ["Last Close", fmt_number(data["last_close"])],
            ["Market Cap", fmt_large(data["market_cap"])],
            ["Total Debt", fmt_large(data["total_debt"])],
            ["Cash", fmt_large(data["cash"])],
            ["Enterprise Value", fmt_large(data["enterprise_value"])],
            ["EBITDA", fmt_large(data["ebitda"])],
            ["Net Income", fmt_large(data["net_income"])],
            ["Forward EPS", fmt_number(data["forward_eps"])],
        ],
        columns=["Input", "Value"],
    )

    st.dataframe(raw_inputs, use_container_width=True)


# ============================================================
# OPTIONS TAB
# ============================================================

with tabs[3]:
    st.header("Options Snapshot")

    options = get_options_snapshot(ticker)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Nearest Expiration", options["expiration"] or "N/A")

    with c2:
        st.metric("ATM Strike", metric_value(options["atm_strike"]))

    with c3:
        call_iv = options["atm_call_iv"]
        st.metric(
            "ATM Call IV",
            f"{call_iv * 100:.2f}%" if call_iv is not None else "N/A",
        )

    with c4:
        put_iv = options["atm_put_iv"]
        st.metric(
            "ATM Put IV",
            f"{put_iv * 100:.2f}%" if put_iv is not None else "N/A",
        )

    if options["notes"]:
        with st.expander("Options Notes"):
            for note in options["notes"]:
                st.write(f"- {note}")

    st.info(
        "For bull put spreads, you may look for liquid stocks with elevated IV, "
        "solid trend support, and strikes below major technical levels."
    )


# ============================================================
# SCANNER TAB
# ============================================================

with tabs[4]:
    st.header("Simple Scanner")

    st.write(
        "Enter tickers separated by commas. The scanner uses the same robust "
        "fallback logic as the main dashboard."
    )

    ticker_text = st.text_area(
        "Tickers",
        value="NVDA, AMD, AVGO, TSM, ASML, MU, MOD, ORCL, META, MSFT",
        height=100,
    )

    run_scan = st.button("Run Scanner")

    if run_scan:
        scan_tickers = [
            t.strip().upper()
            for t in ticker_text.replace("\n", ",").split(",")
            if t.strip()
        ]

        rows = []

        with st.spinner("Scanning tickers..."):
            for symbol in scan_tickers:
                try:
                    d = get_fundamentals(symbol, FMP_API_KEY, FINNHUB_API_KEY)
                    tech = technical_summary(d["history"])

                    rows.append(
                        {
                            "Ticker": symbol,
                            "Last Close": d["last_close"],
                            "Trailing P/E": d["trailing_pe"],
                            "Forward P/E": d["forward_pe"],
                            "EV / EBITDA": d["ev_ebitda"],
                            "Market Cap": d["market_cap"],
                            "Trend": tech["trend"],
                            "RSI": tech["rsi"],
                            "Next Earnings": d["next_earnings_date"],
                        }
                    )

                except Exception as e:
                    rows.append(
                        {
                            "Ticker": symbol,
                            "Last Close": None,
                            "Trailing P/E": None,
                            "Forward P/E": None,
                            "EV / EBITDA": None,
                            "Market Cap": None,
                            "Trend": "Error",
                            "RSI": None,
                            "Next Earnings": "N/A",
                        }
                    )

        scan_df = pd.DataFrame(rows)

        st.dataframe(
            scan_df.style.format(
                {
                    "Last Close": "{:,.2f}",
                    "Trailing P/E": "{:,.2f}",
                    "Forward P/E": "{:,.2f}",
                    "EV / EBITDA": "{:,.2f}",
                    "Market Cap": "{:,.0f}",
                    "RSI": "{:,.2f}",
                },
                na_rep="N/A",
            ),
            use_container_width=True,
        )


# ============================================================
# DATA DEBUG TAB
# ============================================================

with tabs[5]:
    st.header("Data Debug")

    st.write(
        "This table helps identify whether a value was available, calculated, "
        "or truly missing."
    )

    st.dataframe(data["debug_table"], use_container_width=True)

    if data["notes"]:
        st.subheader("Data Notes")
        for note in data["notes"]:
            st.write(f"- {note}")

    st.subheader("Important Reminder")

    st.warning(
        "Some fields may still show N/A when the required raw data is unavailable. "
        "For example, Forward P/E requires Forward EPS estimates. If no provider "
        "returns Forward EPS, the app should not invent it."
    )
