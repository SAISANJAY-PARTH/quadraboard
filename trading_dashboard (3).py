import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Pro Trading Dashboard", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .main { background-color: #0a0e1a; }
    .stMetric { background: #111827; border-radius: 10px; padding: 10px; }
    .stAlert { border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
    .conflict-box {
        background: #1a1f2e;
        border-left: 4px solid #FFD700;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .verdict-strong-buy  { border-left-color: #00FF7F !important; }
    .verdict-buy         { border-left-color: #7CFC00 !important; }
    .verdict-strong-sell { border-left-color: #FF0000 !important; }
    .verdict-sell        { border-left-color: #FF4500 !important; }
    .verdict-range       { border-left-color: #FFD700 !important; }
    .verdict-wait        { border-left-color: #888888 !important; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Pro Technical Analysis Dashboard")

st.subheader("⚠️ Data & Usage Disclaimer")
st.info("""
This dashboard uses Yahoo Finance (yfinance) data, which may have delays, missing values, or inaccuracies.

Supported assets include:
• Stocks (India, US, global markets) | • Indices (NIFTY, S&P 500, etc.) | • ETFs | • Cryptocurrencies (BTC-USD, ETH-USD)

This tool is for **educational and informational purposes only** and should NOT be considered financial advice.
""")

# ---------------------------
# TICKER SEARCH ENGINE
# ---------------------------

@st.cache_data(ttl=3600)
def search_tickers(query):
    import urllib.request, json
    query = query.strip()
    if not query:
        return []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(query)}&lang=en-US&region=IN&quotesCount=15&newsCount=0&listsCount=0"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        results = []
        for item in data.get("quotes", []):
            sym   = item.get("symbol", "")
            name  = item.get("longname") or item.get("shortname") or sym
            exch  = item.get("exchDisp", "")
            qtype = item.get("quoteType", "")
            if sym:
                results.append({
                    "symbol": sym,
                    "name": name,
                    "exchange": exch,
                    "type": qtype,
                    "label": f"{name}  [{sym}]  {exch}"
                })
        return results
    except Exception:
        return []

import urllib.parse

st.markdown("### 🔍 Stock Search")
st.caption("Type a company name (e.g. **Reliance**, **TCS**, **Infosys**, **Apple**, **BTC**) — no need to know the ticker symbol")

search_col1, search_col2 = st.columns([2, 1])

with search_col1:
    search_query = st.text_input(
        "Search Company / Stock Name",
        value=st.session_state.get("last_search", "Reliance"),
        placeholder="e.g. Reliance, Tata Motors, Apple, Bitcoin...",
        key="search_input"
    )

search_results = []
ticker = st.session_state.get("selected_ticker", "RELIANCE.NS")

if search_query and len(search_query) >= 2:
    with st.spinner("Searching..."):
        search_results = search_tickers(search_query)

if search_results:
    labels  = [r["label"] for r in search_results]
    symbols = [r["symbol"] for r in search_results]

    default_idx = 0
    for i, sym in enumerate(symbols):
        if sym == st.session_state.get("selected_ticker", ""):
            default_idx = i
            break

    st.markdown(f"**{len(search_results)} results found for '{search_query}':**")

    selected_label = st.selectbox(
        "Select Stock",
        labels,
        index=default_idx,
        key="ticker_select"
    )

    selected_idx    = labels.index(selected_label)
    ticker          = symbols[selected_idx]
    selected_info   = search_results[selected_idx]

    st.session_state["selected_ticker"] = ticker
    st.session_state["last_search"]     = search_query

    badge_color = {
        "EQUITY": "🟦", "ETF": "🟩", "CRYPTOCURRENCY": "🟡",
        "INDEX": "🟣", "MUTUALFUND": "🟠"
    }.get(selected_info["type"].upper(), "⬜")

    st.success(f"{badge_color} **Selected:** {selected_info['name']}  |  Symbol: `{ticker}`  |  Exchange: {selected_info['exchange']}  |  Type: {selected_info['type']}")

elif search_query and len(search_query) >= 2:
    st.warning("No results found. Try a different name.")
    ticker = st.text_input("Or enter ticker manually", value=ticker).upper()
else:
    st.info("Start typing a company name above to search.")
    ticker = st.session_state.get("selected_ticker", "RELIANCE.NS")

col2, col3 = st.columns(2)
with col2:
    period = st.selectbox(
        "Time Period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=4
    )
with col3:
    chart_type = st.selectbox("Chart Type", ["Candlestick", "Heikin Ashi", "Line"])

# ---------------------------
# FETCH DATA
# ---------------------------
try:
    stock = yf.Ticker(ticker)
    hist  = stock.history(period=period)

    if hist.empty:
        st.error(f"❌ No data returned for `{ticker}`. Try a different selection.")
        st.stop()
except Exception as e:
    st.error(f"❌ Data fetch error: {e}")
    st.stop()

try:
    fast = stock.fast_info
except:
    fast = {}

# ---------------------------
# INDICATORS ENGINE
# ---------------------------
df = hist.copy()

df["EMA9"]   = df["Close"].ewm(span=9).mean()
df["EMA20"]  = df["Close"].ewm(span=20).mean()
df["EMA44"]  = df["Close"].ewm(span=44).mean()
df["EMA50"]  = df["Close"].ewm(span=50).mean()
df["EMA200"] = df["Close"].ewm(span=200).mean()
df["SMA50"]  = df["Close"].rolling(50).mean()
df["SMA200"] = df["Close"].rolling(200).mean()

delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

rsi_min = df["RSI"].rolling(14).min()
rsi_max = df["RSI"].rolling(14).max()
df["StochRSI"] = (df["RSI"] - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
df["StochK"] = df["StochRSI"].rolling(3).mean()
df["StochD"] = df["StochK"].rolling(3).mean()

df["MACD"]        = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

df["TR"] = df["High"] - df["Low"]
df["ATR"] = df["TR"].rolling(14).mean()

prev_close = df["Close"].shift(1)
tr1 = df["High"] - df["Low"]
tr2 = (df["High"] - prev_close).abs()
tr3 = (df["Low"] - prev_close).abs()
df["True_ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

df["+DM"] = df["High"].diff()
df["-DM"] = df["Low"].diff()
df["+DM"] = df["+DM"].where((df["+DM"] > df["-DM"]) & (df["+DM"] > 0), 0)
df["-DM"] = df["-DM"].where((df["-DM"] > df["+DM"]) & (df["-DM"] > 0), 0)
tr14 = df["True_ATR"].rolling(14).sum()
plus_di  = 100 * (df["+DM"].rolling(14).sum() / (tr14 + 1e-10))
minus_di = 100 * (df["-DM"].rolling(14).sum() / (tr14 + 1e-10))
dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
df["ADX"]      = dx.rolling(14).mean()
df["Plus_DI"]  = plus_di
df["Minus_DI"] = minus_di

df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

df["BB_Mid"]   = df["Close"].rolling(20).mean()
df["BB_STD"]   = df["Close"].rolling(20).std()
df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_STD"]
df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_STD"]
df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
df["BB_Pct"]   = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-10)

df["KC_Mid"]   = df["Close"].ewm(span=20).mean()
df["KC_Upper"] = df["KC_Mid"] + 2 * df["True_ATR"]
df["KC_Lower"] = df["KC_Mid"] - 2 * df["True_ATR"]
df["Squeeze"]  = df["BB_Width"] < (df["KC_Upper"] - df["KC_Lower"]) / df["KC_Mid"]

df["Support"]    = df["Low"].rolling(20).min()
df["Resistance"] = df["High"].rolling(20).max()

df["Pivot"] = (df["High"] + df["Low"] + df["Close"]) / 3
df["R1"]    = 2 * df["Pivot"] - df["Low"]
df["S1"]    = 2 * df["Pivot"] - df["High"]
df["R2"]    = df["Pivot"] + (df["High"] - df["Low"])
df["S2"]    = df["Pivot"] - (df["High"] - df["Low"])

obv = [0]
for i in range(1, len(df)):
    if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
        obv.append(obv[-1] + df["Volume"].iloc[i])
    elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
        obv.append(obv[-1] - df["Volume"].iloc[i])
    else:
        obv.append(obv[-1])
df["OBV"] = obv

mfv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-10) * df["Volume"]
df["CMF"] = mfv.rolling(20).sum() / df["Volume"].rolling(20).sum()

df["WilliamsR"] = -100 * (df["High"].rolling(14).max() - df["Close"]) / (df["High"].rolling(14).max() - df["Low"].rolling(14).min() + 1e-10)

tp = (df["High"] + df["Low"] + df["Close"]) / 3
df["CCI"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-10)

multiplier = 3
df["BasicUpper"] = (df["High"] + df["Low"]) / 2 + multiplier * df["True_ATR"]
df["BasicLower"] = (df["High"] + df["Low"]) / 2 - multiplier * df["True_ATR"]
df["Supertrend"] = df["BasicLower"]
df["ST_Dir"] = 1
for i in range(1, len(df)):
    if df["Close"].iloc[i] > df["BasicUpper"].iloc[i-1]:
        df.loc[df.index[i], "ST_Dir"] = 1
    elif df["Close"].iloc[i] < df["BasicLower"].iloc[i-1]:
        df.loc[df.index[i], "ST_Dir"] = -1
    else:
        df.loc[df.index[i], "ST_Dir"] = df["ST_Dir"].iloc[i-1]

df["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
df["HA_Open"]  = ((df["Open"].shift(1) + df["Close"].shift(1)) / 2).fillna(df["Open"])
df["HA_High"]  = df[["High", "HA_Open", "HA_Close"]].max(axis=1)
df["HA_Low"]   = df[["Low",  "HA_Open", "HA_Close"]].min(axis=1)

df["ROC"] = df["Close"].pct_change(10) * 100

tp2 = (df["High"] + df["Low"] + df["Close"]) / 3
mf  = tp2 * df["Volume"]
pos_mf = mf.where(tp2 > tp2.shift(1), 0).rolling(14).sum()
neg_mf = mf.where(tp2 < tp2.shift(1), 0).rolling(14).sum()
df["MFI"] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))

af, max_af, sar = 0.02, 0.20, df["Low"].iloc[0]
trend, ep = 1, df["High"].iloc[0]
psar = []
for i in range(len(df)):
    psar.append(sar)
    if trend == 1:
        sar = sar + af * (ep - sar)
        sar = min(sar, df["Low"].iloc[max(0,i-1)], df["Low"].iloc[max(0,i-2)])
        if df["Low"].iloc[i] < sar:
            trend, sar, ep, af = -1, ep, df["Low"].iloc[i], 0.02
        else:
            if df["High"].iloc[i] > ep:
                ep = df["High"].iloc[i]
                af = min(af + 0.02, max_af)
    else:
        sar = sar + af * (ep - sar)
        sar = max(sar, df["High"].iloc[max(0,i-1)], df["High"].iloc[max(0,i-2)])
        if df["High"].iloc[i] > sar:
            trend, sar, ep, af = 1, ep, df["High"].iloc[i], 0.02
        else:
            if df["Low"].iloc[i] < ep:
                ep = df["Low"].iloc[i]
                af = min(af + 0.02, max_af)
df["PSAR"] = psar

# ---------------------------
# INDICATOR SELECTOR
# ---------------------------
st.subheader("📈 Advanced Chart")

selected = st.multiselect(
    "Select Indicators",
    ["EMA", "SMA", "VWAP", "Bollinger Bands", "Keltner Channel",
     "Support/Resistance", "Pivot Points", "Supertrend", "PSAR",
     "RSI", "Stoch RSI", "MACD", "Williams %R", "CCI", "MFI",
     "Volume", "OBV", "CMF", "ATR", "ADX", "ROC"],
    default=["EMA", "RSI", "MACD"]
)

def add_label(fig, x, y, name, row):
    y_clean = y.dropna()
    if y_clean.empty:
        return
    fig.add_annotation(
        x=x[-1], y=y_clean.iloc[-1],
        text=f"{name}: {y_clean.iloc[-1]:.2f}",
        showarrow=False, xanchor="left",
        bgcolor="#111827", bordercolor="gray",
        font=dict(size=10), row=row, col=1
    )

lower_panels = []
if any(x in selected for x in ["RSI", "Stoch RSI", "Williams %R", "MFI"]):
    lower_panels.append("momentum")
if any(x in selected for x in ["MACD"]):
    lower_panels.append("macd")
if any(x in selected for x in ["Volume", "OBV", "CMF"]):
    lower_panels.append("volume")
if any(x in selected for x in ["ATR", "ADX", "ROC", "CCI"]):
    lower_panels.append("misc")

n_rows = 1 + len(lower_panels)
row_heights = [0.55] + [round(0.45 / max(len(lower_panels), 1), 2)] * len(lower_panels)

fig = make_subplots(
    rows=n_rows, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=row_heights,
    subplot_titles=["Price"] + [p.upper() for p in lower_panels]
)

panel = {name: (i + 2) for i, name in enumerate(lower_panels)}

if chart_type == "Heikin Ashi":
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["HA_Open"], high=df["HA_High"],
        low=df["HA_Low"], close=df["HA_Close"], name="Heikin Ashi"
    ), row=1, col=1)
elif chart_type == "Line":
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                              line=dict(color="#00d4ff", width=2)), row=1, col=1)
else:
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"
    ), row=1, col=1)

ema_colors = {"EMA9":"#FF6B6B","EMA20":"#FFD93D","EMA50":"#6BCB77","EMA200":"#4D96FF"}
if "EMA" in selected:
    for col_name, color in ema_colors.items():
        if len(df) >= int(col_name.replace("EMA","")):
            fig.add_trace(go.Scatter(x=df.index, y=df[col_name], name=col_name,
                                      line=dict(color=color, width=1.2)), row=1, col=1)

if "SMA" in selected:
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50",
                              line=dict(color="#FFA07A", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200",
                              line=dict(color="#20B2AA", dash="dot")), row=1, col=1)

if "VWAP" in selected:
    fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP",
                              line=dict(color="#DA70D6", dash="dash")), row=1, col=1)

if "Bollinger Bands" in selected:
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                              line=dict(color="#888", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                              line=dict(color="#888", dash="dash"),
                              fill="tonexty", fillcolor="rgba(100,100,100,0.1)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Mid"], name="BB Mid",
                              line=dict(color="#aaa", width=1)), row=1, col=1)

if "Keltner Channel" in selected:
    fig.add_trace(go.Scatter(x=df.index, y=df["KC_Upper"], name="KC Upper",
                              line=dict(color="#FF8C00", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["KC_Lower"], name="KC Lower",
                              line=dict(color="#FF8C00", dash="dot")), row=1, col=1)

if "Support/Resistance" in selected:
    fig.add_trace(go.Scatter(x=df.index, y=df["Support"], name="Support",
                              line=dict(color="#00FF7F", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Resistance"], name="Resistance",
                              line=dict(color="#FF4500", dash="dot")), row=1, col=1)

if "Pivot Points" in selected:
    for lvl, color in [("Pivot","#FFD700"),("R1","#FF6347"),("S1","#7CFC00"),("R2","#FF0000"),("S2","#00FA9A")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[lvl], name=lvl,
                                  line=dict(color=color, width=1, dash="dot")), row=1, col=1)

if "Supertrend" in selected:
    bull = df["Supertrend"].where(df["ST_Dir"] == 1)
    bear = df["Supertrend"].where(df["ST_Dir"] == -1)
    fig.add_trace(go.Scatter(x=df.index, y=bull, name="ST Bull",
                              line=dict(color="#00FF7F", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bear, name="ST Bear",
                              line=dict(color="#FF4500", width=2)), row=1, col=1)

if "PSAR" in selected:
    fig.add_trace(go.Scatter(x=df.index, y=df["PSAR"], name="PSAR",
                              mode="markers", marker=dict(size=3, color="#FFD700")), row=1, col=1)

if "momentum" in panel:
    r = panel["momentum"]
    if "RSI" in selected:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                                  line=dict(color="#FFD93D")), row=r, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=r, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=r, col=1)
    if "Stoch RSI" in selected:
        fig.add_trace(go.Scatter(x=df.index, y=df["StochK"], name="StochK",
                                  line=dict(color="#00d4ff")), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["StochD"], name="StochD",
                                  line=dict(color="#FF6B6B")), row=r, col=1)
    if "Williams %R" in selected:
        fig.add_trace(go.Scatter(x=df.index, y=df["WilliamsR"], name="%R",
                                  line=dict(color="#DA70D6")), row=r, col=1)
    if "MFI" in selected:
        fig.add_trace(go.Scatter(x=df.index, y=df["MFI"], name="MFI",
                                  line=dict(color="#FFA07A")), row=r, col=1)

if "macd" in panel:
    r = panel["macd"]
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                              line=dict(color="#00d4ff")), row=r, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                              line=dict(color="#FF6B6B")), row=r, col=1)
    colors_hist = ["#00FF7F" if v >= 0 else "#FF4500" for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram",
                          marker_color=colors_hist), row=r, col=1)

if "volume" in panel:
    r = panel["volume"]
    if "Volume" in selected:
        vol_colors = ["#00FF7F" if c >= o else "#FF4500"
                      for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                              marker_color=vol_colors), row=r, col=1)
    if "OBV" in selected:
        fig.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV",
                                  line=dict(color="#FFD93D")), row=r, col=1)
    if "CMF" in selected:
        cmf_colors = ["#00FF7F" if v >= 0 else "#FF4500" for v in df["CMF"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["CMF"], name="CMF",
                              marker_color=cmf_colors), row=r, col=1)

if "misc" in panel:
    r = panel["misc"]
    if "ATR" in selected:
        fig.add_trace(go.Scatter(x=df.index, y=df["True_ATR"], name="ATR",
                                  line=dict(color="#FF8C00")), row=r, col=1)
    if "ADX" in selected:
        fig.add_trace(go.Scatter(x=df.index, y=df["ADX"], name="ADX",
                                  line=dict(color="#4D96FF")), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Plus_DI"], name="+DI",
                                  line=dict(color="#00FF7F", dash="dot")), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Minus_DI"], name="-DI",
                                  line=dict(color="#FF4500", dash="dot")), row=r, col=1)
    if "ROC" in selected:
        fig.add_trace(go.Scatter(x=df.index, y=df["ROC"], name="ROC",
                                  line=dict(color="#DA70D6")), row=r, col=1)
    if "CCI" in selected:
        fig.add_trace(go.Scatter(x=df.index, y=df["CCI"], name="CCI",
                                  line=dict(color="#FFD93D")), row=r, col=1)

for i in range(1, n_rows + 1):
    fig.update_yaxes(side="right", row=i, col=1)

fig.update_layout(
    template="plotly_dark",
    height=900,
    showlegend=True,
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    margin=dict(l=10, r=80, t=60, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# 📌 PRICE SUMMARY
# ---------------------------
st.subheader("📌 Price Summary")

def get_currency_symbol(stock, ticker):
    try:
        info = stock.info
        currency = info.get("currency", "")
        mapping = {"INR":"₹","USD":"$","EUR":"€","GBP":"£","JPY":"¥"}
        return mapping.get(currency, currency)
    except:
        if ticker.endswith(".NS") or ticker.endswith(".BO"):
            return "₹"
        return "$"

symbol = get_currency_symbol(stock, ticker)

def format_price(x):
    if isinstance(x, (int, float)):
        return f"{symbol} {x:,.2f}"
    return "N/A"

price    = fast.get("lastPrice")
high     = fast.get("dayHigh")
low      = fast.get("dayLow")
prev     = fast.get("previousClose")
mkt_cap  = fast.get("marketCap")
volume   = fast.get("lastVolume")

if isinstance(price, (int, float)) and isinstance(prev, (int, float)):
    change     = price - prev
    change_pct = (change / prev) * 100
    delta      = f"{change:+.2f} ({change_pct:+.2f}%)"
else:
    delta = None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", format_price(price), delta)
c2.metric("Day High", format_price(high))
c3.metric("Day Low", format_price(low))
c4.metric("Prev Close", format_price(prev))
if isinstance(volume, (int, float)):
    c5.metric("Volume", f"{volume:,.0f}")

if isinstance(mkt_cap, (int, float)):
    st.caption(f"Market Cap: {symbol} {mkt_cap:,.0f}")

# ===========================================================
# 🧠 MULTI-FACTOR DECISION ENGINE  (FIXED & ENHANCED)
# ===========================================================
st.subheader("🧠 Multi-Factor Decision Engine")

last      = df.iloc[-1]
price_cur = last["Close"]
ema20_v   = last["EMA20"]
ema50_v   = last["EMA50"]
ema200_v  = last["EMA200"]
rsi_v     = last["RSI"]
macd_v    = last["MACD"]
macd_sig  = last["MACD_Signal"]
adx_v     = last["ADX"]
atr_v     = last["True_ATR"]
bb_pct    = last["BB_Pct"]
cmf_v     = last["CMF"] if not pd.isna(last["CMF"]) else 0
obv_slope = df["OBV"].iloc[-1] - df["OBV"].iloc[-5] if len(df) > 5 else 0
cci_v     = last["CCI"] if not pd.isna(last["CCI"]) else 0
mfi_v     = last["MFI"] if not pd.isna(last["MFI"]) else 50
will_r    = last["WilliamsR"] if not pd.isna(last["WilliamsR"]) else -50
st_dir    = last["ST_Dir"]
psar_v    = last["PSAR"]
# Candle structure (entry confirmation)
last_candle = df.iloc[-1]

upper_wick = last_candle["High"] - max(last_candle["Close"], last_candle["Open"])
lower_wick = min(last_candle["Close"], last_candle["Open"]) - last_candle["Low"]
body = abs(last_candle["Close"] - last_candle["Open"])

bearish_rejection = upper_wick > body * 1.5 and last_candle["Close"] < last_candle["Open"]
bullish_rejection = lower_wick > body * 1.5 and last_candle["Close"] > last_candle["Open"]

# ── MARKET STRUCTURE ANALYSIS ──
# Detect range / sideways condition
# ADX < 20 = no trend; 15-25 = transitional; >25 = trending
adx_trending  = adx_v > 20
adx_strong    = adx_v > 25
adx_very_weak = adx_v < 15
# ADX slope (trend strength direction)
adx_slope = df["ADX"].iloc[-1] - df["ADX"].iloc[-5] if len(df) > 5 else 0
adx_rising = adx_slope > 0

# Price range tightness: how wide is BB relative to its mean (squeeze detection)
bb_width_now  = last["BB_Width"] if not pd.isna(last["BB_Width"]) else 0
bb_width_avg  = df["BB_Width"].rolling(50).mean().iloc[-1] if not pd.isna(df["BB_Width"].rolling(50).mean().iloc[-1]) else bb_width_now
bb_squeeze    = bb_width_now < bb_width_avg * 0.7  # BB tighter than 70% of average = range

# EMA compression: all EMAs within tight band = range
ema_spread_pct = abs(ema20_v - ema50_v) / (ema50_v + 1e-10) * 100
ema_compressed = ema_spread_pct < 1.5  # EMAs within 1.5% of each other

# Pullback detection (IMPROVED)
pullback_active = (
    price_cur > ema20_v and
    price_cur < ema50_v and
    rsi_v > 50 and
    macd_v > macd_sig
)




# ── SCORING SYSTEM ──
bull_score = 0
bear_score = 0
signals    = []

# — Trend (max 6 points) —
if price_cur > ema200_v:
    bull_score += 2; signals.append(("✅ Price > EMA200 (Uptrend)", "bull"))
else:
    bear_score += 2; signals.append(("❌ Price < EMA200 (Downtrend)", "bear"))

if ema20_v > ema50_v > ema200_v:
    bull_score += 2; signals.append(("✅ EMA stack bullish (20>50>200)", "bull"))
elif ema20_v < ema50_v < ema200_v:
    bear_score += 2; signals.append(("❌ EMA stack bearish (20<50<200)", "bear"))
else:
    signals.append(("⚠️ EMA stack mixed — possible transition/range", "neutral"))

if st_dir == 1:
    bull_score += 1; signals.append(("✅ Supertrend Bullish", "bull"))
else:
    bear_score += 1; signals.append(("❌ Supertrend Bearish", "bear"))

if price_cur > psar_v:
    bull_score += 1; signals.append(("✅ Price above PSAR", "bull"))
else:
    bear_score += 1; signals.append(("❌ Price below PSAR", "bear"))

# — Momentum (max 4 points) —
if 50 < rsi_v < 70:
    bull_score += 2; signals.append((f"✅ RSI bullish zone ({rsi_v:.1f})", "bull"))
elif rsi_v > 70:
    signals.append((f"⚠️ RSI Overbought ({rsi_v:.1f})", "neutral"))
elif 30 < rsi_v < 50:
    bear_score += 2; signals.append((f"❌ RSI bearish zone ({rsi_v:.1f})", "bear"))
elif rsi_v < 30:
    signals.append((f"⚠️ RSI Oversold ({rsi_v:.1f})", "neutral"))

if macd_v > macd_sig:
    bull_score += 1; signals.append(("✅ MACD bullish crossover", "bull"))
else:
    bear_score += 1; signals.append(("❌ MACD bearish crossover", "bear"))

if -20 > will_r > -50:
    bull_score += 1; signals.append((f"✅ Williams %R neutral-bull ({will_r:.1f})", "bull"))
elif will_r > -20:
    signals.append((f"⚠️ Williams %R Overbought ({will_r:.1f})", "neutral"))
elif will_r < -80:
    signals.append((f"⚠️ Williams %R Oversold ({will_r:.1f})", "neutral"))
else:
    bear_score += 1; signals.append((f"❌ Williams %R bearish zone ({will_r:.1f})", "bear"))

# — Volume / Flow (max 3 points) —
if cmf_v > 0.05:
    bull_score += 1; signals.append(("✅ CMF positive (buying pressure)", "bull"))
elif cmf_v < -0.05:
    bear_score += 1; signals.append(("❌ CMF negative (selling pressure)", "bear"))
else:
    signals.append((f"⚠️ CMF near zero ({cmf_v:.3f}) — neutral flow", "neutral"))

if obv_slope > 0:
    bull_score += 1; signals.append(("✅ OBV rising (accumulation)", "bull"))
else:
    bear_score += 1; signals.append(("❌ OBV falling (distribution)", "bear"))

if mfi_v > 60:
    bull_score += 1; signals.append((f"✅ MFI bullish ({mfi_v:.1f})", "bull"))
elif mfi_v < 40:
    bear_score += 1; signals.append((f"❌ MFI bearish ({mfi_v:.1f})", "bear"))
else:
    signals.append((f"⚠️ MFI neutral ({mfi_v:.1f})", "neutral"))

# — Volatility context (informational, no score) —
if adx_strong:
    signals.append((f"✅ ADX strong trend ({adx_v:.1f}) — trend-follow OK", "bull" if bull_score > bear_score else "bear"))
elif adx_trending:
    signals.append((f"⚠️ ADX moderate ({adx_v:.1f}) — light trend present", "neutral"))
elif adx_very_weak:
    signals.append((f"⚠️ ADX very weak ({adx_v:.1f}) — choppy/range market", "neutral"))
else:
    signals.append((f"⚠️ ADX weak ({adx_v:.1f}) — range market, avoid trend signals", "neutral"))

# ── RANGE / CHOP DETECTION ──
range_evidence = []
if adx_very_weak:
    range_evidence.append(f"ADX={adx_v:.1f} (< 15, no trend)")
if ema_compressed:
    range_evidence.append(f"EMA spread={ema_spread_pct:.1f}% (< 1.5%, EMAs converging)")
if bb_squeeze:
    range_evidence.append(f"BB squeeze (bands tighter than {bb_width_avg*100:.1f}% avg)")

is_range_market = len(range_evidence) >= 2
# Strategy mode
if is_range_market:
    strategy_mode = "MEAN REVERSION"
elif adx_v > 20 and adx_rising:
    strategy_mode = "TREND FOLLOWING"
else:
    strategy_mode = "NO TRADE"
    # ── STRATEGY MODE OVERRIDE (NEW) ──

mid_price = (last["Support"] + last["Resistance"]) / 2
mid_range = abs(price_cur - mid_price) < (atr_v * 1.2)# 2 of 3 range signals = confirmed range
is_weak_trend   = not adx_trending            # ADX < 20

# ===========================================================
# 🧠 CLEAN DECISION ENGINE (FINAL FIXED VERSION)
# ===========================================================

# ── RAW SIGNAL ──
total     = bull_score + bear_score
bull_pct  = int((bull_score / max(total, 1)) * 100)

if bull_score >= 9:
    raw_signal = "STRONG BUY"
elif bull_score >= 6:
    raw_signal = "BUY"
elif bear_score >= 9:
    raw_signal = "STRONG SELL"
elif bear_score >= 6:
    raw_signal = "SELL"
else:
    raw_signal = "WAIT"


# ── INITIAL ASSIGNMENT ──
final_signal = raw_signal
signal_override = None

# ===========================================================
# 🧱 1. STRUCTURE FILTER (FIRST)
# ===========================================================
if strategy_mode == "NO TRADE":
    final_signal = "WAIT"
    signal_override = "⚠️ No clear structure. Skip trades."

# ===========================================================
# 📦 2. RANGE LOGIC (HIGHEST PRIORITY)
# ===========================================================
elif is_range_market:

    if mid_range:
        final_signal = "WAIT"
        signal_override = "⚠️ Mid-range. No edge. Avoid trades."

    else:
        final_signal = "RANGE"

        if raw_signal in ["BUY", "STRONG BUY"]:
            signal_override = "📦 Range: Buy ONLY near support."
        elif raw_signal in ["SELL", "STRONG SELL"]:
            signal_override = "📦 Range: Sell ONLY near resistance."
        else:
            signal_override = "📦 Sideways market. Wait for extremes."
# ===========================================================
# 🔄 2.5 PULLBACK DETECTION (NEW)
# ===========================================================
elif pullback_active and price_cur < ema200_v:
    final_signal = "PULLBACK"
    signal_override = "⚠️ Bearish trend + bullish pullback. Wait for short entry."

# ===========================================================
# 📉 3. WEAK TREND ADJUSTMENT
# ===========================================================
elif is_weak_trend:

    if raw_signal == "STRONG BUY":
        final_signal = "BUY"
        signal_override = f"⚠️ Weak trend (ADX={adx_v:.1f}). Reduced confidence."
    elif raw_signal == "STRONG SELL":
        final_signal = "SELL"
        signal_override = f"⚠️ Weak trend (ADX={adx_v:.1f}). Reduced confidence."

# ===========================================================
# ⚡ 4. MOMENTUM CONFLICT FILTER (SMART)
# ===========================================================
momentum_conflict_bear = (rsi_v > 60 and macd_v > macd_sig)
momentum_conflict_bull = (rsi_v < 40 and macd_v < macd_sig)

if "SELL" in final_signal and momentum_conflict_bear:
    final_signal = "WAIT"
    signal_override = "⚠️ Bearish setup but momentum bullish. Wait."

if "BUY" in final_signal and momentum_conflict_bull:
    final_signal = "WAIT"
    signal_override = "⚠️ Bullish setup but momentum weak. Wait."

# ===========================================================
# 📊 5. ADX QUALITY FILTER (FINAL GATE)
# ===========================================================
if adx_v < 18:
    final_signal = "WAIT"
    signal_override = f"⚠️ ADX={adx_v:.1f} → No real trend."

elif adx_v < 22:
    if "STRONG" in final_signal:
        final_signal = final_signal.replace("STRONG ", "")
        signal_override = f"⚠️ ADX={adx_v:.1f} → Strength downgraded."

# ===========================================================
# 🎯 6. ENTRY CONFIRMATION (SOFT ONLY)
# ===========================================================
entry_warning = None

if "SELL" in final_signal and not bearish_rejection:
    entry_warning = "⚠️ Weak bearish candle confirmation"

if "BUY" in final_signal and not bullish_rejection:
    entry_warning = "⚠️ Weak bullish candle confirmation"
    
col_a, col_b = st.columns(2)

with col_a:

    # 🧭 Strategy Context
    st.caption(f"🧭 Strategy Mode: {strategy_mode}")

    # 🎯 SIGNAL DISPLAY
    if final_signal == "STRONG BUY":
        st.success(f"🚀 {final_signal}")
    elif final_signal == "BUY":
        st.success(f"🟢 {final_signal}")
    elif final_signal == "STRONG SELL":
        st.error(f"🔻 {final_signal}")
    elif final_signal == "SELL":
        st.error(f"🔴 {final_signal}")
    elif final_signal == "RANGE":
        st.info("📦 RANGE — Market sideways. Trade extremes only.")
    elif final_signal == "PULLBACK":
        st.warning("🔄 PULLBACK — Trend intact, wait for continuation entry.")
    else:
        st.warning("⚠️ WAIT — No Clear Setup")

    # ⚠️ Entry Confirmation
    if entry_warning:
        st.warning(entry_warning)

    # 🔔 Override Display
    if signal_override:
        st.markdown(f"""
        <div style="background:#1a1f2e; border-left:4px solid #FFD700; border-radius:6px; padding:10px 14px; margin-top:8px;">
        <b>🔔 Signal Override</b><br><small>{signal_override}</small>
        </div>
        """, unsafe_allow_html=True)

    # 📊 Metrics (INSIDE column, not global)
    m1, m2, m3 = st.columns(3)
    m1.metric("Bull Score", f"{bull_score}", f"{bull_pct}% bullish")
    m2.metric("Bear Score", f"{bear_score}", f"{100-bull_pct}% bearish")
    m3.metric("ADX", f"{adx_v:.1f}", "Trending" if adx_trending else "Range")

    # 🎯 Trade Plan
    if entry and not is_range:
        st.markdown(f"""
        | Level | Price |
        |-------|-------|
        | Entry | {symbol}{entry:.2f} |
        | Stop Loss | {symbol}{sl:.2f} |
        | Target 1 | {symbol}{target:.2f} |
        | Target 2 | {symbol}{target2:.2f} |
        | R:R Ratio | 1:{rr} |
        """)

    elif is_range:
        st.markdown(f"""
        **📦 Range Trade Levels**
        | Level | Price |
        |-------|-------|
        | Support | {symbol}{last['Support']:.2f} |
        | Resistance | {symbol}{last['Resistance']:.2f} |
        | Mid | {symbol}{((last['Support'] + last['Resistance']) / 2):.2f} |
        """)
        st.info("💡 Buy support, sell resistance. Avoid mid-zone.")


with col_b:

    st.markdown("**Signal Breakdown**")

    if range_evidence:
        with st.expander(f"📦 Range Evidence ({len(range_evidence)}/3 signals triggered)", expanded=True):
            for ev in range_evidence:
                st.warning(f"• {ev}")

    for s in signals:
        text, stype = s if isinstance(s, tuple) else (s, "neutral")

        if stype == "bull":
            st.success(text)
        elif stype == "bear":
            st.error(text)
        else:
            st.warning(text)

# ── CONFLICT RESOLUTION SUMMARY ──
st.markdown("---")
st.subheader("⚖️ Signal Conflict Analysis")

tech_direction = "BULLISH" if bull_score > bear_score else ("BEARISH" if bear_score > bull_score else "NEUTRAL")
trend_strength = "STRONG" if adx_strong else ("MODERATE" if adx_trending else "WEAK/NONE")

col_c1, col_c2, col_c3, col_c4 = st.columns(4)
col_c1.metric("Technical Bias",   tech_direction)
col_c2.metric("Trend Strength",   trend_strength, f"ADX {adx_v:.1f}")
col_c3.metric("Market Mode",      "RANGE" if is_range_market else ("WEAK TREND" if is_weak_trend else "TRENDING"))
col_c4.metric("Tradeable?",       "⚠️ CAUTION" if (is_range_market or is_weak_trend) else "✅ YES")

timeframe_data = {
    "Timeframe / Factor": ["Primary Trend (EMA200, Stack)", "Short-Term Momentum (RSI, MACD)", "Volume Flow (OBV, CMF, MFI)", "Trend Strength (ADX)", "Volatility (BB, Squeeze)", "Net Assessment"],
    "Reading": [
        "Bearish" if price_cur < ema200_v and ema20_v < ema200_v else "Bullish",
        "Bearish" if rsi_v < 50 and macd_v < macd_sig else ("Bullish" if rsi_v > 50 and macd_v > macd_sig else "Mixed"),
        "Bullish (accumulation)" if obv_slope > 0 and cmf_v > 0 else ("Bearish" if obv_slope < 0 and cmf_v < 0 else "Mixed"),
        f"{'Strong trend' if adx_strong else ('Weak trend' if adx_trending else 'No trend')} ({adx_v:.1f})",
        "Squeeze (range likely)" if bb_squeeze else ("Wide bands (volatile)" if bb_width_now > bb_width_avg * 1.3 else "Normal"),
        final_signal
    ],
    "Action Implication": [
        "Avoid longs, short bias",
        "Weak selling / possible bounce forming",
        "Smart money buying — watch for reversal",
        "⚠️ Weak ADX = unreliable trend signals" if not adx_trending else "Trend signals more reliable",
        "Range trade: fade extremes" if bb_squeeze else "Breakout may be coming",
        "Range-bound: buy support, sell resistance" if is_range_market else final_signal
    ]
}

conflict_df = pd.DataFrame(timeframe_data)
st.dataframe(conflict_df, use_container_width=True, hide_index=True)

# ── PRO INTERPRETATION NOTE ──
with st.expander("🎓 Pro Interpretation — What This Market Is Saying"):
    if is_range_market:
        st.markdown(f"""
### 📦 Range-Bound Market Detected

Your indicators are giving **conflicting signals** because the market is NOT trending — it's consolidating.

**What you're likely seeing:**
- Technical engine says SELL/BUY with high score
- But ADX = {adx_v:.1f} (very weak trend) + BB squeeze + EMA compression = RANGE

**The correct trade approach right now:**
1. **DO NOT** aggressively short or long based on trend signals
2. **Buy near support** ({symbol}{last['Support']:.2f}) with tight stop
3. **Sell near resistance** ({symbol}{last['Resistance']:.2f}) with tight stop
4. **Wait for breakout confirmation:** ADX rises above 20, price closes cleanly above/below range

**When to switch to trend-following:**
- ADX crosses above 20 with rising slope
- Price breaks out of BB with volume expansion
- OBV confirms the breakout direction
        """)
    elif is_weak_trend:
        st.markdown(f"""
### ⚠️ Weak Trend Environment

A directional bias exists but **trend strength is insufficient** for high-confidence trend trades.

- ADX = {adx_v:.1f} — below 20 threshold
- Technical bias = {tech_direction}
- Best approach: **smaller position size**, wider stops, take profits earlier
        """)
    else:
        st.markdown(f"""
### ✅ Trending Market

ADX = {adx_v:.1f} confirms a real trend is in place. Signal: **{final_signal}**

- Trend-following strategies are appropriate
- Use ATR-based stops as calculated above
- Respect the R:R ratio
        """)

# ---------------------------
# 📊 INDICATOR SNAPSHOT TABLE
# ---------------------------
st.subheader("📊 Indicator Snapshot")

snap_data = {
    "Indicator": ["RSI (14)", "MACD", "Stoch K/D", "Williams %R", "CCI (20)",
                  "MFI (14)", "CMF (20)", "ADX (14)", "ATR (14)", "BB %B",
                  "Supertrend", "PSAR"],
    "Value": [
        f"{rsi_v:.2f}",
        f"{macd_v:.4f} / Signal: {macd_sig:.4f}",
        f"{last['StochK']:.2f} / {last['StochD']:.2f}",
        f"{will_r:.2f}",
        f"{cci_v:.2f}",
        f"{mfi_v:.2f}",
        f"{cmf_v:.4f}",
        f"{adx_v:.2f}",
        f"{atr_v:.2f}",
        f"{bb_pct:.2f}",
        "Bullish" if st_dir == 1 else "Bearish",
        "Above" if price_cur > psar_v else "Below"
    ],
    "Signal": [
        "Overbought" if rsi_v > 70 else ("Oversold" if rsi_v < 30 else ("Bullish" if rsi_v > 50 else "Bearish")),
        "Bullish" if macd_v > macd_sig else "Bearish",
        "Bullish" if last['StochK'] > last['StochD'] else "Bearish",
        "Overbought" if will_r > -20 else ("Oversold" if will_r < -80 else "Neutral"),
        "Overbought" if cci_v > 100 else ("Oversold" if cci_v < -100 else "Neutral"),
        "Overbought" if mfi_v > 80 else ("Oversold" if mfi_v < 20 else ("Bullish" if mfi_v > 50 else "Bearish")),
        "Buying" if cmf_v > 0 else "Selling",
        "Strong trend" if adx_v > 25 else ("Moderate" if adx_v > 20 else ("Weak" if adx_v > 15 else "No trend / Range")),
        "High Vol" if atr_v > df["True_ATR"].mean() else "Low Vol",
        "Near Upper" if bb_pct > 0.8 else ("Near Lower" if bb_pct < 0.2 else "Middle"),
        "Bullish" if st_dir == 1 else "Bearish",
        "Bullish" if price_cur > psar_v else "Bearish"
    ]
}

snap_df = pd.DataFrame(snap_data)
st.dataframe(snap_df, use_container_width=True, hide_index=True)
# ---------------------------
# 💼 PORTFOLIO TRACKER
# ---------------------------
st.subheader("💼 Portfolio Tracker")

col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    p_ticker = st.text_input("Portfolio Ticker", ticker)
with col_p2:
    qty = st.number_input("Quantity", value=10, min_value=0)
with col_p3:
    buy = st.number_input("Buy Price", value=float(price_cur) if isinstance(price_cur, (int,float)) else 2500.0, min_value=0.0)

try:
    hist_port = yf.Ticker(p_ticker).history(period=period)

    if hist_port.empty:
        st.warning("No price data available for this ticker")
    else:
        df_port = hist_port.dropna(subset=["Close"]).copy()

        if df_port.empty:
            st.warning("No valid price data")
        else:
            df_port["Portfolio Value"] = df_port["Close"] * qty
            df_port["Invested"]        = qty * buy

            current  = df_port["Portfolio Value"].iloc[-1]
            invested = qty * buy
            pnl      = current - invested
            pnl_pct  = (pnl / invested) * 100 if invested != 0 else 0

            df_port["Peak"]     = df_port["Portfolio Value"].cummax()
            df_port["Drawdown"] = (df_port["Portfolio Value"] - df_port["Peak"]) / df_port["Peak"] * 100
            max_dd   = df_port["Drawdown"].min()
            best_day = df_port["Close"].pct_change().max() * 100
            worst_day= df_port["Close"].pct_change().min() * 100

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Invested",  f"{symbol}{invested:,.2f}")
            c2.metric("Current",   f"{symbol}{current:,.2f}")
            c3.metric("P&L",       f"{symbol}{pnl:,.2f}", f"{pnl_pct:.2f}%")
            c4.metric("Max Drawdown", f"{max_dd:.2f}%")
            c5.metric("Best Day",  f"{best_day:.2f}%")

            st.markdown("### 📈 Portfolio Performance")
            color_line = "green" if current > invested else "red"

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_port.index, y=df_port["Portfolio Value"],
                                      name="Portfolio Value", line=dict(color=color_line, width=3)))
            fig2.add_trace(go.Scatter(x=df_port.index, y=df_port["Invested"],
                                      name="Invested", line=dict(dash="dash", width=2, color="#888")))
            fig2.update_layout(template="plotly_dark", height=380,
                               margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("🧠 Portfolio Insight")

            if pnl_pct < -30:
                st.error("⚠️ Heavy Loss Zone — Risk very high. Consider reviewing position sizing.")
            elif pnl_pct < -10:
                st.warning("⚠️ Drawdown — Review your position. Stop-loss may have been missed.")
            elif pnl_pct > 50:
                st.success("🚀 Exceptional gain — Consider booking significant partial profits.")
            elif pnl_pct > 20:
                st.success("✅ Strong profit — Consider booking partial gains or trailing stop.")
            else:
                st.info("ℹ️ Neutral zone — No strong action needed. Monitor closely.")

            st.subheader("📉 Drawdown Chart")
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df_port.index, y=df_port["Drawdown"],
                                      fill="tozeroy", name="Drawdown %",
                                      line=dict(color="#FF4500"), fillcolor="rgba(255,69,0,0.2)"))
            fig3.update_layout(template="plotly_dark", height=250,
                               margin=dict(l=10, r=10, t=20, b=10),
                               yaxis_title="Drawdown %")
            st.plotly_chart(fig3, use_container_width=True)

except Exception as e:
    st.warning(f"Portfolio error: {e}")

# ---------------------------
# 📉 PRICE TREND
# ---------------------------
st.subheader("📉 Price Trend (Close)")
st.line_chart(hist["Close"])

# ---------------------------
# 📥 DOWNLOAD
# ---------------------------
st.subheader("📥 Download Data")

csv_all = df.to_csv().encode("utf-8")
st.download_button("📥 Download Full Data (with Indicators)", csv_all, "stock_data_full.csv", "text/csv")

raw_csv = hist.to_csv().encode("utf-8")
st.download_button("📥 Download Raw Price Data", raw_csv, "stock_data_raw.csv", "text/csv")

# ---------------------------
# 📰 NEWS
# ---------------------------
st.subheader("📰 Latest News")

query = ticker.replace(".NS", "").replace(".BO", "").replace("-USD", "")
url   = f"https://news.google.com/rss/search?q={query}+stock&hl=en-IN&gl=IN&ceid=IN:en"

feed = feedparser.parse(url)

if feed.entries:
    for entry in feed.entries[:7]:
        st.markdown(f"**{entry.title}**")
        if hasattr(entry, "published"):
            st.caption(entry.published)
        st.markdown(f"[Read more →]({entry.link})")
        st.divider()
else:
    st.info("No news found for this ticker")

# ---------------------------
# ℹ️ DISCLAIMERS
# ---------------------------
with st.expander("📖 How the Signal Engine Works (v2 — Fixed)"):
    st.markdown("""
### Multi-Factor Scoring System (v2)

The signal is generated by scoring **12 independent factors** across 4 categories:

| Category | Indicators | Max Score |
|----------|-----------|-----------|
| **Trend** | Price vs EMA200, EMA Stack, Supertrend, PSAR | 6 |
| **Momentum** | RSI, MACD, Williams %R | 4 |
| **Volume/Flow** | CMF, OBV, MFI | 3 |
| **Volatility** | ADX | Contextual |

### ADX Gate (NEW in v2)
Raw score signals are now filtered through an **ADX gate**:

| Condition | What happens |
|-----------|-------------|
| ADX < 15 + BB squeeze + EMA compression | → Override to **RANGE-BOUND** |
| ADX 15–20 | → Downgrade STRONG signals to normal |
| ADX > 20 | → Trust the directional signal |
| ADX > 25 | → Full confidence in trend direction |

### ML Reliability Guard (NEW in v2)
ML predictions are now tagged with accuracy:

- **Accuracy < 50%**: Model flagged as UNRELIABLE — prediction discarded
- **Accuracy 50–53%**: Low confidence warning shown
- **Accuracy > 53%**: Prediction shown normally

### Range Detection (NEW in v2)
Range market is confirmed when **2 of 3** signals trigger:
1. ADX < 15 (no trend strength)
2. BB bandwidth < 70% of rolling average (squeeze)
3. EMA 20/50 spread < 1.5% (EMAs converging)

**Signal Thresholds (unchanged):**
- STRONG BUY → Bull Score ≥ 9 + ADX > 20
- BUY → Bull Score ≥ 6
- STRONG SELL → Bear Score ≥ 9 + ADX > 20
- SELL → Bear Score ≥ 6
- RANGE-BOUND → 2/3 range signals triggered (overrides above)
- WAIT → No clear majority

**Trade Plan (ATR-based):**
- Trend trades: Stop Loss = Entry ± 1.5× ATR, Target = Entry ± 2.5× ATR
- Range trades: Buy support, sell resistance with 1× ATR stops
    """)

st.warning("""
⚠️ **Disclaimer**: All signals are rule-based technical indicators and NOT financial advice.
Markets are unpredictable. Past performance does not guarantee future results.
Always apply your own analysis, risk management, and position sizing.
Never risk capital you cannot afford to lose.
""")
