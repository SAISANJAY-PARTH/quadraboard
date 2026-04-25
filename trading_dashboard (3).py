import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
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
    """Search Yahoo Finance for tickers matching query string."""
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

# Run search
search_results = []
ticker = st.session_state.get("selected_ticker", "RELIANCE.NS")

if search_query and len(search_query) >= 2:
    with st.spinner("Searching..."):
        search_results = search_tickers(search_query)

if search_results:
    labels  = [r["label"] for r in search_results]
    symbols = [r["symbol"] for r in search_results]

    # Find default index (keep previously selected if still in results)
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

    # Extract symbol from selection
    selected_idx    = labels.index(selected_label)
    ticker          = symbols[selected_idx]
    selected_info   = search_results[selected_idx]

    st.session_state["selected_ticker"] = ticker
    st.session_state["last_search"]     = search_query

    # Show badge
    badge_color = {
        "EQUITY": "🟦", "ETF": "🟩", "CRYPTOCURRENCY": "🟡",
        "INDEX": "🟣", "MUTUALFUND": "🟠"
    }.get(selected_info["type"].upper(), "⬜")

    st.success(f"{badge_color} **Selected:** {selected_info['name']}  |  Symbol: `{ticker}`  |  Exchange: {selected_info['exchange']}  |  Type: {selected_info['type']}")

elif search_query and len(search_query) >= 2:
    st.warning("No results found. Try a different name.")
    # Fall back to manual entry
    ticker = st.text_input("Or enter ticker manually", value=ticker).upper()
else:
    st.info("Start typing a company name above to search.")
    ticker = st.session_state.get("selected_ticker", "RELIANCE.NS")

# ── Time Period + Chart Type ──
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

fast = stock.fast_info  # DO NOT CHANGE

# ---------------------------
# INDICATORS ENGINE
# ---------------------------
df = hist.copy()

# ── EMA ──
df["EMA9"]   = df["Close"].ewm(span=9).mean()
df["EMA20"]  = df["Close"].ewm(span=20).mean()
df["EMA44"]  = df["Close"].ewm(span=44).mean()
df["EMA50"]  = df["Close"].ewm(span=50).mean()
df["EMA200"] = df["Close"].ewm(span=200).mean()

# ── SMA ──
df["SMA50"]  = df["Close"].rolling(50).mean()
df["SMA200"] = df["Close"].rolling(200).mean()

# ── RSI (14) ──
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# ── Stochastic RSI ──
rsi_min = df["RSI"].rolling(14).min()
rsi_max = df["RSI"].rolling(14).max()
df["StochRSI"] = (df["RSI"] - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
df["StochK"] = df["StochRSI"].rolling(3).mean()
df["StochD"] = df["StochK"].rolling(3).mean()

# ── MACD ──
df["MACD"]        = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

# ── ATR ──
df["TR"] = df["High"] - df["Low"]
df["ATR"] = df["TR"].rolling(14).mean()

# ── True ATR (with gaps) ──
prev_close = df["Close"].shift(1)
tr1 = df["High"] - df["Low"]
tr2 = (df["High"] - prev_close).abs()
tr3 = (df["Low"] - prev_close).abs()
df["True_ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

# ── ADX ──
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

# ── VWAP ──
df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

# ── Bollinger Bands ──
df["BB_Mid"]   = df["Close"].rolling(20).mean()
df["BB_STD"]   = df["Close"].rolling(20).std()
df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_STD"]
df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_STD"]
df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
df["BB_Pct"]   = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-10)

# ── Keltner Channels ──
df["KC_Mid"]   = df["Close"].ewm(span=20).mean()
df["KC_Upper"] = df["KC_Mid"] + 2 * df["True_ATR"]
df["KC_Lower"] = df["KC_Mid"] - 2 * df["True_ATR"]

# ── Squeeze Momentum (Lazybear) ──
df["Squeeze"] = df["BB_Width"] < (df["KC_Upper"] - df["KC_Lower"]) / df["KC_Mid"]

# ── Support / Resistance (rolling) ──
df["Support"]    = df["Low"].rolling(20).min()
df["Resistance"] = df["High"].rolling(20).max()

# ── Pivot Points (Classic) ──
df["Pivot"] = (df["High"] + df["Low"] + df["Close"]) / 3
df["R1"]    = 2 * df["Pivot"] - df["Low"]
df["S1"]    = 2 * df["Pivot"] - df["High"]
df["R2"]    = df["Pivot"] + (df["High"] - df["Low"])
df["S2"]    = df["Pivot"] - (df["High"] - df["Low"])

# ── OBV (On-Balance Volume) ──
obv = [0]
for i in range(1, len(df)):
    if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
        obv.append(obv[-1] + df["Volume"].iloc[i])
    elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
        obv.append(obv[-1] - df["Volume"].iloc[i])
    else:
        obv.append(obv[-1])
df["OBV"] = obv

# ── CMF (Chaikin Money Flow) ──
mfv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-10) * df["Volume"]
df["CMF"] = mfv.rolling(20).sum() / df["Volume"].rolling(20).sum()

# ── Williams %R ──
df["WilliamsR"] = -100 * (df["High"].rolling(14).max() - df["Close"]) / (df["High"].rolling(14).max() - df["Low"].rolling(14).min() + 1e-10)

# ── CCI (Commodity Channel Index) ──
tp = (df["High"] + df["Low"] + df["Close"]) / 3
df["CCI"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-10)

# ── Supertrend ──
multiplier = 3
df["BasicUpper"] = (df["High"] + df["Low"]) / 2 + multiplier * df["True_ATR"]
df["BasicLower"] = (df["High"] + df["Low"]) / 2 - multiplier * df["True_ATR"]
df["Supertrend"] = df["BasicLower"]
df["ST_Dir"] = 1  # 1=bullish, -1=bearish
for i in range(1, len(df)):
    if df["Close"].iloc[i] > df["BasicUpper"].iloc[i-1]:
        df.loc[df.index[i], "ST_Dir"] = 1
    elif df["Close"].iloc[i] < df["BasicLower"].iloc[i-1]:
        df.loc[df.index[i], "ST_Dir"] = -1
    else:
        df.loc[df.index[i], "ST_Dir"] = df["ST_Dir"].iloc[i-1]

# ── Heikin Ashi ──
df["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
df["HA_Open"]  = ((df["Open"].shift(1) + df["Close"].shift(1)) / 2).fillna(df["Open"])
df["HA_High"]  = df[["High", "HA_Open", "HA_Close"]].max(axis=1)
df["HA_Low"]   = df[["Low",  "HA_Open", "HA_Close"]].min(axis=1)

# ── ROC (Rate of Change) ──
df["ROC"] = df["Close"].pct_change(10) * 100

# ── MFI (Money Flow Index) ──
tp2 = (df["High"] + df["Low"] + df["Close"]) / 3
mf  = tp2 * df["Volume"]
pos_mf = mf.where(tp2 > tp2.shift(1), 0).rolling(14).sum()
neg_mf = mf.where(tp2 < tp2.shift(1), 0).rolling(14).sum()
df["MFI"] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))

# ── PSAR (Parabolic SAR) — simplified ──
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

# ---------------------------
# HELPER: VALUE LABEL
# ---------------------------
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

# ---------------------------
# BUILD SUBPLOT LAYOUT
# ---------------------------
# Determine how many subplots we need
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

# Panel index map
panel = {name: (i + 2) for i, name in enumerate(lower_panels)}

# ── Price ──
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

# ── EMA ──
ema_colors = {"EMA9":"#FF6B6B","EMA20":"#FFD93D","EMA50":"#6BCB77","EMA200":"#4D96FF"}
if "EMA" in selected:
    for col_name, color in ema_colors.items():
        if len(df) >= int(col_name.replace("EMA","")):
            fig.add_trace(go.Scatter(x=df.index, y=df[col_name], name=col_name,
                                      line=dict(color=color, width=1.2)), row=1, col=1)

# ── SMA ──
if "SMA" in selected:
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50",
                              line=dict(color="#FFA07A", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200",
                              line=dict(color="#20B2AA", dash="dot")), row=1, col=1)

# ── VWAP ──
if "VWAP" in selected:
    fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP",
                              line=dict(color="#DA70D6", dash="dash")), row=1, col=1)

# ── Bollinger ──
if "Bollinger Bands" in selected:
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                              line=dict(color="#888", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                              line=dict(color="#888", dash="dash"),
                              fill="tonexty", fillcolor="rgba(100,100,100,0.1)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Mid"], name="BB Mid",
                              line=dict(color="#aaa", width=1)), row=1, col=1)

# ── Keltner ──
if "Keltner Channel" in selected:
    fig.add_trace(go.Scatter(x=df.index, y=df["KC_Upper"], name="KC Upper",
                              line=dict(color="#FF8C00", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["KC_Lower"], name="KC Lower",
                              line=dict(color="#FF8C00", dash="dot")), row=1, col=1)

# ── Support / Resistance ──
if "Support/Resistance" in selected:
    fig.add_trace(go.Scatter(x=df.index, y=df["Support"], name="Support",
                              line=dict(color="#00FF7F", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Resistance"], name="Resistance",
                              line=dict(color="#FF4500", dash="dot")), row=1, col=1)

# ── Pivot Points ──
if "Pivot Points" in selected:
    for lvl, color in [("Pivot","#FFD700"),("R1","#FF6347"),("S1","#7CFC00"),("R2","#FF0000"),("S2","#00FA9A")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[lvl], name=lvl,
                                  line=dict(color=color, width=1, dash="dot")), row=1, col=1)

# ── Supertrend ──
if "Supertrend" in selected:
    bull = df["Supertrend"].where(df["ST_Dir"] == 1)
    bear = df["Supertrend"].where(df["ST_Dir"] == -1)
    fig.add_trace(go.Scatter(x=df.index, y=bull, name="ST Bull",
                              line=dict(color="#00FF7F", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bear, name="ST Bear",
                              line=dict(color="#FF4500", width=2)), row=1, col=1)

# ── PSAR ──
if "PSAR" in selected:
    fig.add_trace(go.Scatter(x=df.index, y=df["PSAR"], name="PSAR",
                              mode="markers", marker=dict(size=3, color="#FFD700")), row=1, col=1)

# ── RSI / Stoch / WilliamsR / MFI → momentum panel ──
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

# ── MACD ──
if "macd" in panel:
    r = panel["macd"]
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                              line=dict(color="#00d4ff")), row=r, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                              line=dict(color="#FF6B6B")), row=r, col=1)
    colors_hist = ["#00FF7F" if v >= 0 else "#FF4500" for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram",
                          marker_color=colors_hist), row=r, col=1)

# ── Volume / OBV / CMF ──
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

# ── ATR / ADX / ROC / CCI ──
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

# ---------------------------
# 🧠 MULTI-FACTOR DECISION ENGINE
# ---------------------------
st.subheader("🧠 Multi-Factor Decision Engine")

last = df.iloc[-1]
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
obv_slope = (df["OBV"].iloc[-1] - df["OBV"].iloc[-5]) / (df["OBV"].iloc[-5] + 1e-10) if len(df) > 5 else 0
cci_v     = last["CCI"] if not pd.isna(last["CCI"]) else 0
mfi_v     = last["MFI"] if not pd.isna(last["MFI"]) else 50
will_r    = last["WilliamsR"] if not pd.isna(last["WilliamsR"]) else -50
st_dir    = last["ST_Dir"]
psar_v    = last["PSAR"]

# ── Scoring System ──
bull_score = 0
bear_score = 0
signals    = []

# Trend (weight 3)
if price_cur > ema200_v:
    bull_score += 2; signals.append(("✅ Price > EMA200 (Uptrend)", "bull"))
else:
    bear_score += 2; signals.append(("❌ Price < EMA200 (Downtrend)", "bear"))

if ema20_v > ema50_v > ema200_v:
    bull_score += 2; signals.append(("✅ EMA stack bullish (20>50>200)", "bull"))
elif ema20_v < ema50_v < ema200_v:
    bear_score += 2; signals.append(("❌ EMA stack bearish (20<50<200)", "bear"))

if st_dir == 1:
    bull_score += 1; signals.append(("✅ Supertrend Bullish", "bull"))
else:
    bear_score += 1; signals.append(("❌ Supertrend Bearish", "bear"))

if price_cur > psar_v:
    bull_score += 1; signals.append(("✅ Price above PSAR", "bull"))
else:
    bear_score += 1; signals.append(("❌ Price below PSAR", "bear"))

# Momentum (weight 2)
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
    bull_score += 1; signals.append(f"✅ Williams %R neutral-bull ({will_r:.1f})")
elif will_r > -20:
    signals.append((f"⚠️ Williams %R Overbought", "neutral"))

# Volume / Flow
if cmf_v > 0.05:
    bull_score += 1; signals.append(("✅ CMF positive (buying pressure)", "bull"))
elif cmf_v < -0.05:
    bear_score += 1; signals.append(("❌ CMF negative (selling pressure)", "bear"))

if obv_slope > 0:
    bull_score += 1; signals.append(("✅ OBV rising", "bull"))
else:
    bear_score += 1; signals.append(("❌ OBV falling", "bear"))

if mfi_v > 60:
    bull_score += 1; signals.append((f"✅ MFI bullish ({mfi_v:.1f})", "bull"))
elif mfi_v < 40:
    bear_score += 1; signals.append((f"❌ MFI bearish ({mfi_v:.1f})", "bear"))

# Volatility context
if adx_v > 25:
    signals.append((f"✅ ADX strong trend ({adx_v:.1f})", "bull"))
else:
    signals.append((f"⚠️ ADX weak trend ({adx_v:.1f}) — range market", "neutral"))

total = bull_score + bear_score
bull_pct = int((bull_score / max(total, 1)) * 100)

# ── Signal ──
if bull_score >= 9 and adx_v > 20:
    final_signal = "STRONG BUY"
elif bull_score >= 6:
    final_signal = "BUY"
elif bear_score >= 9 and adx_v > 20:
    final_signal = "STRONG SELL"
elif bear_score >= 6:
    final_signal = "SELL"
else:
    final_signal = "WAIT"

# ── Trade Plan ──
entry, sl, target, target2 = None, None, None, None
rr = 2.5

if "BUY" in final_signal:
    entry   = price_cur
    sl      = price_cur - (atr_v * 1.5)
    target  = price_cur + (atr_v * rr)
    target2 = price_cur + (atr_v * rr * 1.6)
elif "SELL" in final_signal:
    entry   = price_cur
    sl      = price_cur + (atr_v * 1.5)
    target  = price_cur - (atr_v * rr)
    target2 = price_cur - (atr_v * rr * 1.6)

# ── Display ──
col_a, col_b = st.columns([1, 2])

with col_a:
    if "STRONG BUY" == final_signal:
        st.success(f"🚀 {final_signal}")
    elif "BUY" == final_signal:
        st.success(f"🟢 {final_signal}")
    elif "STRONG SELL" == final_signal:
        st.error(f"🔻 {final_signal}")
    elif "SELL" == final_signal:
        st.error(f"🔴 {final_signal}")
    else:
        st.warning("⚠️ WAIT — No Clear Setup")

    st.metric("Bull Score", f"{bull_score}", f"{bull_pct}% bullish")
    st.metric("Bear Score", f"{bear_score}")

    if entry:
        st.markdown(f"""
| Level | Price |
|-------|-------|
| Entry | {symbol}{entry:.2f} |
| Stop Loss | {symbol}{sl:.2f} |
| Target 1 | {symbol}{target:.2f} |
| Target 2 | {symbol}{target2:.2f} |
| R:R Ratio | 1:{rr} |
        """)

with col_b:
    st.markdown("**Signal Breakdown**")
    for s in signals:
        if isinstance(s, tuple):
            text, stype = s
        else:
            text, stype = s, "neutral"
        if stype == "bull":
            st.success(text)
        elif stype == "bear":
            st.error(text)
        else:
            st.warning(text)

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
        "Strong" if adx_v > 25 else "Weak",
        "High Vol" if atr_v > df["True_ATR"].mean() else "Low Vol",
        "Near Upper" if bb_pct > 0.8 else ("Near Lower" if bb_pct < 0.2 else "Middle"),
        "Bullish" if st_dir == 1 else "Bearish",
        "Bullish" if price_cur > psar_v else "Bearish"
    ]
}

snap_df = pd.DataFrame(snap_data)
st.dataframe(snap_df, use_container_width=True, hide_index=True)

# ---------------------------
# 🧠 MARKET REGIME (ML)
# ---------------------------
st.subheader("🧠 Market Regime (ML Cluster)")

try:
    df_ml = df.copy()
    df_ml["returns"]    = df_ml["Close"].pct_change()
    df_ml["volatility"] = df_ml["returns"].rolling(10).std()
    df_ml["momentum"]   = df_ml["returns"].rolling(5).mean()
    df_ml["trend"]      = (df_ml["Close"] - df_ml["Close"].rolling(20).mean()) / (df_ml["Close"].rolling(20).std() + 1e-10)
    df_ml = df_ml.dropna()

    if len(df_ml) > 30:
        features = df_ml[["returns", "volatility", "momentum", "trend"]]
        scaler   = StandardScaler()
        X        = scaler.fit_transform(features)

        model = KMeans(n_clusters=3, n_init=10, random_state=42)
        df_ml["regime"] = model.fit_predict(X)

        vol_rank = df_ml.groupby("regime")["volatility"].mean().sort_values()
        mapping  = {vol_rank.index[0]:"Stable 🟢", vol_rank.index[1]:"Moderate 🟡", vol_rank.index[2]:"Volatile 🔴"}

        current_regime = mapping[df_ml["regime"].iloc[-1]]
        regime_counts  = df_ml["regime"].value_counts()

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Current Regime", current_regime)

        regime_returns = df_ml.groupby("regime")["returns"].mean()
        avg_ret        = regime_returns.get(df_ml["regime"].iloc[-1], 0)
        col_r2.metric("Avg Daily Return (regime)", f"{avg_ret*100:.3f}%")

        regime_vol = df_ml.groupby("regime")["volatility"].mean()
        avg_vol    = regime_vol.get(df_ml["regime"].iloc[-1], 0)
        col_r3.metric("Avg Volatility (regime)", f"{avg_vol*100:.3f}%")

        if "Stable" in current_regime:
            st.success("Stable market — Trend-following strategies favoured")
        elif "Moderate" in current_regime:
            st.warning("Moderate volatility — Use tighter stops")
        else:
            st.error("High volatility — Reduce position size, widen stops or stay out")

except Exception as e:
    st.warning(f"ML Regime detection failed: {e}")

# ---------------------------
# 🌳 DECISION TREE CLASSIFIER
# ---------------------------
st.subheader("🌳 Decision Tree — Next-Day Direction Predictor")

try:
    df_dt = df.copy()

    # ── Feature Engineering ──
    df_dt["returns"]       = df_dt["Close"].pct_change()
    df_dt["returns_lag1"]  = df_dt["returns"].shift(1)
    df_dt["returns_lag2"]  = df_dt["returns"].shift(2)
    df_dt["returns_lag3"]  = df_dt["returns"].shift(3)
    df_dt["vol10"]         = df_dt["returns"].rolling(10).std()
    df_dt["rsi_norm"]      = df_dt["RSI"] / 100
    df_dt["macd_norm"]     = df_dt["MACD"] / (df_dt["Close"] + 1e-10)
    df_dt["bb_pct"]        = df_dt["BB_Pct"]
    df_dt["adx_norm"]      = df_dt["ADX"] / 100
    df_dt["cmf"]           = df_dt["CMF"]
    df_dt["ema_ratio"]     = df_dt["EMA20"] / (df_dt["EMA50"] + 1e-10) - 1
    df_dt["ema200_ratio"]  = df_dt["Close"] / (df_dt["EMA200"] + 1e-10) - 1
    df_dt["cci_norm"]      = df_dt["CCI"] / 200
    df_dt["mfi_norm"]      = df_dt["MFI"] / 100
    df_dt["obv_chg"]       = df_dt["OBV"].pct_change(5)
    df_dt["atr_norm"]      = df_dt["True_ATR"] / (df_dt["Close"] + 1e-10)
    df_dt["will_r_norm"]   = (df_dt["WilliamsR"] + 100) / 100
    df_dt["stochk_norm"]   = df_dt["StochK"] / 100
    df_dt["psar_dist"]     = (df_dt["Close"] - df_dt["PSAR"]) / (df_dt["Close"] + 1e-10)
    df_dt["st_dir"]        = df_dt["ST_Dir"]

    # ── Label: 1 = next day Up, 0 = next day Down ──
    df_dt["target"] = (df_dt["Close"].shift(-1) > df_dt["Close"]).astype(int)

    FEATURES = [
        "returns_lag1", "returns_lag2", "returns_lag3", "vol10",
        "rsi_norm", "macd_norm", "bb_pct", "adx_norm", "cmf",
        "ema_ratio", "ema200_ratio", "cci_norm", "mfi_norm",
        "obv_chg", "atr_norm", "will_r_norm", "stochk_norm",
        "psar_dist", "st_dir"
    ]

    df_dt = df_dt[FEATURES + ["target"]].dropna()

    if len(df_dt) < 60:
        st.warning("Not enough data for Decision Tree (need 60+ candles)")
    else:
        X = df_dt[FEATURES]
        y = df_dt["target"]

        # Time-based split (no shuffling — respects time order)
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # ── What is Max Depth? ──
        st.markdown("""
---
#### 🎛️ Decision Tree Max Depth — What does this slider do?

A **Decision Tree** works like a flowchart of yes/no questions on your indicators:

```
Is RSI > 0.60?
 ├── YES → Is EMA20/EMA50 ratio > 1.01?
 │          ├── YES →  Predict: UP  📈
 │          └── NO  →  Predict: WAIT ⚠️
 └── NO  → Is MACD > Signal?
            ├── YES →  Predict: WAIT ⚠️
            └── NO  →  Predict: DOWN 📉
```

**Max Depth controls how many levels of questions** the tree is allowed to ask:

| Depth | Complexity | Risk | Best used when |
|-------|-----------|------|----------------|
| **2** | Very simple | Underfit | Short data (3mo), quick test |
| **3** | Simple | Low | 6mo data, conservative |
| **4** ✅ | Balanced | Medium | Recommended default (1y data) |
| **5–6** | Complex | High | 2y+ data only |
| **7–10** | Very deep | Overfit ⚠️ | Usually bad — memorizes noise |

> ⚠️ **Warning signs of overfitting:** Confidence = 100% but Test Accuracy ≈ 50–55%.
> This means the tree memorized training data but can't predict new data.
> **Fix: drag the slider LEFT to depth 2 or 3.**

> ✅ **Healthy model:** Confidence between 55–72%, Test Accuracy above 52%.
---
""")

        dt_depth = st.slider(
            "🎚️ Max Depth  (⬅ lower = simpler & safer  |  higher = complex & overfits ➡)",
            min_value=2, max_value=10, value=4, key="dt_depth"
        )

        # Live depth feedback
        if dt_depth <= 3:
            st.info(f"📐 Depth {dt_depth} — Simple tree. Lower chance of overfitting. Good for shorter datasets.")
        elif dt_depth <= 5:
            st.success(f"📐 Depth {dt_depth} — Balanced tree. Recommended for most datasets.")
        elif dt_depth <= 7:
            st.warning(f"📐 Depth {dt_depth} — Complex tree. Watch for 100% confidence (overfitting signal).")
        else:
            st.error(f"📐 Depth {dt_depth} — Very deep tree. High overfitting risk. Use only with 3y+ data.")

        dt_model = DecisionTreeClassifier(
            max_depth=dt_depth,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        )
        dt_model.fit(X_train, y_train)

        y_pred    = dt_model.predict(X_test)
        dt_acc    = accuracy_score(y_test, y_pred)
        dt_report = classification_report(y_test, y_pred, target_names=["Down","Up"], output_dict=True)

        # Current prediction
        latest_features = X.iloc[[-1]]
        dt_pred_label   = dt_model.predict(latest_features)[0]
        dt_pred_proba   = dt_model.predict_proba(latest_features)[0]
        dt_confidence   = max(dt_pred_proba) * 100

        # ── Display ──
        col_dt1, col_dt2, col_dt3 = st.columns(3)
        col_dt1.metric("🎯 Next-Day Prediction", "📈 UP" if dt_pred_label == 1 else "📉 DOWN")
        col_dt2.metric("Confidence", f"{dt_confidence:.1f}%")
        col_dt3.metric("Test Accuracy", f"{dt_acc*100:.1f}%")

        # ── Confidence health check ──
        if dt_confidence >= 95:
            st.error(f"🚨 Confidence {dt_confidence:.0f}% is suspiciously high — likely **overfitting**. Lower the depth slider to 2 or 3.")
        elif dt_confidence >= 80:
            st.warning(f"⚠️ Confidence {dt_confidence:.0f}% is high — possible overfitting. Try reducing depth.")
        elif dt_confidence >= 55:
            st.success(f"✅ Confidence {dt_confidence:.0f}% looks realistic — model is healthy.")
        else:
            st.info(f"ℹ️ Confidence {dt_confidence:.0f}% is low — market may be in a choppy/unpredictable phase.")

        if dt_pred_label == 1:
            st.success(f"🌳 Decision Tree says: **UP** with {dt_confidence:.1f}% confidence")
        else:
            st.error(f"🌳 Decision Tree says: **DOWN** with {dt_confidence:.1f}% confidence")

        # ── Class Report ──
        with st.expander("📋 Detailed Classification Report (Decision Tree)"):
            cr_df = pd.DataFrame(dt_report).T.round(3)
            st.dataframe(cr_df, use_container_width=True)

        # ── Feature Importance ──
        fi_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": dt_model.feature_importances_
        }).sort_values("Importance", ascending=False).head(10)

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"],
            y=fi_df["Feature"],
            orientation="h",
            marker_color="#4D96FF"
        ))
        fig_fi.update_layout(
            template="plotly_dark", height=360,
            title="Top 10 Feature Importances (Decision Tree)",
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Gini Importance",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # ── Tree Rules (text) ──
        with st.expander("🌿 View Decision Tree Rules (text)"):
            tree_rules = export_text(dt_model, feature_names=FEATURES, max_depth=3)
            st.code(tree_rules)

        # ── Rolling Accuracy chart ──
        roll_window = 20
        if len(y_test) >= roll_window:
            correct = (y_pred == y_test.values).astype(int)
            roll_acc = pd.Series(correct).rolling(roll_window).mean() * 100
            fig_roll = go.Figure(go.Scatter(
                y=roll_acc, mode="lines", name=f"{roll_window}-bar Rolling Accuracy",
                line=dict(color="#FFD93D", width=2)
            ))
            fig_roll.add_hline(y=50, line_dash="dash", line_color="gray")
            fig_roll.update_layout(
                template="plotly_dark", height=250,
                title=f"Rolling {roll_window}-Bar Accuracy (%)",
                yaxis_title="Accuracy %", margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_roll, use_container_width=True)

except Exception as e:
    st.warning(f"Decision Tree failed: {e}")

# ---------------------------
# 📈 LOGISTIC REGRESSION CLASSIFIER
# ---------------------------
st.subheader("📈 Logistic Regression — Next-Day Direction Predictor")

try:
    # Reuse df_dt features (already computed above)
    df_lr = df.copy()
    df_lr["returns"]       = df_lr["Close"].pct_change()
    df_lr["returns_lag1"]  = df_lr["returns"].shift(1)
    df_lr["returns_lag2"]  = df_lr["returns"].shift(2)
    df_lr["returns_lag3"]  = df_lr["returns"].shift(3)
    df_lr["vol10"]         = df_lr["returns"].rolling(10).std()
    df_lr["rsi_norm"]      = df_lr["RSI"] / 100
    df_lr["macd_norm"]     = df_lr["MACD"] / (df_lr["Close"] + 1e-10)
    df_lr["bb_pct"]        = df_lr["BB_Pct"]
    df_lr["adx_norm"]      = df_lr["ADX"] / 100
    df_lr["cmf"]           = df_lr["CMF"]
    df_lr["ema_ratio"]     = df_lr["EMA20"] / (df_lr["EMA50"] + 1e-10) - 1
    df_lr["ema200_ratio"]  = df_lr["Close"] / (df_lr["EMA200"] + 1e-10) - 1
    df_lr["cci_norm"]      = df_lr["CCI"] / 200
    df_lr["mfi_norm"]      = df_lr["MFI"] / 100
    df_lr["obv_chg"]       = df_lr["OBV"].pct_change(5)
    df_lr["atr_norm"]      = df_lr["True_ATR"] / (df_lr["Close"] + 1e-10)
    df_lr["will_r_norm"]   = (df_lr["WilliamsR"] + 100) / 100
    df_lr["stochk_norm"]   = df_lr["StochK"] / 100
    df_lr["psar_dist"]     = (df_lr["Close"] - df_lr["PSAR"]) / (df_lr["Close"] + 1e-10)
    df_lr["st_dir"]        = df_lr["ST_Dir"]
    df_lr["target"]        = (df_lr["Close"].shift(-1) > df_lr["Close"]).astype(int)

    df_lr = df_lr[FEATURES + ["target"]].dropna()

    if len(df_lr) < 60:
        st.warning("Not enough data for Logistic Regression (need 60+ candles)")
    else:
        X_lr = df_lr[FEATURES]
        y_lr = df_lr["target"]

        split_lr   = int(len(X_lr) * 0.8)
        X_tr, X_te = X_lr.iloc[:split_lr], X_lr.iloc[split_lr:]
        y_tr, y_te = y_lr.iloc[:split_lr], y_lr.iloc[split_lr:]

        # Scale for LR
        scaler_lr = StandardScaler()
        X_tr_s    = scaler_lr.fit_transform(X_tr)
        X_te_s    = scaler_lr.transform(X_te)

        lr_C = st.select_slider(
            "Logistic Regression Regularization (C) — higher = less regularization",
            options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            value=1.0,
            key="lr_C"
        )

        lr_model = LogisticRegression(C=lr_C, max_iter=1000, class_weight="balanced", random_state=42)
        lr_model.fit(X_tr_s, y_tr)

        y_pred_lr  = lr_model.predict(X_te_s)
        lr_acc     = accuracy_score(y_te, y_pred_lr)
        lr_report  = classification_report(y_te, y_pred_lr, target_names=["Down","Up"], output_dict=True)

        # Current prediction
        latest_lr_scaled = scaler_lr.transform(X_lr.iloc[[-1]])
        lr_pred_label    = lr_model.predict(latest_lr_scaled)[0]
        lr_pred_proba    = lr_model.predict_proba(latest_lr_scaled)[0]
        lr_confidence    = max(lr_pred_proba) * 100
        lr_up_prob       = lr_pred_proba[1] * 100  # probability of UP

        # ── Display ──
        col_lr1, col_lr2, col_lr3, col_lr4 = st.columns(4)
        col_lr1.metric("🎯 Next-Day Prediction", "📈 UP" if lr_pred_label == 1 else "📉 DOWN")
        col_lr2.metric("P(UP)", f"{lr_up_prob:.1f}%")
        col_lr3.metric("P(DOWN)", f"{100 - lr_up_prob:.1f}%")
        col_lr4.metric("Test Accuracy", f"{lr_acc*100:.1f}%")

        if lr_pred_label == 1:
            st.success(f"📈 Logistic Regression says: **UP** — P(Up)={lr_up_prob:.1f}%")
        else:
            st.error(f"📉 Logistic Regression says: **DOWN** — P(Down)={100-lr_up_prob:.1f}%")

        # ── Probability gauge bar ──
        fig_gauge = go.Figure(go.Bar(
            x=[lr_up_prob, 100 - lr_up_prob],
            y=["UP", "DOWN"],
            orientation="h",
            marker_color=["#00FF7F", "#FF4500"],
            text=[f"{lr_up_prob:.1f}%", f"{100-lr_up_prob:.1f}%"],
            textposition="inside"
        ))
        fig_gauge.update_layout(
            template="plotly_dark", height=180,
            title="Predicted Probability Distribution",
            xaxis_title="Probability (%)",
            margin=dict(l=10, r=10, t=40, b=10),
            barmode="stack"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Coefficient chart ──
        coef_df = pd.DataFrame({
            "Feature": FEATURES,
            "Coefficient": lr_model.coef_[0]
        }).sort_values("Coefficient", key=abs, ascending=False).head(12)

        coef_colors = ["#00FF7F" if v > 0 else "#FF4500" for v in coef_df["Coefficient"]]

        fig_coef = go.Figure(go.Bar(
            x=coef_df["Coefficient"],
            y=coef_df["Feature"],
            orientation="h",
            marker_color=coef_colors
        ))
        fig_coef.update_layout(
            template="plotly_dark", height=400,
            title="LR Coefficients — Green=Bullish, Red=Bearish",
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Coefficient",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_coef, use_container_width=True)

        # ── Class Report ──
        with st.expander("📋 Detailed Classification Report (Logistic Regression)"):
            lr_cr_df = pd.DataFrame(lr_report).T.round(3)
            st.dataframe(lr_cr_df, use_container_width=True)

        # ── Predicted probability over time ──
        all_scaled  = scaler_lr.transform(X_lr)
        all_proba   = lr_model.predict_proba(all_scaled)[:, 1] * 100
        fig_prob_ts = go.Figure()
        fig_prob_ts.add_trace(go.Scatter(
            x=df_lr.index, y=all_proba,
            name="P(Up) %", line=dict(color="#4D96FF", width=2)
        ))
        fig_prob_ts.add_hline(y=50, line_dash="dash", line_color="gray",
                               annotation_text="50% (neutral)")
        fig_prob_ts.update_layout(
            template="plotly_dark", height=280,
            title="LR Predicted P(UP) Over Time",
            yaxis_title="P(UP) %",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_prob_ts, use_container_width=True)

        # ── Combined ML Summary ──
        st.subheader("🤖 ML Model Consensus")

        dt_vote  = "UP" if dt_pred_label == 1 else "DOWN"
        lr_vote  = "UP" if lr_pred_label == 1 else "DOWN"
        votes_up = [dt_vote, lr_vote].count("UP")

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Decision Tree", f"{'📈' if dt_vote=='UP' else '📉'} {dt_vote}", f"{dt_confidence:.1f}% conf")
        col_m2.metric("Logistic Reg",  f"{'📈' if lr_vote=='UP' else '📉'} {lr_vote}",  f"{lr_confidence:.1f}% conf")
        col_m3.metric("ML Consensus",  f"{'🟢 UP' if votes_up >= 2 else ('🔴 DOWN' if votes_up == 0 else '🟡 SPLIT')}")

        if votes_up == 2:
            st.success("✅ Both ML models agree: **BULLISH** next day")
        elif votes_up == 0:
            st.error("❌ Both ML models agree: **BEARISH** next day")
        else:
            st.warning("⚠️ ML models disagree — Mixed signal, exercise caution")

        st.info("""
ℹ️ **How ML predictions work here:**
- Both models are trained on the last 80% of historical data and tested on the remaining 20%.
- Features include: RSI, MACD, EMA ratios, Bollinger %B, ADX, CMF, MFI, Williams %R, Stochastic, PSAR, Supertrend, OBV slope, ATR, and lagged returns.
- Label: whether the next candle closes **higher** (1=UP) or **lower** (0=DOWN).
- These are **in-sample trained models** and accuracy may decline on unseen data. Not financial advice.
        """)

except Exception as e:
    st.warning(f"Logistic Regression failed: {e}")

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

            # Max drawdown
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

            # ── Portfolio Insight ──
            st.subheader("🧠 Portfolio Insight")

            if pnl_pct < -30:
                st.error("⚠️ Heavy Loss Zone — Risk very high. Consider reviewing position sizing.")
            elif pnl_pct < -10:
                st.warning("⚠️ Drawdown — Review your position. Stop-loss may have been missed.")
            elif pnl_pct > 50:
                st.success("🚀 Exceptional gain — Strong consider booking significant partial profits.")
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
with st.expander("📖 How the Signal Engine Works"):
    st.markdown("""
### Multi-Factor Scoring System

The signal is generated by scoring **12 independent factors** across 4 categories:

| Category | Indicators | Max Score |
|----------|-----------|-----------|
| **Trend** | Price vs EMA200, EMA Stack, Supertrend, PSAR | 6 |
| **Momentum** | RSI, MACD, Williams %R | 4 |
| **Volume/Flow** | CMF, OBV, MFI | 3 |
| **Volatility** | ADX | Contextual |

**Signal Thresholds:**
- STRONG BUY → Bull Score ≥ 9 + ADX > 20
- BUY → Bull Score ≥ 6
- STRONG SELL → Bear Score ≥ 9 + ADX > 20
- SELL → Bear Score ≥ 6
- WAIT → No clear majority

**Trade Plan (ATR-based):**
- Stop Loss: Entry ± 1.5× ATR
- Target 1: Entry ± 2.5× ATR
- Target 2: Entry ± 4× ATR
    """)

st.warning("""
⚠️ **Disclaimer**: All signals are rule-based technical indicators and NOT financial advice.
Markets are unpredictable. Past performance does not guarantee future results.
Always apply your own analysis, risk management, and position sizing.
Never risk capital you cannot afford to lose.
""")
