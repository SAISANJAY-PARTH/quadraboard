import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import urllib.parse  # ✅ FIX 1: Moved to top — was imported AFTER use causing NameError
import urllib.request
import json
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

df["EMA9"]   = df["Close"].ewm(span=9, adjust=False).mean()
df["EMA20"]  = df["Close"].ewm(span=20, adjust=False).mean()
df["EMA44"]  = df["Close"].ewm(span=44, adjust=False).mean()
df["EMA50"]  = df["Close"].ewm(span=50, adjust=False).mean()
df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
df["SMA50"]  = df["Close"].rolling(50).mean()
df["SMA200"] = df["Close"].rolling(200).mean()

# ── RSI ──
delta = df["Close"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
rs    = gain / (loss + 1e-10)
df["RSI"] = 100 - (100 / (1 + rs))

# ── Stoch RSI ──
rsi_min = df["RSI"].rolling(14).min()
rsi_max = df["RSI"].rolling(14).max()
df["StochRSI"] = (df["RSI"] - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
df["StochK"]   = df["StochRSI"].rolling(3).mean()
df["StochD"]   = df["StochK"].rolling(3).mean()

# ── MACD ──
df["MACD"]        = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

# ── True ATR (Wilder's) ──
prev_close = df["Close"].shift(1)
tr1 = df["High"] - df["Low"]
tr2 = (df["High"] - prev_close).abs()
tr3 = (df["Low"]  - prev_close).abs()
df["True_Range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df["True_ATR"]   = df["True_Range"].ewm(span=14, adjust=False).mean()   # ✅ FIX 2: Wilder's smoothing = EWM span 14
df["ATR"]        = df["True_ATR"]  # alias

# ── ADX (Wilder's — FIX 3: was producing NaN due to wrong rolling logic) ──
df["+DM"] = df["High"].diff()
df["-DM"] = -df["Low"].diff()
df["+DM"] = df["+DM"].where((df["+DM"] > df["-DM"]) & (df["+DM"] > 0), 0.0)
df["-DM"] = df["-DM"].where((df["-DM"] > df["+DM"]) & (df["-DM"] > 0), 0.0)

# Wilder smoothing (EWM) — NOT simple rolling sum
atr14      = df["True_ATR"]
plus_di14  = df["+DM"].ewm(span=14, adjust=False).mean()
minus_di14 = df["-DM"].ewm(span=14, adjust=False).mean()
df["Plus_DI"]  = 100 * (plus_di14  / (atr14 + 1e-10))
df["Minus_DI"] = 100 * (minus_di14 / (atr14 + 1e-10))
dx             = (abs(df["Plus_DI"] - df["Minus_DI"]) / (df["Plus_DI"] + df["Minus_DI"] + 1e-10)) * 100
df["ADX"]      = dx.ewm(span=14, adjust=False).mean()

# ── VWAP (session-based reset would be ideal but cumsum is acceptable for multi-day) ──
df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / (df["Volume"].cumsum() + 1e-10)

# ── Bollinger Bands ──
df["BB_Mid"]   = df["Close"].rolling(20).mean()
df["BB_STD"]   = df["Close"].rolling(20).std()
df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_STD"]
df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_STD"]
df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["BB_Mid"] + 1e-10)
df["BB_Pct"]   = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-10)

# ── Keltner Channel ──
df["KC_Mid"]   = df["Close"].ewm(span=20, adjust=False).mean()
df["KC_Upper"] = df["KC_Mid"] + 2 * df["True_ATR"]
df["KC_Lower"] = df["KC_Mid"] - 2 * df["True_ATR"]
df["Squeeze"]  = df["BB_Width"] < (df["KC_Upper"] - df["KC_Lower"]) / (df["KC_Mid"] + 1e-10)

# ── Support / Resistance / Pivots ──
df["Support"]    = df["Low"].rolling(20).min()
df["Resistance"] = df["High"].rolling(20).max()
df["Pivot"]      = (df["High"] + df["Low"] + df["Close"]) / 3
df["R1"]         = 2 * df["Pivot"] - df["Low"]
df["S1"]         = 2 * df["Pivot"] - df["High"]
df["R2"]         = df["Pivot"] + (df["High"] - df["Low"])
df["S2"]         = df["Pivot"] - (df["High"] - df["Low"])

# ── OBV ──
obv = [0]
for i in range(1, len(df)):
    if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
        obv.append(obv[-1] + df["Volume"].iloc[i])
    elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
        obv.append(obv[-1] - df["Volume"].iloc[i])
    else:
        obv.append(obv[-1])
df["OBV"] = obv

# ── CMF ──
mfv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-10) * df["Volume"]
df["CMF"] = mfv.rolling(20).sum() / (df["Volume"].rolling(20).sum() + 1e-10)

# ── Williams %R ──
df["WilliamsR"] = -100 * (df["High"].rolling(14).max() - df["Close"]) / (
    df["High"].rolling(14).max() - df["Low"].rolling(14).min() + 1e-10)

# ── CCI ──
tp = (df["High"] + df["Low"] + df["Close"]) / 3
df["CCI"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-10)

# ── Supertrend ──
multiplier = 3
df["BasicUpper"] = (df["High"] + df["Low"]) / 2 + multiplier * df["True_ATR"]
df["BasicLower"] = (df["High"] + df["Low"]) / 2 - multiplier * df["True_ATR"]
supertrend = df["BasicLower"].copy()
st_dir_list = [1] * len(df)
for i in range(1, len(df)):
    if df["Close"].iloc[i] > df["BasicUpper"].iloc[i-1]:
        st_dir_list[i] = 1
    elif df["Close"].iloc[i] < df["BasicLower"].iloc[i-1]:
        st_dir_list[i] = -1
    else:
        st_dir_list[i] = st_dir_list[i-1]
df["Supertrend"] = supertrend
df["ST_Dir"]     = st_dir_list

# ── Heikin Ashi ──
df["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
df["HA_Open"]  = ((df["Open"].shift(1) + df["Close"].shift(1)) / 2).fillna(df["Open"])
df["HA_High"]  = df[["High", "HA_Open", "HA_Close"]].max(axis=1)
df["HA_Low"]   = df[["Low",  "HA_Open", "HA_Close"]].min(axis=1)

# ── ROC ──
df["ROC"] = df["Close"].pct_change(10) * 100

# ── MFI ──
tp2    = (df["High"] + df["Low"] + df["Close"]) / 3
mf     = tp2 * df["Volume"]
pos_mf = mf.where(tp2 > tp2.shift(1), 0.0).rolling(14).sum()
neg_mf = mf.where(tp2 < tp2.shift(1), 0.0).rolling(14).sum()
df["MFI"] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))

# ── Parabolic SAR ──
af, max_af = 0.02, 0.20
sar   = float(df["Low"].iloc[0])
trend = 1
ep    = float(df["High"].iloc[0])
psar  = []
for i in range(len(df)):
    psar.append(sar)
    if trend == 1:
        sar = sar + af * (ep - sar)
        sar = min(sar, float(df["Low"].iloc[max(0, i-1)]), float(df["Low"].iloc[max(0, i-2)]))
        if float(df["Low"].iloc[i]) < sar:
            trend, sar, ep, af = -1, ep, float(df["Low"].iloc[i]), 0.02
        else:
            if float(df["High"].iloc[i]) > ep:
                ep = float(df["High"].iloc[i])
                af = min(af + 0.02, max_af)
    else:
        sar = sar + af * (ep - sar)
        sar = max(sar, float(df["High"].iloc[max(0, i-1)]), float(df["High"].iloc[max(0, i-2)]))
        if float(df["High"].iloc[i]) > sar:
            trend, sar, ep, af = 1, ep, float(df["High"].iloc[i]), 0.02
        else:
            if float(df["Low"].iloc[i]) < ep:
                ep = float(df["Low"].iloc[i])
                af = min(af + 0.02, max_af)
df["PSAR"] = psar

# ─────────────────────────────────────────────
# 🧠 DECISION ENGINE  (all variables computed FIRST, then display)
# ─────────────────────────────────────────────

last      = df.iloc[-1]
price_cur = float(last["Close"])
ema20_v   = float(last["EMA20"])
ema50_v   = float(last["EMA50"])
ema200_v  = float(last["EMA200"])
rsi_v     = float(last["RSI"])   if not pd.isna(last["RSI"])   else 50.0
macd_v    = float(last["MACD"])  if not pd.isna(last["MACD"])  else 0.0
macd_sig  = float(last["MACD_Signal"]) if not pd.isna(last["MACD_Signal"]) else 0.0
adx_v     = float(last["ADX"])   if not pd.isna(last["ADX"])   else 0.0
atr_v     = float(last["True_ATR"]) if not pd.isna(last["True_ATR"]) else 0.0
bb_pct    = float(last["BB_Pct"]) if not pd.isna(last["BB_Pct"]) else 0.5
cmf_v     = float(last["CMF"])   if not pd.isna(last["CMF"])   else 0.0
cci_v     = float(last["CCI"])   if not pd.isna(last["CCI"])   else 0.0
mfi_v     = float(last["MFI"])   if not pd.isna(last["MFI"])   else 50.0
will_r    = float(last["WilliamsR"]) if not pd.isna(last["WilliamsR"]) else -50.0
st_dir    = int(last["ST_Dir"])
psar_v    = float(last["PSAR"])
obv_slope = float(df["OBV"].iloc[-1] - df["OBV"].iloc[-5]) if len(df) > 5 else 0.0

# ── Candle structure ──
upper_wick = float(last["High"] - max(last["Close"], last["Open"]))
lower_wick = float(min(last["Close"], last["Open"]) - last["Low"])
body       = float(abs(last["Close"] - last["Open"]))
bearish_rejection = upper_wick > body * 1.5 and last["Close"] < last["Open"]
bullish_rejection = lower_wick > body * 1.5 and last["Close"] > last["Open"]

# ── Market Structure ──
adx_trending  = adx_v > 20
adx_strong    = adx_v > 25
adx_very_weak = adx_v < 15
adx_slope     = float(df["ADX"].iloc[-1] - df["ADX"].iloc[-5]) if len(df) > 5 and not pd.isna(df["ADX"].iloc[-5]) else 0.0
adx_rising    = adx_slope > 0

bb_width_now  = float(last["BB_Width"]) if not pd.isna(last["BB_Width"]) else 0.0
bb_width_series = df["BB_Width"].rolling(50).mean()
bb_width_avg  = float(bb_width_series.iloc[-1]) if not pd.isna(bb_width_series.iloc[-1]) else bb_width_now
bb_squeeze    = bb_width_now < bb_width_avg * 0.7

ema_spread_pct = abs(ema20_v - ema50_v) / (ema50_v + 1e-10) * 100
ema_compressed = ema_spread_pct < 1.5

# ✅ FIX 4: pullback_active — corrected logic (was always False or checking wrong condition)
# A pullback in a downtrend = price bouncing UP inside a bearish structure
pullback_active = (
    price_cur < ema200_v and          # primary downtrend
    price_cur > ema20_v  and          # short-term bounce above EMA20
    rsi_v > 50           and          # momentum temporarily up
    macd_v > macd_sig                 # MACD showing short-term bullish momentum
)

# ── Range Evidence ──
range_evidence = []
if adx_very_weak:
    range_evidence.append(f"ADX={adx_v:.1f} (< 15, no trend)")
if ema_compressed:
    range_evidence.append(f"EMA spread={ema_spread_pct:.1f}% (< 1.5%, EMAs converging)")
if bb_squeeze:
    range_evidence.append(f"BB squeeze (bands tighter than avg {bb_width_avg*100:.2f}%)")

is_range_market = len(range_evidence) >= 2
is_weak_trend   = not adx_trending

# ── Strategy Mode ──
if is_range_market:
    strategy_mode = "MEAN REVERSION"
elif adx_v > 20 and adx_rising:
    strategy_mode = "TREND FOLLOWING"
else:
    strategy_mode = "NO TRADE"

mid_price = (float(last["Support"]) + float(last["Resistance"])) / 2
mid_range = abs(price_cur - mid_price) < (atr_v * 1.2)

# ── Scoring System ──
bull_score = 0
bear_score = 0
signals    = []

# Trend (max 6)
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

# Momentum (max 4)
# ✅ FIX 5: RSI oversold was missing bear score — now symmetric
if rsi_v > 70:
    bear_score += 1; signals.append((f"⚠️ RSI Overbought ({rsi_v:.1f}) — potential reversal", "neutral"))
elif 50 < rsi_v <= 70:
    bull_score += 2; signals.append((f"✅ RSI bullish zone ({rsi_v:.1f})", "bull"))
elif 30 <= rsi_v < 50:
    bear_score += 2; signals.append((f"❌ RSI bearish zone ({rsi_v:.1f})", "bear"))
else:  # rsi_v < 30
    bull_score += 1; signals.append((f"⚠️ RSI Oversold ({rsi_v:.1f}) — potential bounce", "neutral"))

if macd_v > macd_sig:
    bull_score += 1; signals.append(("✅ MACD bullish crossover", "bull"))
else:
    bear_score += 1; signals.append(("❌ MACD bearish crossover", "bear"))

if -20 > will_r > -50:
    bull_score += 1; signals.append((f"✅ Williams %R neutral-bull ({will_r:.1f})", "bull"))
elif will_r > -20:
    bear_score += 1; signals.append((f"⚠️ Williams %R Overbought ({will_r:.1f})", "neutral"))
elif will_r < -80:
    bull_score += 1; signals.append((f"⚠️ Williams %R Oversold ({will_r:.1f})", "neutral"))
else:
    bear_score += 1; signals.append((f"❌ Williams %R bearish zone ({will_r:.1f})", "bear"))

# Volume / Flow (max 3)
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

# Volatility context (informational)
if adx_strong:
    signals.append((f"✅ ADX strong trend ({adx_v:.1f}) — trend-follow OK",
                    "bull" if bull_score > bear_score else "bear"))
elif adx_trending:
    signals.append((f"⚠️ ADX moderate ({adx_v:.1f}) — light trend present", "neutral"))
elif adx_very_weak:
    signals.append((f"⚠️ ADX very weak ({adx_v:.1f}) — choppy/range market", "neutral"))
else:
    signals.append((f"⚠️ ADX weak ({adx_v:.1f}) — range market, avoid trend signals", "neutral"))

# ── Raw Signal ──
total    = bull_score + bear_score
bull_pct = int((bull_score / max(total, 1)) * 100)

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

# ✅ FIX 6: ALL decision logic runs BEFORE any display code
final_signal    = raw_signal
signal_override = None
entry_warning   = None
rr              = 2.5
entry = sl = target = target2 = None

# 1. Structure Filter
if strategy_mode == "NO TRADE":
    final_signal    = "WAIT"
    signal_override = "⚠️ No clear structure (ADX too weak, not enough trend). Skip trades."

# 2. Range Logic
elif is_range_market:
    if mid_range:
        final_signal    = "WAIT"
        signal_override = "⚠️ Mid-range. No edge. Avoid trades."
    else:
        final_signal = "RANGE"
        if raw_signal in ["BUY", "STRONG BUY"]:
            signal_override = "📦 Range: Signals near SUPPORT — buy only with tight stop."
        elif raw_signal in ["SELL", "STRONG SELL"]:
            signal_override = "📦 Range: Signals near RESISTANCE — sell only with tight stop."
        else:
            signal_override = "📦 Sideways market. Wait for extremes."

# 2.5 Pullback Detection (bearish trend + bullish bounce = wait for continuation short)
elif pullback_active:
    final_signal    = "PULLBACK"
    signal_override = "⚠️ Bearish trend + bullish pullback active. Wait for short entry on rejection."

# 3. Weak Trend Downgrade
elif is_weak_trend:
    if raw_signal == "STRONG BUY":
        final_signal    = "BUY"
        signal_override = f"⚠️ Weak trend (ADX={adx_v:.1f}). Downgraded STRONG BUY → BUY."
    elif raw_signal == "STRONG SELL":
        final_signal    = "SELL"
        signal_override = f"⚠️ Weak trend (ADX={adx_v:.1f}). Downgraded STRONG SELL → SELL."

# 4. Momentum Conflict Filter
momentum_conflict_bear = (rsi_v > 60 and macd_v > macd_sig)
momentum_conflict_bull = (rsi_v < 40 and macd_v < macd_sig)

if "SELL" in final_signal and momentum_conflict_bear:
    final_signal    = "WAIT"
    signal_override = "⚠️ Bearish price structure but momentum indicators still bullish. Wait for momentum to turn."

if "BUY" in final_signal and momentum_conflict_bull:
    final_signal    = "WAIT"
    signal_override = "⚠️ Bullish price structure but momentum is weak/bearish. Wait for momentum confirmation."

# 5. ADX Quality Gate
if adx_v < 18:
    final_signal    = "WAIT"
    signal_override = f"⚠️ ADX={adx_v:.1f} — Insufficient trend strength. No reliable signal."
elif adx_v < 22 and "STRONG" in final_signal:
    final_signal    = final_signal.replace("STRONG ", "")
    signal_override = f"⚠️ ADX={adx_v:.1f} — Borderline trend. Strength downgraded."

# 6. Entry Confirmation (soft warning only)
if "SELL" in final_signal and not bearish_rejection:
    entry_warning = "⚠️ No strong bearish candle confirmation on latest bar — consider waiting one more candle"
if "BUY" in final_signal and not bullish_rejection:
    entry_warning = "⚠️ No strong bullish candle confirmation on latest bar — consider waiting one more candle"

# 7. Trade Plan Levels (ATR-based)
is_range = "RANGE" in final_signal
is_buy   = "BUY"  in final_signal
is_sell  = "SELL" in final_signal

if is_buy and not is_range and atr_v > 0:
    entry   = price_cur
    sl      = price_cur - (atr_v * 1.5)
    target  = price_cur + (atr_v * rr)
    target2 = price_cur + (atr_v * rr * 1.6)
elif is_sell and not is_range and atr_v > 0:
    entry   = price_cur
    sl      = price_cur + (atr_v * 1.5)
    target  = price_cur - (atr_v * rr)
    target2 = price_cur - (atr_v * rr * 1.6)

# ─────────────────────────────────────────────
# 📈 ADVANCED CHART
# ─────────────────────────────────────────────
st.subheader("📈 Advanced Chart")

selected = st.multiselect(
    "Select Indicators",
    ["EMA", "SMA", "VWAP", "Bollinger Bands", "Keltner Channel",
     "Support/Resistance", "Pivot Points", "Supertrend", "PSAR",
     "RSI", "Stoch RSI", "MACD", "Williams %R", "CCI", "MFI",
     "Volume", "OBV", "CMF", "ATR", "ADX", "ROC"],
    default=["EMA", "RSI", "MACD"]
)

lower_panels = []
if any(x in selected for x in ["RSI", "Stoch RSI", "Williams %R", "MFI"]):
    lower_panels.append("momentum")
if "MACD" in selected:
    lower_panels.append("macd")
if any(x in selected for x in ["Volume", "OBV", "CMF"]):
    lower_panels.append("volume")
if any(x in selected for x in ["ATR", "ADX", "ROC", "CCI"]):
    lower_panels.append("misc")

n_rows       = 1 + len(lower_panels)
row_heights  = [0.55] + [round(0.45 / max(len(lower_panels), 1), 2)] * len(lower_panels)

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

ema_colors = {"EMA9": "#FF6B6B", "EMA20": "#FFD93D", "EMA50": "#6BCB77", "EMA200": "#4D96FF"}
if "EMA" in selected:
    for col_name, color in ema_colors.items():
        if len(df) >= int(col_name.replace("EMA", "")):
            fig.add_trace(go.Scatter(x=df.index, y=df[col_name], name=col_name,
                                     line=dict(color=color, width=1.2)), row=1, col=1)

if "SMA" in selected:
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"],  name="SMA50",
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
    fig.add_trace(go.Scatter(x=df.index, y=df["Support"],    name="Support",
                             line=dict(color="#00FF7F", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Resistance"], name="Resistance",
                             line=dict(color="#FF4500", dash="dot")), row=1, col=1)

if "Pivot Points" in selected:
    for lvl, color in [("Pivot","#FFD700"),("R1","#FF6347"),("S1","#7CFC00"),("R2","#FF0000"),("S2","#00FA9A")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[lvl], name=lvl,
                                 line=dict(color=color, width=1, dash="dot")), row=1, col=1)

if "Supertrend" in selected:
    bull_st = df["Supertrend"].where(pd.Series(st_dir_list, index=df.index) == 1)
    bear_st = df["Supertrend"].where(pd.Series(st_dir_list, index=df.index) == -1)
    fig.add_trace(go.Scatter(x=df.index, y=bull_st, name="ST Bull",
                             line=dict(color="#00FF7F", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bear_st, name="ST Bear",
                             line=dict(color="#FF4500", width=2)), row=1, col=1)

if "PSAR" in selected:
    fig.add_trace(go.Scatter(x=df.index, y=df["PSAR"], name="PSAR",
                             mode="markers", marker=dict(size=3, color="#FFD700")), row=1, col=1)

if "momentum" in panel:
    r = panel["momentum"]
    if "RSI" in selected:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                                 line=dict(color="#FFD93D")), row=r, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",   row=r, col=1)
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
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],        name="MACD",
                             line=dict(color="#00d4ff")), row=r, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                             line=dict(color="#FF6B6B")), row=r, col=1)
    colors_hist = ["#00FF7F" if v >= 0 else "#FF4500" for v in df["MACD_Hist"].fillna(0)]
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
        fig.add_trace(go.Scatter(x=df.index, y=df["ADX"],       name="ADX",
                                 line=dict(color="#4D96FF")), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Plus_DI"],   name="+DI",
                                 line=dict(color="#00FF7F", dash="dot")), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Minus_DI"],  name="-DI",
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

def get_currency_symbol(stock_obj, tkr):
    try:
        info     = stock_obj.info
        currency = info.get("currency", "")
        mapping  = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
        return mapping.get(currency, currency + " ")
    except:
        if tkr.endswith(".NS") or tkr.endswith(".BO"):
            return "₹"
        return "$"

symbol = get_currency_symbol(stock, ticker)

def format_price(x):
    if isinstance(x, (int, float)) and not np.isnan(x):
        return f"{symbol} {x:,.2f}"
    return "N/A"

price   = fast.get("lastPrice")
high    = fast.get("dayHigh")
low     = fast.get("dayLow")
prev    = fast.get("previousClose")
mkt_cap = fast.get("marketCap")
volume  = fast.get("lastVolume")

if isinstance(price, (int, float)) and isinstance(prev, (int, float)):
    change     = price - prev
    change_pct = (change / (prev + 1e-10)) * 100
    delta_str  = f"{change:+.2f} ({change_pct:+.2f}%)"
else:
    delta_str = None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price",      format_price(price), delta_str)
c2.metric("Day High",   format_price(high))
c3.metric("Day Low",    format_price(low))
c4.metric("Prev Close", format_price(prev))
if isinstance(volume, (int, float)):
    c5.metric("Volume", f"{volume:,.0f}")

if isinstance(mkt_cap, (int, float)):
    st.caption(f"Market Cap: {symbol} {mkt_cap:,.0f}")

# ─────────────────────────────────────────────
# 🧠 DECISION ENGINE DISPLAY
# ─────────────────────────────────────────────
st.subheader("🧠 Multi-Factor Decision Engine")

col_a, col_b = st.columns(2)

with col_a:
    st.caption(f"🧭 Strategy Mode: **{strategy_mode}**")

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
        st.warning("🔄 PULLBACK — Downtrend intact, short-term bounce active. Wait for continuation entry.")
    else:
        st.warning("⚠️ WAIT — No Clear Setup")

    if entry_warning:
        st.warning(entry_warning)

    if signal_override:
        st.markdown(f"""
        <div style="background:#1a1f2e; border-left:4px solid #FFD700; border-radius:6px; padding:10px 14px; margin-top:8px;">
        <b>🔔 Signal Override / Note</b><br><small>{signal_override}</small>
        </div>
        """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Bull Score", f"{bull_score}", f"{bull_pct}% bullish")
    m2.metric("Bear Score", f"{bear_score}", f"{100 - bull_pct}% bearish")
    m3.metric("ADX",        f"{adx_v:.1f}",  "Trending" if adx_trending else "Range/Weak")

    if entry is not None and not is_range:
        st.markdown(f"""
| Level     | Price |
|-----------|-------|
| Entry     | {symbol}{entry:.2f} |
| Stop Loss | {symbol}{sl:.2f} |
| Target 1  | {symbol}{target:.2f} |
| Target 2  | {symbol}{target2:.2f} |
| R:R Ratio | 1:{rr} |
        """)
    elif is_range:
        st.markdown(f"""
**📦 Range Trade Levels**

| Level      | Price |
|------------|-------|
| Support    | {symbol}{float(last['Support']):.2f} |
| Resistance | {symbol}{float(last['Resistance']):.2f} |
| Mid Zone   | {symbol}{mid_price:.2f} |
        """)
        st.info("💡 Buy near support, sell near resistance. Avoid the mid-zone.")

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

# ── Conflict Analysis ──
st.markdown("---")
st.subheader("⚖️ Signal Conflict Analysis")

tech_direction = ("BULLISH" if bull_score > bear_score
                  else ("BEARISH" if bear_score > bull_score else "NEUTRAL"))
trend_strength = "STRONG" if adx_strong else ("MODERATE" if adx_trending else "WEAK/NONE")

col_c1, col_c2, col_c3, col_c4 = st.columns(4)
col_c1.metric("Technical Bias", tech_direction)
col_c2.metric("Trend Strength", trend_strength, f"ADX {adx_v:.1f}")
col_c3.metric("Market Mode",    "RANGE" if is_range_market else ("WEAK TREND" if is_weak_trend else "TRENDING"))
col_c4.metric("Tradeable?",     "⚠️ CAUTION" if (is_range_market or is_weak_trend) else "✅ YES")

primary_trend_read  = "Bearish" if (price_cur < ema200_v and ema20_v < ema200_v) else "Bullish"
momentum_read       = ("Bullish" if (rsi_v > 50 and macd_v > macd_sig)
                       else ("Bearish" if (rsi_v < 50 and macd_v < macd_sig) else "Mixed"))
flow_read           = ("Bullish (accumulation)" if (obv_slope > 0 and cmf_v > 0)
                       else ("Bearish (distribution)" if (obv_slope < 0 and cmf_v < 0) else "Mixed"))

timeframe_data = {
    "Factor": ["Primary Trend (EMA200 + Stack)", "Short-Term Momentum (RSI, MACD)",
                "Volume Flow (OBV, CMF, MFI)", "Trend Strength (ADX)",
                "Volatility (BB Width / Squeeze)", "Net Assessment"],
    "Reading": [
        primary_trend_read,
        momentum_read,
        flow_read,
        f"{'Strong' if adx_strong else ('Moderate' if adx_trending else 'Weak/None')} ({adx_v:.1f})",
        "Squeeze — range likely" if bb_squeeze else ("Wide bands — volatile" if bb_width_now > bb_width_avg * 1.3 else "Normal"),
        final_signal
    ],
    "Action Implication": [
        "Short bias / avoid longs" if primary_trend_read == "Bearish" else "Long bias",
        "Momentum confirms direction" if momentum_read != "Mixed" else "Wait for momentum alignment",
        "Smart money positioning" if flow_read != "Mixed" else "Indeterminate flow",
        "⚠️ Weak — signals less reliable" if not adx_trending else "Signals more reliable",
        "Fade extremes" if bb_squeeze else "Breakout potential",
        "Range: buy support / sell resistance" if is_range_market else final_signal
    ]
}

st.dataframe(pd.DataFrame(timeframe_data), use_container_width=True, hide_index=True)

with st.expander("🎓 Pro Interpretation — What This Market Is Saying"):
    if is_range_market:
        st.markdown(f"""
### 📦 Range-Bound Market Detected

Indicators are giving **conflicting signals** because the market is NOT trending — it's consolidating.

**Range evidence active ({len(range_evidence)}/3):**
{chr(10).join('- ' + e for e in range_evidence)}

**Correct trade approach:**
1. **DO NOT** aggressively trend-trade — signals are unreliable in sideways markets
2. **Buy near support** ({symbol}{float(last['Support']):.2f}) with a stop below
3. **Sell near resistance** ({symbol}{float(last['Resistance']):.2f}) with a stop above
4. **Watch for breakout:** ADX > 20 + rising + price closes with volume outside the range

**Breakout confirmation checklist:**
- ADX crosses above 20 with rising slope ✓
- BB bands expand (squeeze releasing) ✓
- OBV breaks in breakout direction ✓
        """)
    elif is_weak_trend:
        st.markdown(f"""
### ⚠️ Weak Trend Environment

Directional bias exists but **trend strength is insufficient** for high-confidence trend trades.

- ADX = {adx_v:.1f} (below 20 threshold) — trend signals are less reliable
- Technical bias = **{tech_direction}**
- Recommended: smaller position size, wider stops, take profits earlier than normal
        """)
    else:
        st.markdown(f"""
### ✅ Trending Market

ADX = {adx_v:.1f} confirms a real trend is in place.

**Signal: {final_signal}**

- Trend-following strategies are appropriate
- Use ATR-based stops (as calculated in the trade plan above)
- Respect the R:R ratio of 1:{rr}
- Trail stop as price moves in your favour
        """)

# ---------------------------
# 📊 INDICATOR SNAPSHOT TABLE
# ---------------------------
st.subheader("📊 Indicator Snapshot")

snap_data = {
    "Indicator": ["RSI (14)", "MACD", "Stoch K / D", "Williams %R", "CCI (20)",
                  "MFI (14)", "CMF (20)", "ADX (14)", "ATR (14)", "BB %B",
                  "Supertrend", "PSAR"],
    "Value": [
        f"{rsi_v:.2f}",
        f"MACD {macd_v:.4f}  |  Signal {macd_sig:.4f}",
        f"{float(last['StochK']):.2f} / {float(last['StochD']):.2f}",
        f"{will_r:.2f}",
        f"{cci_v:.2f}",
        f"{mfi_v:.2f}",
        f"{cmf_v:.4f}",
        f"{adx_v:.2f}",
        f"{atr_v:.4f}",
        f"{bb_pct:.2f}",
        "Bullish ▲" if st_dir == 1 else "Bearish ▼",
        f"{'Above ▲' if price_cur > psar_v else 'Below ▼'}  ({symbol}{psar_v:.2f})"
    ],
    "Signal": [
        "Overbought 🔴" if rsi_v > 70 else ("Oversold 🟢" if rsi_v < 30 else ("Bullish" if rsi_v > 50 else "Bearish")),
        "Bullish ✅"    if macd_v > macd_sig else "Bearish ❌",
        "Bullish ✅"    if float(last['StochK']) > float(last['StochD']) else "Bearish ❌",
        "Overbought 🔴" if will_r > -20 else ("Oversold 🟢" if will_r < -80 else "Neutral"),
        "Overbought 🔴" if cci_v > 100   else ("Oversold 🟢" if cci_v < -100 else "Neutral"),
        "Overbought 🔴" if mfi_v > 80    else ("Oversold 🟢" if mfi_v < 20    else ("Bullish" if mfi_v > 50 else "Bearish")),
        "Buying ✅"     if cmf_v > 0     else "Selling ❌",
        "Strong Trend ✅" if adx_v > 25  else ("Moderate" if adx_v > 20 else ("Weak" if adx_v > 15 else "No Trend / Range ⚠️")),
        "High Volatility ⚠️" if atr_v > df["True_ATR"].mean() else "Low Volatility",
        "Near Upper Band 🔴" if bb_pct > 0.8 else ("Near Lower Band 🟢" if bb_pct < 0.2 else "Middle"),
        "Bullish ✅" if st_dir == 1 else "Bearish ❌",
        "Bullish ✅" if price_cur > psar_v else "Bearish ❌"
    ]
}

st.dataframe(pd.DataFrame(snap_data), use_container_width=True, hide_index=True)

# ---------------------------
# 💼 PORTFOLIO TRACKER
# ---------------------------
st.subheader("💼 Portfolio Tracker")

# ✅ FIX 7: Default buy price uses last close price (was causing ₹nan)
default_price = float(price_cur) if isinstance(price_cur, float) and not np.isnan(price_cur) else 100.0

col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    p_ticker = st.text_input("Portfolio Ticker", ticker)
with col_p2:
    qty = st.number_input("Quantity", value=10, min_value=0)
with col_p3:
    buy = st.number_input("Buy Price", value=default_price, min_value=0.01, format="%.2f")

try:
    hist_port = yf.Ticker(p_ticker).history(period=period)

    if hist_port.empty:
        st.warning("No price data available for this ticker.")
    else:
        df_port = hist_port.dropna(subset=["Close"]).copy()

        if df_port.empty:
            st.warning("No valid price data.")
        else:
            df_port["Portfolio Value"] = df_port["Close"] * qty
            df_port["Invested"]        = qty * buy

            current  = float(df_port["Portfolio Value"].iloc[-1])
            invested = qty * buy
            pnl      = current - invested
            pnl_pct  = (pnl / (invested + 1e-10)) * 100

            df_port["Peak"]     = df_port["Portfolio Value"].cummax()
            df_port["Drawdown"] = (df_port["Portfolio Value"] - df_port["Peak"]) / (df_port["Peak"] + 1e-10) * 100
            max_dd    = float(df_port["Drawdown"].min())
            best_day  = float(df_port["Close"].pct_change().max() * 100)
            worst_day = float(df_port["Close"].pct_change().min() * 100)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Invested",     f"{symbol}{invested:,.2f}")
            c2.metric("Current",      f"{symbol}{current:,.2f}")
            c3.metric("P&L",          f"{symbol}{pnl:,.2f}", f"{pnl_pct:.2f}%")
            c4.metric("Max Drawdown", f"{max_dd:.2f}%")
            c5.metric("Best Day",     f"{best_day:.2f}%")

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
                st.error("⚠️ Heavy Loss Zone — Risk very high. Review position sizing.")
            elif pnl_pct < -10:
                st.warning("⚠️ In drawdown — Review position. Stop-loss may have been missed.")
            elif pnl_pct > 50:
                st.success("🚀 Exceptional gain — Consider booking significant partial profits.")
            elif pnl_pct > 20:
                st.success("✅ Strong profit — Consider partial profit booking or trailing stop.")
            else:
                st.info("ℹ️ Neutral zone — No strong action needed. Monitor closely.")

            st.subheader("📉 Drawdown Chart")
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df_port.index, y=df_port["Drawdown"],
                                      fill="tozeroy", name="Drawdown %",
                                      line=dict(color="#FF4500"),
                                      fillcolor="rgba(255,69,0,0.2)"))
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
# 📰 LATEST NEWS
# ---------------------------
st.subheader("📰 Latest News")

query = ticker.replace(".NS", "").replace(".BO", "").replace("-USD", "")
url   = f"https://news.google.com/rss/search?q={query}+stock&hl=en-IN&gl=IN&ceid=IN:en"
feed  = feedparser.parse(url)

if feed.entries:
    for entry in feed.entries[:7]:
        st.markdown(f"**{entry.title}**")
        if hasattr(entry, "published"):
            st.caption(entry.published)
        st.markdown(f"[Read more →]({entry.link})")
        st.divider()
else:
    st.info("No news found for this ticker.")

# ---------------------------
# ℹ️ METHODOLOGY
# ---------------------------
with st.expander("📖 How the Signal Engine Works (v3 — Fixed)"):
    st.markdown("""
### Multi-Factor Scoring System (v3 — Fixed)

Signals are generated by scoring **13 independent factors** across 4 categories:

| Category | Indicators | Max Score |
|---|---|---|
| **Trend** | Price vs EMA200, EMA Stack (20>50>200), Supertrend, PSAR | 6 |
| **Momentum** | RSI, MACD, Williams %R | 4 |
| **Volume / Flow** | CMF, OBV, MFI | 3 |
| **Volatility context** | ADX (gate + contextual) | — |

### Fixes Applied in v3
| # | Bug | Fix |
|---|---|---|
| 1 | `urllib.parse` imported after use → `NameError` | Moved to top of file |
| 2 | ATR used `rolling().mean()` instead of Wilder's smoothing | Switched to `ewm(span=14)` |
| 3 | ADX always returned `NaN` (used rolling sum incorrectly) | Rewritten with Wilder's EWM smoothing |
| 4 | `pullback_active` condition was logically inverted | Fixed to: bearish trend + short-term bounce |
| 5 | RSI oversold gave no `bear_score`, RSI overbought gave no score at all | Made scoring symmetric |
| 6 | Entry/SL/Target block ran before `final_signal` was computed | Moved to after all decision logic |
| 7 | Portfolio buy price defaulted to `NaN` when price was unavailable | Now defaults to last close |

### Signal Thresholds
| Signal | Condition |
|---|---|
| STRONG BUY | Bull Score ≥ 9 |
| BUY | Bull Score ≥ 6 |
| STRONG SELL | Bear Score ≥ 9 |
| SELL | Bear Score ≥ 6 |
| RANGE | 2/3 range signals triggered |
| WAIT | No majority OR ADX < 18 OR momentum conflict |

### Trade Plan (ATR-based)
- Stop Loss = Entry ± 1.5× ATR
- Target 1  = Entry ± 2.5× ATR
- Target 2  = Entry ± 4.0× ATR
    """)

st.warning("""
⚠️ **Disclaimer**: All signals are rule-based technical indicators and NOT financial advice.
Markets are unpredictable. Past performance does not guarantee future results.
Always apply your own analysis, risk management, and position sizing.
Never risk capital you cannot afford to lose.
""")
