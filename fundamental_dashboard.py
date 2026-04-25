import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import urllib.request, urllib.parse, json
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Fundamental Analysis", layout="wide", page_icon="📊")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── CARD ── */
.card {
    background: linear-gradient(160deg, #0d1117 0%, #111827 100%);
    padding: 22px 24px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    border-color: rgba(99,102,241,0.3);
}

.card-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 8px;
}
.card-value {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    color: #f9fafb;
    line-height: 1.1;
}
.card-sub {
    font-size: 12px;
    color: #4b5563;
    margin-top: 6px;
}
.card-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    margin-top: 8px;
}
.badge-green  { background: rgba(34,197,94,0.15);  color: #22c55e; }
.badge-red    { background: rgba(239,68,68,0.15);   color: #ef4444; }
.badge-yellow { background: rgba(250,204,21,0.15);  color: #facc15; }
.badge-blue   { background: rgba(99,102,241,0.15);  color: #818cf8; }

/* ── SCORE RING ── */
.score-wrap {
    display: flex; align-items: center; gap: 20px;
    background: linear-gradient(160deg,#0d1117,#111827);
    border-radius: 14px; border: 1px solid rgba(255,255,255,0.07);
    padding: 20px 24px;
}
.score-ring {
    width: 90px; height: 90px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'DM Serif Display', serif;
    font-size: 28px; font-weight: 700; flex-shrink: 0;
}
.score-text .s-label { font-size: 11px; letter-spacing: 1px; text-transform: uppercase; color: #6b7280; }
.score-text .s-title { font-size: 20px; font-weight: 600; color: #f9fafb; margin: 4px 0; }
.score-text .s-desc  { font-size: 13px; color: #9ca3af; }

/* ── INSIGHT PILL ── */
.insight {
    padding: 12px 16px; border-radius: 10px; margin-bottom: 8px;
    font-size: 13px; font-weight: 500;
    display: flex; align-items: flex-start; gap: 10px;
    border-left: 3px solid;
}
.insight-bull { background: rgba(34,197,94,0.08);  border-color: #22c55e; color: #dcfce7; }
.insight-bear { background: rgba(239,68,68,0.08);  border-color: #ef4444; color: #fee2e2; }
.insight-warn { background: rgba(250,204,21,0.08); border-color: #facc15; color: #fef9c3; }
.insight-info { background: rgba(99,102,241,0.08); border-color: #818cf8; color: #e0e7ff; }

hr.divider { border: none; border-top: 1px solid #1f2937; margin: 28px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# TICKER SEARCH
# ---------------------------
@st.cache_data(ttl=3600)
def search_tickers(query):
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(query)}&lang=en-US&quotesCount=12&newsCount=0"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        results = []
        for item in data.get("quotes", []):
            sym  = item.get("symbol", "")
            name = item.get("longname") or item.get("shortname") or sym
            exch = item.get("exchDisp", "")
            qt   = item.get("quoteType", "")
            if sym:
                results.append({"symbol": sym, "name": name, "exchange": exch,
                                 "type": qt, "label": f"{name}  [{sym}]  {exch}"})
        return results
    except:
        return []

st.title("📊 Fundamental Analysis Dashboard")
st.caption("Deep-dive financials, valuation, growth & health for any stock worldwide")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

search_q = st.text_input("🔍 Search Company Name", value=st.session_state.get("fa_search","Reliance"),
                          placeholder="e.g. Reliance, TCS, Apple, HDFC Bank...")
ticker = st.session_state.get("fa_ticker", "RELIANCE.NS")

if search_q and len(search_q) >= 2:
    results = search_tickers(search_q)
    if results:
        labels  = [r["label"]  for r in results]
        symbols = [r["symbol"] for r in results]
        def_idx = next((i for i,s in enumerate(symbols) if s == ticker), 0)
        chosen  = st.selectbox("Select Stock", labels, index=def_idx)
        idx     = labels.index(chosen)
        ticker  = symbols[idx]
        st.session_state["fa_ticker"] = ticker
        st.session_state["fa_search"] = search_q
        r = results[idx]
        type_icon = {"EQUITY":"🟦","ETF":"🟩","CRYPTOCURRENCY":"🟡","INDEX":"🟣"}.get(r["type"].upper(),"⬜")
        st.success(f"{type_icon} **{r['name']}** — `{ticker}` — {r['exchange']} — {r['type']}")
    else:
        st.warning("No results. Try another name.")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data(ttl=600)
def load_data(t):
    try:
        s = yf.Ticker(t)
        return s.info, s.financials, s.balance_sheet, s.cashflow, s.history(period="5y")
    except:
        return None, None, None, None, None

with st.spinner("Loading fundamentals..."):
    info, financials, balance, cashflow, hist = load_data(ticker)

if not info or not isinstance(info, dict):
    st.error("❌ Failed to fetch data. Try a different ticker.")
    st.stop()

# ---------------------------
# FORMATTERS
# ---------------------------
def fmt_cr(x):
    if not isinstance(x, (int,float)): return "N/A"
    if abs(x) >= 1e12: return f"₹{x/1e12:.2f}T"
    if abs(x) >= 1e9:  return f"₹{x/1e9:.2f}B" if not ticker.endswith(".NS") else f"₹{x/1e7:,.0f} Cr"
    if abs(x) >= 1e7:  return f"₹{x/1e7:,.2f} Cr"
    return f"{x:,.0f}"

def fmt_pct(x):
    if isinstance(x,(int,float)): return f"{x*100:.2f}%"
    return "N/A"

def fmt_num(x, dec=2):
    if isinstance(x,(int,float)): return f"{x:,.{dec}f}"
    return "N/A"

def safe(key, fallback="N/A"):
    v = info.get(key)
    return v if v is not None else fallback

# ---------------------------
# 📌 OVERVIEW HEADER
# ---------------------------
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.subheader("📌 Company Overview")

price     = safe("currentPrice") or safe("regularMarketPrice") or safe("previousClose")
prev      = safe("previousClose")
change    = (price - prev) if isinstance(price,(int,float)) and isinstance(prev,(int,float)) else None
change_p  = (change/prev*100) if change and prev else None
mktcap    = safe("marketCap")
currency  = info.get("currency","")
sym       = "₹" if currency=="INR" else ("$" if currency=="USD" else currency+" ")

price_str   = f"{sym}{price:,.2f}" if isinstance(price,(int,float)) else "N/A"
change_str  = f"{change:+.2f} ({change_p:+.2f}%)" if change else ""
change_cls  = "badge-green" if change and change >= 0 else "badge-red"

o1,o2,o3,o4 = st.columns(4)

o1.markdown(f"""
<div class="card">
  <div class="card-label">Company</div>
  <div class="card-value" style="font-size:20px">{safe('longName','N/A')}</div>
  <div class="card-sub">{safe('sector','-')} · {safe('industry','-')}</div>
  <div class="card-sub" style="margin-top:6px">📍 {safe('country','')}</div>
</div>""", unsafe_allow_html=True)

o2.markdown(f"""
<div class="card">
  <div class="card-label">Current Price</div>
  <div class="card-value">{price_str}</div>
  <span class="card-badge {change_cls}">{change_str}</span>
  <div class="card-sub">52W High: {sym}{safe('fiftyTwoWeekHigh','N/A')} | Low: {sym}{safe('fiftyTwoWeekLow','N/A')}</div>
</div>""", unsafe_allow_html=True)

o3.markdown(f"""
<div class="card">
  <div class="card-label">Market Cap</div>
  <div class="card-value">{fmt_cr(mktcap)}</div>
  <div class="card-sub">Float Shares: {fmt_num(safe('floatShares'),0) if isinstance(safe('floatShares'),(int,float)) else 'N/A'}</div>
  <div class="card-sub">Beta: {fmt_num(safe('beta'))}</div>
</div>""", unsafe_allow_html=True)

o4.markdown(f"""
<div class="card">
  <div class="card-label">Dividends</div>
  <div class="card-value">{fmt_pct(safe('dividendYield'))}</div>
  <div class="card-sub">Div Rate: {sym}{fmt_num(safe('dividendRate'))}</div>
  <div class="card-sub">Payout Ratio: {fmt_pct(safe('payoutRatio'))}</div>
</div>""", unsafe_allow_html=True)

# ---------------------------
# 📖 BUSINESS SUMMARY
# ---------------------------
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
desc = info.get("longBusinessSummary","")
if desc:
    with st.expander("📖 Business Summary", expanded=False):
        st.write(desc)

# ---------------------------
# 📊 VALUATION METRICS
# ---------------------------
st.subheader("📊 Valuation Metrics")

v1,v2,v3,v4,v5,v6 = st.columns(6)

def val_card(col, label, value, good_if, threshold, unit=""):
    v = info.get(value) if isinstance(value, str) else value
    if isinstance(v,(int,float)):
        val_str = f"{v:.2f}{unit}"
        if good_if == "low":
            cls = "badge-green" if v < threshold else "badge-red"
            badge = "✅ Cheap" if v < threshold else "⚠️ Expensive"
        else:
            cls = "badge-green" if v > threshold else "badge-red"
            badge = "✅ Good" if v > threshold else "⚠️ Low"
    else:
        val_str, cls, badge = "N/A", "badge-blue", "—"

    col.markdown(f"""
    <div class="card">
      <div class="card-label">{label}</div>
      <div class="card-value" style="font-size:24px">{val_str}</div>
      <span class="card-badge {cls}">{badge}</span>
    </div>""", unsafe_allow_html=True)

val_card(v1, "P/E Ratio",        "trailingPE",       "low",  25)
val_card(v2, "Forward P/E",      "forwardPE",        "low",  20)
val_card(v3, "P/B Ratio",        "priceToBook",      "low",  3)
val_card(v4, "EV/EBITDA",        "enterpriseToEbitda","low", 15)
val_card(v5, "PEG Ratio",        "pegRatio",         "low",  1.5)
val_card(v6, "Price/Sales",      "priceToSalesTrailing12Months","low",3)

# ---------------------------
# 💰 PROFITABILITY
# ---------------------------
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.subheader("💰 Profitability & Returns")

p1,p2,p3,p4,p5,p6 = st.columns(6)

metrics_prof = [
    (p1, "Profit Margin",   fmt_pct(safe("profitMargins"))),
    (p2, "Gross Margin",    fmt_pct(safe("grossMargins"))),
    (p3, "EBITDA Margin",   fmt_pct(safe("ebitdaMargins"))),
    (p4, "Operating Margin",fmt_pct(safe("operatingMargins"))),
    (p5, "ROE",             fmt_pct(safe("returnOnEquity"))),
    (p6, "ROA",             fmt_pct(safe("returnOnAssets"))),
]

for col, label, val in metrics_prof:
    col.markdown(f"""
    <div class="card">
      <div class="card-label">{label}</div>
      <div class="card-value" style="font-size:22px">{val}</div>
    </div>""", unsafe_allow_html=True)

# ---------------------------
# 🏦 FINANCIAL HEALTH
# ---------------------------
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.subheader("🏦 Financial Health")

h1,h2,h3,h4,h5 = st.columns(5)

health_items = [
    (h1, "Total Cash",       fmt_cr(safe("totalCash"))),
    (h2, "Total Debt",       fmt_cr(safe("totalDebt"))),
    (h3, "Debt / Equity",    fmt_num(safe("debtToEquity"))),
    (h4, "Current Ratio",    fmt_num(safe("currentRatio"))),
    (h5, "Quick Ratio",      fmt_num(safe("quickRatio"))),
]

for col, label, val in health_items:
    col.markdown(f"""
    <div class="card">
      <div class="card-label">{label}</div>
      <div class="card-value" style="font-size:22px">{val}</div>
    </div>""", unsafe_allow_html=True)

# ---------------------------
# 📈 GROWTH ANALYSIS
# ---------------------------
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.subheader("📈 Growth Analysis")

def yoy_growth(series, label):
    try:
        vals = series.dropna()
        if len(vals) >= 2:
            g = (vals.iloc[0] - vals.iloc[1]) / abs(vals.iloc[1]) * 100
            return round(g, 2)
    except:
        pass
    return None

growth_data = {}
try:
    if financials is not None and not financials.empty:
        for row_label, key in [
            ("Revenue Growth",      "Total Revenue"),
            ("Net Income Growth",   "Net Income"),
            ("Gross Profit Growth", "Gross Profit"),
            ("EBITDA Growth",       "EBITDA"),
        ]:
            if key in financials.index:
                g = yoy_growth(financials.loc[key], key)
                if g is not None:
                    growth_data[row_label] = g
except:
    pass

if growth_data:
    gcols = st.columns(len(growth_data))
    for i, (label, val) in enumerate(growth_data.items()):
        cls   = "badge-green" if val >= 0 else "badge-red"
        arrow = "▲" if val >= 0 else "▼"
        gcols[i].markdown(f"""
        <div class="card">
          <div class="card-label">{label} (YoY)</div>
          <div class="card-value" style="font-size:24px">{val:+.2f}%</div>
          <span class="card-badge {cls}">{arrow} YoY</span>
        </div>""", unsafe_allow_html=True)

    # Revenue trend chart
    try:
        rev_series = financials.loc["Total Revenue"].dropna().sort_index()
        rev_years  = [str(d.year) for d in rev_series.index]
        rev_vals   = [v/1e7 for v in rev_series.values]

        fig_rev = go.Figure()
        fig_rev.add_trace(go.Bar(
            x=rev_years, y=rev_vals,
            marker_color=["#6366f1"]*len(rev_years),
            name="Revenue (Cr)"
        ))
        if "Net Income" in financials.index:
            ni_series = financials.loc["Net Income"].dropna().sort_index()
            ni_years  = [str(d.year) for d in ni_series.index]
            ni_vals   = [v/1e7 for v in ni_series.values]
            fig_rev.add_trace(go.Bar(x=ni_years, y=ni_vals,
                                      marker_color=["#22c55e"]*len(ni_years),
                                      name="Net Income (Cr)"))

        fig_rev.update_layout(
            template="plotly_dark", height=320, barmode="group",
            title="Revenue & Net Income Trend (₹ Cr)",
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_rev, use_container_width=True)
    except:
        pass
else:
    st.info("Growth data not available for this stock")

# ---------------------------
# 🌡️ FUNDAMENTAL SCORE
# ---------------------------
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.subheader("🌡️ Fundamental Score")

def calc_score(info):
    score  = 0
    total  = 0
    detail = []

    def check(key, good_fn, weight, label, good_desc, bad_desc):
        nonlocal score, total
        v = info.get(key)
        if isinstance(v, (int,float)):
            total += weight
            if good_fn(v):
                score += weight
                detail.append(("bull", f"✅ {label}: {bad_desc if not good_fn(v) else good_desc}"))
            else:
                detail.append(("bear", f"❌ {label}: {bad_desc}"))
        else:
            detail.append(("info", f"ℹ️ {label}: N/A"))

    check("trailingPE",          lambda x: 0 < x < 30,   2, "P/E Ratio",      "Reasonable valuation",    "Expensive or negative")
    check("priceToBook",         lambda x: 0 < x < 4,    1, "P/B Ratio",       "Book value reasonable",   "High premium to book")
    check("profitMargins",       lambda x: x > 0.08,      2, "Profit Margin",   "Healthy margins (>8%)",   "Low/negative margins")
    check("returnOnEquity",      lambda x: x > 0.12,      2, "ROE",             "Strong returns (>12%)",   "Weak returns on equity")
    check("returnOnAssets",      lambda x: x > 0.05,      1, "ROA",             "Efficient use of assets", "Poor asset utilisation")
    check("debtToEquity",        lambda x: x < 100,       2, "Debt/Equity",     "Manageable debt",         "High leverage risk")
    check("currentRatio",        lambda x: x > 1.2,       1, "Current Ratio",   "Liquid enough (>1.2)",    "Liquidity risk")
    check("revenueGrowth",       lambda x: x > 0,         1, "Revenue Growth",  "Growing top line",        "Declining revenue")
    check("earningsGrowth",      lambda x: x > 0,         1, "Earnings Growth", "Earnings expanding",      "Earnings contracting")
    check("dividendYield",       lambda x: x > 0,         1, "Dividend",        "Pays dividend",           "No dividend")

    pct   = round((score / total * 10), 1) if total else 0
    return pct, detail

final_score, score_details = calc_score(info)

ring_color = "#22c55e" if final_score >= 7 else ("#facc15" if final_score >= 5 else "#ef4444")
verdict    = ("Strong Fundamentals", "Looks like a fundamentally solid business.") if final_score >= 7 else \
             (("Moderate Fundamentals", "Mixed signals — some strengths, some concerns.") if final_score >= 5 else
              ("Weak Fundamentals",    "Several red flags. Deep research recommended."))

st.markdown(f"""
<div class="score-wrap">
  <div class="score-ring" style="background: conic-gradient({ring_color} {final_score*36}deg, #1f2937 0deg); color:{ring_color}">
    {final_score}
  </div>
  <div class="score-text">
    <div class="s-label">Fundamental Score</div>
    <div class="s-title">{verdict[0]}</div>
    <div class="s-desc">{verdict[1]}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Score detail pills
scol1, scol2 = st.columns(2)
for i, (stype, text) in enumerate(score_details):
    css_cls = {"bull":"insight-bull","bear":"insight-bear","info":"insight-info"}.get(stype,"insight-info")
    (scol1 if i%2==0 else scol2).markdown(
        f'<div class="insight {css_cls}">{text}</div>', unsafe_allow_html=True)

# ---------------------------
# 🧠 SMART INSIGHTS
# ---------------------------
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.subheader("🧠 Smart Insights")

insights = []

pe      = info.get("trailingPE")
roe     = info.get("returnOnEquity")
margin  = info.get("profitMargins")
de      = info.get("debtToEquity")
cr      = info.get("currentRatio")
beta_v  = info.get("beta")
peg     = info.get("pegRatio")
eveb    = info.get("enterpriseToEbitda")
eg      = info.get("earningsGrowth")
rg      = info.get("revenueGrowth")

if isinstance(pe,(int,float)) and isinstance(eg,(int,float)) and eg > 0:
    peg_calc = pe / (eg*100)
    if peg_calc < 1:
        insights.append(("bull", f"💎 PEG ratio (~{peg_calc:.2f}) suggests stock may be undervalued relative to growth rate"))

if isinstance(roe,(int,float)) and isinstance(de,(int,float)):
    if roe > 0.2 and de < 80:
        insights.append(("bull", "🚀 High ROE with controlled debt — hallmark of a quality compounder"))
    elif roe > 0.2 and de > 150:
        insights.append(("warn", "⚠️ High ROE but also high leverage — returns may be debt-fuelled"))

if isinstance(margin,(int,float)):
    if margin > 0.2:
        insights.append(("bull", f"✅ Exceptional profit margins ({margin*100:.1f}%) — strong pricing power or moat"))
    elif margin < 0:
        insights.append(("bear", "❌ Negative profit margins — company is loss-making"))
    elif margin < 0.05:
        insights.append(("warn", f"⚠️ Thin margins ({margin*100:.1f}%) — vulnerable to cost pressures"))

if isinstance(cr,(int,float)):
    if cr < 1:
        insights.append(("bear", f"❌ Current ratio {cr:.2f} < 1 — short-term liquidity risk"))
    elif cr > 3:
        insights.append(("info", f"ℹ️ Very high current ratio ({cr:.2f}) — may indicate idle cash or inventory buildup"))

if isinstance(beta_v,(int,float)):
    if beta_v > 1.5:
        insights.append(("warn", f"⚠️ High beta ({beta_v:.2f}) — significantly more volatile than the market"))
    elif beta_v < 0.5:
        insights.append(("bull", f"🛡️ Low beta ({beta_v:.2f}) — defensive stock, lower market sensitivity"))

if isinstance(rg,(int,float)) and isinstance(eg,(int,float)):
    if rg > 0.15 and eg > 0.15:
        insights.append(("bull", f"📈 Strong double-digit revenue ({rg*100:.1f}%) and earnings ({eg*100:.1f}%) growth"))
    elif rg < 0 and eg < 0:
        insights.append(("bear", "📉 Both revenue and earnings are shrinking — watch carefully"))

if isinstance(eveb,(int,float)):
    if eveb < 10:
        insights.append(("bull", f"💡 Low EV/EBITDA ({eveb:.1f}x) — potentially undervalued vs peers"))
    elif eveb > 30:
        insights.append(("warn", f"⚠️ High EV/EBITDA ({eveb:.1f}x) — premium priced, growth priced in"))

css_map = {"bull":"insight-bull","bear":"insight-bear","warn":"insight-warn","info":"insight-info"}
if insights:
    ic1, ic2 = st.columns(2)
    for i, (itype, text) in enumerate(insights):
        (ic1 if i%2==0 else ic2).markdown(
            f'<div class="insight {css_map[itype]}">{text}</div>', unsafe_allow_html=True)
else:
    st.info("No strong signal patterns detected from available data.")

# ---------------------------
# 📉 PRICE HISTORY CHART
# ---------------------------
if hist is not None and not hist.empty:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.subheader("📉 5-Year Price History")

    hist["MA50"]  = hist["Close"].rolling(50).mean()
    hist["MA200"] = hist["Close"].rolling(200).mean()

    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Price",
                                line=dict(color="#6366f1", width=2)))
    fig_h.add_trace(go.Scatter(x=hist.index, y=hist["MA50"],  name="50 MA",
                                line=dict(color="#facc15", width=1.2, dash="dot")))
    fig_h.add_trace(go.Scatter(x=hist.index, y=hist["MA200"], name="200 MA",
                                line=dict(color="#f97316", width=1.2, dash="dash")))

    fig_h.update_layout(
        template="plotly_dark", height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.05),
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig_h, use_container_width=True)

# ---------------------------
# 📑 FINANCIAL STATEMENTS
# ---------------------------
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.subheader("📑 Financial Statements")

def clean_df(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    try:
        df.columns = [pd.to_datetime(c).strftime("%Y") for c in df.columns]
    except:
        pass
    def fmt(x):
        if isinstance(x,(int,float)):
            return f"₹{x/1e7:,.2f} Cr" if ticker.endswith(".NS") or ticker.endswith(".BO") else f"{x/1e9:.3f}B"
        return x
    return df.map(fmt)

KEY_INCOME = ["Total Revenue","Gross Profit","EBITDA","EBIT","Net Income","Basic EPS"]
KEY_BALANCE = ["Total Assets","Total Liabilities Net Minority Interest","Stockholders Equity",
               "Total Debt","Cash And Cash Equivalents","Current Assets","Current Liabilities"]
KEY_CASH = ["Operating Cash Flow","Investing Cash Flow","Financing Cash Flow","Free Cash Flow","Capital Expenditure"]

def filter_rows(df, keys):
    if df is None or df.empty: return df
    return df.loc[[k for k in keys if k in df.index]]

t1, t2, t3 = st.tabs(["📊 Income Statement", "🏦 Balance Sheet", "💸 Cash Flow"])

with t1:
    try:
        d = filter_rows(financials, KEY_INCOME)
        c = clean_df(d)
        if c is not None:
            st.dataframe(c, use_container_width=True)
        else:
            st.info("Income statement not available")
    except Exception as e:
        st.info(f"Income data unavailable: {e}")

with t2:
    try:
        d = filter_rows(balance, KEY_BALANCE)
        c = clean_df(d)
        if c is not None:
            st.dataframe(c, use_container_width=True)
        else:
            st.info("Balance sheet not available")
    except Exception as e:
        st.info(f"Balance sheet unavailable: {e}")

with t3:
    try:
        d = filter_rows(cashflow, KEY_CASH)
        c = clean_df(d)
        if c is not None:
            st.dataframe(c, use_container_width=True)
        else:
            st.info("Cash flow not available")
    except Exception as e:
        st.info(f"Cash flow unavailable: {e}")

# ---------------------------
# 📥 EXPORT
# ---------------------------
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.subheader("📥 Export Data")

ec1, ec2, ec3 = st.columns(3)

try:
    if financials is not None:
        ec1.download_button("📥 Income Statement CSV",
                            clean_df(financials).to_csv().encode(),
                            f"{ticker}_income.csv","text/csv")
except: pass
try:
    if balance is not None:
        ec2.download_button("📥 Balance Sheet CSV",
                            clean_df(balance).to_csv().encode(),
                            f"{ticker}_balance.csv","text/csv")
except: pass
try:
    if cashflow is not None:
        ec3.download_button("📥 Cash Flow CSV",
                            clean_df(cashflow).to_csv().encode(),
                            f"{ticker}_cashflow.csv","text/csv")
except: pass

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.warning("⚠️ This dashboard is for **educational purposes only**. Data sourced from Yahoo Finance and may have delays or inaccuracies. Nothing here constitutes financial advice.")
