import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(
    page_title="QUANT-X AI PRO",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {background-color:#070a10;color:#e9eef7}
[data-testid="stMetric"] {
    background:#111827;
    border:1px solid #263247;
    padding:18px;
    border-radius:12px;
}
h1,h2,h3 {color:#65e6bd}
</style>
""", unsafe_allow_html=True)

st.title("📈 QUANT-X AI PRO")
st.caption("داشبورد تحلیل بازار رمزارز | اطلاعات عمومی بازار")

BASE = "https://api.kraken.com/0/public"

COINS = {
    "Bitcoin": "XBTUSD",
    "Ethereum": "ETHUSD",
    "Solana": "SOLUSD",
    "XRP": "XRPUSD",
    "Dogecoin": "XDGUSD"
}

@st.cache_data(ttl=30)
def get_ticker(pair):
    url = f"{BASE}/Ticker"
    r = requests.get(
        url,
        params={"pair": pair},
        timeout=15
    )
    r.raise_for_status()
    data = r.json()

    if data.get("error"):
        raise ValueError(str(data["error"]))

    result = data.get("result", {})
    if not result:
        raise ValueError("No ticker data returned")

    item = next(iter(result.values()))

    return {
        "price": float(item["c"][0]),
        "high": float(item["h"][1]),
        "low": float(item["l"][1]),
        "volume": float(item["v"][1]),
        "open": float(item["o"])
    }

@st.cache_data(ttl=60)
def get_candles(pair, interval=5):
    r = requests.get(
        f"{BASE}/OHLC",
        params={"pair": pair, "interval": interval},
        timeout=15
    )
    r.raise_for_status()
    data = r.json()

    if data.get("error"):
        raise ValueError(str(data["error"]))

    result = data.get("result", {})
    key = next((k for k in result if k != "last"), None)

    if key is None:
        raise ValueError("No candle data returned")

    df = pd.DataFrame(
        result[key],
        columns=[
            "time", "open", "high", "low",
            "close", "vwap", "volume", "count"
        ]
    )

    df["time"] = pd.to_datetime(
        df["time"], unit="s", utc=True
    )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna()

with st.sidebar:
    st.header("⚙️ تنظیمات")
    coin = st.selectbox("انتخاب رمزارز", list(COINS.keys()))
    interval = st.selectbox(
        "بازه زمانی نمودار",
        [1, 5, 15, 30, 60, 240],
        index=1,
        format_func=lambda x: f"{x} دقیقه" if x < 60 else f"{x//60} ساعت"
    )
    st.caption("داده‌ها از API عمومی Kraken دریافت می‌شوند.")
    if st.button("🔄 به‌روزرسانی"):
        st.cache_data.clear()
        st.rerun()

pair = COINS[coin]

try:
    ticker = get_ticker(pair)

    change = (
        (ticker["price"] - ticker["open"])
        / ticker["open"] * 100
        if ticker["open"] else 0
    )

    st.subheader(f"{coin} / USD")

    a, b, c, d = st.columns(4)
    a.metric("قیمت فعلی (USD)", f"${ticker['price']:,.6f}")
    b.metric("تغییر ۲۴ ساعته", f"{change:+.2f}%")
    c.metric("بالاترین قیمت ۲۴ ساعت", f"${ticker['high']:,.6f}")
    d.metric("پایین‌ترین قیمت ۲۴ ساعت", f"${ticker['low']:,.6f}")

    st.caption(
        "آخرین دریافت اطلاعات: "
        + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    )

    df = get_candles(pair, interval)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name=coin
    ))

    fig.update_layout(
        title=f"{coin} — نمودار کندل",
        template="plotly_dark",
        height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 خلاصه بازار")

    x, y = st.columns(2)
    x.metric("حجم معاملات ۲۴ ساعته", f"{ticker['volume']:,.2f}")
    y.metric("تعداد کندل‌های دریافت‌شده", len(df))

    st.subheader("🕯️ آخرین کندل‌ها")
    st.dataframe(
        df.tail(10).sort_values("time", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "این نسخه ابزار نمایش داده و نمودار است؛ "
        "سیگنال قطعی خرید و فروش یا تضمین سود ارائه نمی‌کند."
    )

except Exception as e:
    st.error("دریافت اطلاعات بازار ناموفق بود.")
    st.warning(
        "ممکن است اتصال سرور به منبع داده برقرار نباشد. "
        "کمی بعد دوباره امتحان کن."
    )
    st.caption(f"جزئیات فنی: {e}")
