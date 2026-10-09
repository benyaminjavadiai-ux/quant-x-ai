import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(
    page_title="QUANT-X AI PRO",
    page_icon="📈",
    layout="wide"
)

BASE = "https://api.kraken.com/0/public"

COINS = {
    "Bitcoin": "XBTUSD",
    "Ethereum": "ETHUSD",
    "Solana": "SOLUSD",
    "XRP": "XRPUSD",
    "Dogecoin": "XDGUSD"
}

st.markdown("""
<style>
.stApp {
    background-color: #080c14;
    color: #eef2ff;
}
[data-testid="stMetric"] {
    background: #121b2a;
    border: 1px solid #26374e;
    padding: 15px;
    border-radius: 12px;
}
h1, h2, h3 {
    color: #65e6bd;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=30)
def get_ticker(pair):
    r = requests.get(
        f"{BASE}/Ticker",
        params={"pair": pair},
        timeout=15
    )
    r.raise_for_status()
    data = r.json()

    if data.get("error"):
        raise ValueError(str(data["error"]))

    result = data.get("result", {})
    if not result:
        raise ValueError("No market data")

    item = next(iter(result.values()))

    return {
        "price": float(item["c"][0]),
        "high": float(item["h"][1]),
        "low": float(item["l"][1]),
        "volume": float(item["v"][1]),
        "open": float(item["o"])
    }


@st.cache_data(ttl=60)
def get_candles(pair, interval):
    r = requests.get(
        f"{BASE}/OHLC",
        params={"pair": pair, "interval": interval},
        timeout=20
    )
    r.raise_for_status()
    data = r.json()

    if data.get("error"):
        raise ValueError(str(data["error"]))

    result = data.get("result", {})
    key = next((k for k in result if k != "last"), None)

    if key is None:
        raise ValueError("No candle data")

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

    for col in [
        "open", "high", "low", "close", "volume"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna().reset_index(drop=True)


def add_indicators(df):
    df = df.copy()

    df["EMA20"] = df["close"].ewm(
        span=20, adjust=False
    ).mean()

    df["EMA50"] = df["close"].ewm(
        span=50, adjust=False
    ).mean()

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    ema12 = df["close"].ewm(
        span=12, adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26, adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    df["ATR"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1).rolling(14).mean()

    return df


def analyze(df):
    last = df.iloc[-1]

    score = 0

    if last["close"] > last["EMA20"]:
        score += 1
    else:
        score -= 1

    if last["EMA20"] > last["EMA50"]:
        score += 1
    else:
        score -= 1

    if last["MACD"] > last["MACD_signal"]:
        score += 1
    else:
        score -= 1

    if 50 < last["RSI"] < 70:
        score += 1
    elif 30 < last["RSI"] < 50:
        score -= 1

    if score >= 3:
        signal = "احتمال صعود"
        direction = "LONG"
    elif score <= -3:
        signal = "احتمال نزول"
        direction = "SHORT"
    else:
        signal = "نامشخص؛ صبر بهتر است"
        direction = "WAIT"

    return signal, direction, score


st.title("📈 QUANT-X AI PRO")
st.caption(
    "داشبورد تحلیل بازار | فقط داده عمومی | بدون معامله خودکار"
)

with st.sidebar:
    st.header("⚙️ تنظیمات")

    coin = st.selectbox(
        "انتخاب رمزارز",
        list(COINS.keys())
    )

    interval = st.selectbox(
        "بازه تحلیل",
        [5, 15, 30, 60, 240],
        index=1,
        format_func=lambda x:
        f"{x} دقیقه" if x < 60 else f"{x//60} ساعت"
    )

    st.caption("منبع داده: Kraken Public API")

    if st.button("🔄 به‌روزرسانی"):
        st.cache_data.clear()
        st.rerun()


try:
    pair = COINS[coin]

    ticker = get_ticker(pair)

    df = get_candles(pair, interval)

    if len(df) < 60:
        st.warning("برای تحلیل معتبرتر، داده‌های بیشتری لازم است.")
        st.stop()

    df = add_indicators(df)

    signal, direction, score = analyze(df)

    change = (
        (ticker["price"] - ticker["open"])
        / ticker["open"] * 100
        if ticker["open"] else 0
    )

    st.subheader(f"{coin} / USD")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "قیمت فعلی",
        f"${ticker['price']:,.4f}"
    )

    c2.metric(
        "تغییر ۲۴ ساعته",
        f"{change:+.2f}%"
    )

    c3.metric(
        "بالاترین قیمت ۲۴ ساعت",
        f"${ticker['high']:,.4f}"
    )

    c4.metric(
        "پایین‌ترین قیمت ۲۴ ساعت",
        f"${ticker['low']:,.4f}"
    )

    st.divider()

    st.header("🧠 تحلیل تکنیکال")

    s1, s2, s3 = st.columns(3)

    s1.metric("وضعیت تحلیل", signal)
    s2.metric("امتیاز جهت بازار", f"{score} از 4")
    s3.metric("RSI", f"{df.iloc[-1]['RSI']:.1f}")

    st.caption(
        "این امتیاز حاصل چند قاعده ساده تکنیکال است؛ "
        "احتمال آماری موفقیت معامله محسوب نمی‌شود."
    )

    last = df.iloc[-1]
    price = float(last["close"])
    atr = float(last["ATR"]) if pd.notna(last["ATR"]) else 0

    st.subheader("🎯 سطوح احتمالی مدیریت ریسک")

    if direction == "LONG" and atr > 0:
        stop = price - 1.5 * atr
        target1 = price + 1.5 * atr
        target2 = price + 3 * atr

        a, b, c = st.columns(3)
        a.metric("حد ضرر فرضی", f"${stop:,.4f}")
        b.metric("هدف اول فرضی", f"${target1:,.4f}")
        c.metric("هدف دوم فرضی", f"${target2:,.4f}")

    elif direction == "SHORT" and atr > 0:
        stop = price + 1.5 * atr
        target1 = price - 1.5 * atr
        target2 = price - 3 * atr

        a, b, c = st.columns(3)
        a.metric("حد ضرر فرضی", f"${stop:,.4f}")
        b.metric("هدف اول فرضی", f"${target1:,.4f}")
        c.metric("هدف دوم فرضی", f"${target2:,.4f}")

    else:
        st.info(
            "فعلاً جهت بازار به‌اندازه کافی مشخص نیست؛ "
            "سطوح ورود نمایش داده نمی‌شوند."
        )

    st.caption(
        "این سطوح صرفاً نمونه محاسباتی بر اساس ATR هستند؛ "
        "سیگنال تأییدشده یا توصیه قطعی معامله نیستند."
    )

    st.divider()

    st.header("📊 نمودار قیمت")

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name=coin
    ))

    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["EMA20"],
        name="EMA 20",
        mode="lines"
    ))

    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["EMA50"],
        name="EMA 50",
        mode="lines"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=520,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("📉 شاخص RSI")

    rsi_fig = go.Figure()

    rsi_fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["RSI"],
        name="RSI",
        mode="lines"
    ))

    rsi_fig.add_hline(y=70, line_dash="dash")
    rsi_fig.add_hline(y=30, line_dash="dash")

    rsi_fig.update_layout(
        template="plotly_dark",
        height=260,
        yaxis_range=[0, 100]
    )

    st.plotly_chart(
        rsi_fig,
        use_container_width=True
    )

    st.subheader("🕯️ آخرین کندل‌ها")

    st.dataframe(
        df.tail(10).sort_values(
            "time", ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "آخرین به‌روزرسانی: "
        + datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    )

    st.warning(
        "این ابزار آزمایشی است. هنوز بک‌تست کامل، "
        "کارمزد معاملات و لغزش قیمت در عملکرد استراتژی "
        "محاسبه نشده‌اند. هیچ نرخ برد یا سودی تضمین نمی‌شود."
    )

except Exception as e:
    st.error("دریافت داده یا تحلیل بازار ناموفق بود.")
    st.info(
        "چند لحظه بعد دوباره امتحان کن. "
        "اگر مشکل ادامه داشت، از صفحه عکس بفرست."
    )
    st.caption(f"جزئیات فنی: {e}")
