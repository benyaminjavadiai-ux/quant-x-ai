import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="QUANT-X Backtest", page_icon="🧪", layout="wide")
st.title("🧪 QUANT-X — آزمایشگاه بک‌تست")
st.caption("داده عمومی Kraken؛ فقط معاملات فرضی، بدون کیف پول و بدون معامله واقعی.")

BASE = "https://api.kraken.com/0/public"
PAIRS = {"Bitcoin": "XBTUSD", "Ethereum": "ETHUSD", "Solana": "SOLUSD", "XRP": "XRPUSD", "Dogecoin": "XDGUSD"}

@st.cache_data(ttl=300)
def candles(pair, interval):
    r = requests.get(f"{BASE}/OHLC", params={"pair": pair, "interval": interval}, timeout=25)
    r.raise_for_status()
    j = r.json()
    if j.get("error"):
        raise ValueError(str(j["error"]))
    result = j.get("result", {})
    key = next((k for k in result if k != "last"), None)
    if key is None:
        raise ValueError("داده کندل پیدا نشد")
    d = pd.DataFrame(result[key], columns=["time","open","high","low","close","vwap","volume","count"])
    d["time"] = pd.to_datetime(d["time"], unit="s", utc=True)
    for c in ["open","high","low","close","volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    # حذف آخرین کندل که ممکن است هنوز بسته نشده باشد
    return d.iloc[:-1].reset_index(drop=True) if len(d) > 1 else d

def indicators(d):
    d = d.copy()
    d["ema20"] = d.close.ewm(span=20, adjust=False).mean()
    d["ema50"] = d.close.ewm(span=50, adjust=False).mean()
    delta = d.close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    al = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    d["rsi"] = (100 - 100/(1 + ag/al.replace(0, np.nan))).fillna(50)
    d["macd"] = d.close.ewm(span=12, adjust=False).mean() - d.close.ewm(span=26, adjust=False).mean()
    d["macd_signal"] = d.macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([d.high-d.low, (d.high-d.close.shift()).abs(), (d.low-d.close.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    return d

def score(row):
    s = (1 if row.close > row.ema20 else -1)
    s += (1 if row.ema20 > row.ema50 else -1)
    s += (1 if row.macd > row.macd_signal else -1)
    if 50 < row.rsi < 70: s += 1
    elif 30 < row.rsi < 50: s -= 1
    return s

def backtest(d, fee, slip, stop_mult, target_mult):
    trades, equity = [], 1000.0
    i, n = 60, len(d)
    costs = 2 * (fee + slip) / 100
    while i < n-1:
        row = d.iloc[i]
        s = score(row)
        direction = 1 if s >= 3 else (-1 if s <= -3 else 0)
        if direction == 0 or not np.isfinite(row.atr) or row.atr <= 0:
            i += 1
            continue
        ent_i = i+1
        entry = float(d.iloc[ent_i].open)
        stop = entry - direction * stop_mult * float(row.atr)
        target = entry + direction * target_mult * float(row.atr)
        exit_i, exit_price, reason = n-1, float(d.iloc[-1].close), "پایان داده"
        for j in range(ent_i, n):
            c = d.iloc[j]
            stop_hit = (c.low <= stop) if direction == 1 else (c.high >= stop)
            target_hit = (c.high >= target) if direction == 1 else (c.low <= target)
            # اگر هر دو در یک کندل لمس شوند، محافظه‌کارانه حد ضرر را حساب می‌کنیم.
            if stop_hit:
                exit_i, exit_price, reason = j, stop, "حد ضرر"
                break
            if target_hit:
                exit_i, exit_price, reason = j, target, "هدف"
                break
        gross = direction * (exit_price-entry)/entry
        net = gross - costs
        equity *= max(0.000001, 1+net)
        trades.append({"ورود UTC": d.iloc[ent_i].time, "خروج UTC": d.iloc[exit_i].time,
                       "جهت": "LONG" if direction == 1 else "SHORT", "قیمت ورود": entry,
                       "قیمت خروج": exit_price, "علت خروج": reason,
                       "بازده خالص ٪": net*100, "سرمایه فرضی": equity})
        i = max(exit_i+1, ent_i+1)
    return pd.DataFrame(trades), equity

with st.sidebar:
    coin = st.selectbox("رمزارز", list(PAIRS))
    interval = st.selectbox("بازه کندل", [5,15,30,60,240], index=1,
                            format_func=lambda x: f"{x} دقیقه" if x < 60 else f"{x//60} ساعت")
    fee = st.number_input("کارمزد هر سمت (%)", 0.0, 2.0, 0.26, 0.01)
    slip = st.number_input("لغزش هر سمت (%)", 0.0, 2.0, 0.05, 0.01)
    stop_mult = st.number_input("حد ضرر بر حسب ATR", 0.5, 5.0, 1.5, 0.25)
    target_mult = st.number_input("هدف بر حسب ATR", 0.5, 10.0, 3.0, 0.25)
    run = st.button("▶️ اجرای آزمایش", type="primary")

st.warning("Kraken معمولاً فقط حدود ۷۲۰ کندل اخیر را برمی‌گرداند؛ این آزمایش اولیه است و عملکرد آینده را تضمین نمی‌کند.")
if run:
    try:
        with st.spinner("دریافت داده و آزمایش سیگنال‌ها..."):
            d = indicators(candles(PAIRS[coin], interval))
            if len(d) < 100:
                st.error("داده کافی دریافت نشد.")
                st.stop()
            trades, final_equity = backtest(d, fee, slip, stop_mult, target_mult)
        if trades.empty:
            st.info("در بازه بررسی‌شده سیگنالی با شرایط فعلی پیدا نشد.")
        else:
            wins = int((trades["بازده خالص ٪"] > 0).sum())
            losses = int((trades["بازده خالص ٪"] <= 0).sum())
            gp = trades.loc[trades["بازده خالص ٪"] > 0, "بازده خالص ٪"].sum()
            gl = abs(trades.loc[trades["بازده خالص ٪"] <= 0, "بازده خالص ٪"].sum())
            pf = gp/gl if gl else np.nan
            a,b,c,e = st.columns(4)
            a.metric("معاملات فرضی", len(trades))
            b.metric("نرخ برد", f"{wins/len(trades)*100:.1f}%")
            c.metric("بازده خالص", f"{(final_equity/1000-1)*100:+.2f}%")
            e.metric("Profit Factor", f"{pf:.2f}" if np.isfinite(pf) else "نامشخص")
            st.caption(f"معاملات زیان‌ده/سربه‌سر: {losses} | هزینه رفت‌وبرگشت فرضی: {2*(fee+slip):.2f}%")
            st.dataframe(trades.iloc[::-1], use_container_width=True, hide_index=True)
            st.download_button("دانلود نتایج CSV", trades.to_csv(index=False).encode("utf-8-sig"),
                               "quantx_backtest.csv", "text/csv")
        st.caption(f"کندل‌های بسته‌شده: {len(d)} | از {d.iloc[0].time} تا {d.iloc[-1].time}")
    except Exception as ex:
        st.error("آزمایش اجرا نشد؛ لطفاً متن خطا را بفرست.")
        st.caption(str(ex))
else:
    st.write("از نوار کناری تنظیمات را انتخاب کن و «اجرای آزمایش» را بزن.")
    st.write("سیگنال‌ها از همان قواعد ساده داشبورد اصلی استفاده می‌کنند. نتایج اولیه‌اند و تضمین سود نیستند.")
