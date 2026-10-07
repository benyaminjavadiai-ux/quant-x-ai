import os, time, json, threading
from datetime import datetime, timezone
from collections import deque
import numpy as np, pandas as pd, requests, streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingClassifier

BASE='https://fapi.binance.com'
WS_BASE='wss://fstream.binance.com/market/ws'

st.set_page_config(page_title='QUANT-X AI PRO', page_icon='◈', layout='wide')
st.markdown('''<style>
.stApp{background:#070a10;color:#e9eef7}.block-container{padding-top:1rem;max-width:1800px}
.card{background:#0b111a;border:1px solid #1d2a3c;border-radius:14px;padding:15px;margin:5px 0}
.big{font-size:42px;font-weight:800}.muted{color:#8090a8;font-size:12px}.green{color:#35d07f}.red{color:#ff5d78}.yellow{color:#ffc857}
</style>''', unsafe_allow_html=True)

@st.cache_data(ttl=5, show_spinner=False)
def get(path, params=None):
    r=requests.get(BASE+path,params=params or {},timeout=10); r.raise_for_status(); return r.json()

def candles(symbol,tf,limit=1000):
    raw=get('/fapi/v1/klines',{'symbol':symbol,'interval':tf,'limit':limit})
    c=['t','o','h','l','c','v','ct','qv','n','tb','tq','x']; d=pd.DataFrame(raw,columns=c)
    for x in ['o','h','l','c','v','qv','tb','tq']: d[x]=pd.to_numeric(d[x])
    d.t=pd.to_datetime(d.t,unit='ms',utc=True); return d

def indicators(d):
    x=d.copy(); c=x.c
    x['ema20']=c.ewm(span=20,adjust=False).mean(); x['ema50']=c.ewm(span=50,adjust=False).mean(); x['ema200']=c.ewm(span=200,adjust=False).mean()
    delta=c.diff(); up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); dn=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean(); x['rsi']=100-100/(1+up/dn.replace(0,np.nan))
    prev=c.shift(1); tr=pd.concat([x.h-x.l,(x.h-prev).abs(),(x.l-prev).abs()],axis=1).max(axis=1); x['atr']=tr.ewm(alpha=1/14,adjust=False).mean()
    x['macd']=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); x['macds']=x.macd.ewm(span=9,adjust=False).mean()
    x['ret1']=c.pct_change(); x['volz']=(x.v-x.v.rolling(30).mean())/x.v.rolling(30).std()
    x['range']= (x.h-x.l)/c
    x['body']=(x.c-x.o)/c
    return x.dropna().reset_index(drop=True)

def book(symbol):
    b=get('/fapi/v1/depth',{'symbol':symbol,'limit':100}); bids=np.array(b['bids'],float); asks=np.array(b['asks'],float)
    bv=np.sum(bids[:,0]*bids[:,1]); av=np.sum(asks[:,0]*asks[:,1]); imb=(bv-av)/(bv+av) if bv+av else 0
    return imb,bv,av,bids,asks

def market(symbol):
    p=get('/fapi/v1/premiumIndex',{'symbol':symbol}); oi=get('/fapi/v1/openInterest',{'symbol':symbol})
    try: hist=get('/futures/data/openInterestHist',{'symbol':symbol,'period':'5m','limit':12}); ov=np.array([float(z['sumOpenInterest']) for z in hist]); oid=(ov[-1]-ov[0])/ov[0]
    except: oid=0
    try: ls=get('/futures/data/globalLongShortAccountRatio',{'symbol':symbol,'period':'5m','limit':1}); lsr=float(ls[-1]['longShortRatio'])
    except: lsr=1
    try: tk=get('/futures/data/takerlongshortRatio',{'symbol':symbol,'period':'5m','limit':1}); taker=float(tk[-1]['buySellRatio'])
    except: taker=1
    return float(p['markPrice']),float(p['lastFundingRate']),float(oi['openInterest']),oid,lsr,taker

def features(x5,x15,x1h,imb,funding,oid,lsr,taker):
    def tf(x):
        z=x.iloc[-1]; return [z.c/z.ema20-1,z.c/z.ema50-1,z.c/z.ema200-1,(z.rsi-50)/50,(z.macd-z.macds)/(z.atr/z.c),z.volz,z.body,z.range]
    a=np.array(tf(x5)+tf(x15)+tf(x1h)+[imb,funding*1000,oid,lsr-1,taker-1],float)
    return np.nan_to_num(a,nan=0,posinf=0,neginf=0)

def score_rule(f):
    # Explainable ensemble. No fake win-rate claims.
    s=50+np.tanh(np.sum(f[[0,1,2,8]])*35)*18
    s+=np.clip(f[3]*12,-10,10)+np.clip(f[4]*4,-8,8)+np.clip(f[9]*12,-8,8)+np.clip(-f[10]*2,-5,5)+np.clip(f[12]*15,-6,6)
    return float(np.clip(s,0,100))

def train_model(x):
    cols=['ret1','rsi','macd','volz','range','body','c','v']
    z=x[cols].copy(); z['ret5']=x.c.pct_change(5); z['ret10']=x.c.pct_change(10); z['ema_gap']=x.c/x.ewm(span=20).mean()-1
    y=(x.c.shift(-3)>x.c*1.001).astype(int)
    ok=y.notna(); X=z[ok].replace([np.inf,-np.inf],np.nan).fillna(0); Y=y[ok]
    split=int(len(X)*.8); clf=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,learning_rate=.05,l2_regularization=.2,random_state=7); clf.fit(X.iloc[:split],Y.iloc[:split])
    p=float(clf.predict_proba(X.iloc[-1:].replace([np.inf,-np.inf],np.nan).fillna(0))[0,1]); return clf,p

def backtest(x, fee=0.0005, threshold=67):
    z=x.copy(); s=[]; pos=0; entry=0; eq=1.; trades=[]
    for i in range(210,len(z)-3):
        r=z.iloc[i]; trend=(1 if r.c>r.ema20>r.ema50 else -1 if r.c<r.ema20<r.ema50 else 0)
        sc=50+trend*15+(r.rsi-50)*.3+np.clip(r.volz,-2,2)*3
        if pos==0 and sc>=threshold: pos=1; entry=r.c*(1+fee); trades.append(['LONG',z.t.iloc[i],r.c,sc])
        elif pos==0 and sc<=100-threshold: pos=-1; entry=r.c*(1-fee); trades.append(['SHORT',z.t.iloc[i],r.c,sc])
        elif pos==1 and (sc<55 or r.c<r.ema20): eq*=r.c/entry*(1-fee); pos=0
        elif pos==-1 and (sc>45 or r.c>r.ema20): eq*=entry/r.c*(1-fee); pos=0
    if pos==1: eq*=z.c.iloc[-1]/entry
    if pos==-1: eq*=entry/z.c.iloc[-1]
    return eq-1,len(trades)

def telegram(msg,token,chat):
    if not token or not chat:return False
    try:
        requests.post(f'https://api.telegram.org/bot{token}/sendMessage',json={'chat_id':chat,'text':msg},timeout=8); return True
    except:return False

# Sidebar
with st.sidebar:
    st.title('◈ QUANT-X AI PRO')
    symbol=st.text_input('Symbol','BTCUSDT').upper().strip()
    tf=st.selectbox('Chart timeframe',['1m','5m','15m','1h','4h','1d'],index=1)
    auto=st.checkbox('Auto refresh',True); sec=st.slider('Refresh seconds',5,60,10)
    st.divider(); st.subheader('Telegram Alerts')
    tg_token=st.text_input('Bot token',value=os.getenv('TG_TOKEN',''),type='password')
    tg_chat=st.text_input('Chat ID',value=os.getenv('TG_CHAT',''))
    alert=st.checkbox('Send signal alerts',False)
    st.divider(); st.caption('Public market data. Trading execution is disabled by design.')

try:
    x5=indicators(candles(symbol,'5m')); x15=indicators(candles(symbol,'15m')); x1h=indicators(candles(symbol,'1h')); xc=indicators(candles(symbol,tf))
    price,funding,oi,oid,lsr,taker=market(symbol); imb,bv,av,bids,asks=book(symbol)
    f=features(x5,x15,x1h,imb,funding,oid,lsr,taker); rule=score_rule(f)
    clf,mlprob=train_model(x5); ml=mlprob*100; final=.6*rule+.4*ml
    signal='LONG' if final>=67 else 'SHORT' if final<=33 else 'WAIT'; conf=abs(final-50)*2
except Exception as e:
    st.error(str(e)); st.stop()

st.markdown(f'''<div class="card"><span class="muted">LIVE / BINANCE USDⓈ-M • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</span><h1 style="margin:3px 0">{symbol}</h1><span class="big">${price:,.2f}</span></div>''',unsafe_allow_html=True)
a,b,c,d,e,g=st.columns(6); a.metric('FINAL SCORE',f'{final:.1f}',signal); b.metric('Rule score',f'{rule:.1f}'); c.metric('ML up probability',f'{ml:.1f}%'); d.metric('Funding',f'{funding*100:.4f}%'); e.metric('OI Δ 1h',f'{oid*100:+.2f}%'); g.metric('Book imbalance',f'{imb*100:+.1f}%')

left,right=st.columns([2.3,1])
with left:
    fig=go.Figure(go.Candlestick(x=xc.t,open=xc.o,high=xc.h,low=xc.l,close=xc.c,name='Price'))
    for col in ['ema20','ema50','ema200']: fig.add_trace(go.Scatter(x=xc.t,y=xc[col],name=col,mode='lines'))
    fig.update_layout(template='plotly_dark',height=520,xaxis_rangeslider_visible=False,margin=dict(l=5,r=5,t=10,b=5),paper_bgcolor='#070a10',plot_bgcolor='#070a10')
    st.plotly_chart(fig,use_container_width=True)
with right:
    st.markdown(f'<div class="card" style="text-align:center"><span class="muted">ENGINE DECISION</span><div class="big">{final:.0f}</div><h2>{signal}</h2><span class="muted">confidence proxy {conf:.0f}%</span></div>',unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({'Metric':['RSI','MACD','ATR','Long/Short','Taker ratio','Spread'],'Value':[round(x5.rsi.iloc[-1],2),round(x5.macd.iloc[-1]-x5.macds.iloc[-1],5),round(x5.atr.iloc[-1],2),round(lsr,3),round(taker,3),round((asks[0,0]-bids[0,0])/price*10000,3)]}),hide_index=True,use_container_width=True)

c1,c2=st.columns(2)
with c1:
    r=x5.tail(180); q=go.Figure(); q.add_trace(go.Scatter(x=r.t,y=r.rsi,name='RSI')); q.add_hline(y=70,line_dash='dot'); q.add_hline(y=30,line_dash='dot'); q.update_layout(template='plotly_dark',height=300,margin=dict(l=5,r=5,t=5,b=5)); st.plotly_chart(q,use_container_width=True)
with c2:
    q=go.Figure(); q.add_trace(go.Bar(x=r.t,y=r.v,name='Volume')); q.add_trace(go.Scatter(x=r.t,y=r.v.rolling(20).mean(),name='Volume MA')); q.update_layout(template='plotly_dark',height=300,margin=dict(l=5,r=5,t=5,b=5)); st.plotly_chart(q,use_container_width=True)

st.subheader('Liquidity / Order Book')
st.dataframe(pd.DataFrame({'Bid Price':bids[:20,0],'Bid Qty':bids[:20,1],'Ask Price':asks[:20,0],'Ask Qty':asks[:20,1]}),hide_index=True,use_container_width=True)

st.subheader('Multi-Timeframe Matrix')
rows=[]
for name,z in [('5m',x5),('15m',x15),('1h',x1h),('4h',indicators(candles(symbol,'4h'))),('1d',indicators(candles(symbol,'1d')))]:
    u=z.iloc[-1]; stt='BULL' if u.c>u.ema20>u.ema50 else 'BEAR' if u.c<u.ema20<u.ema50 else 'MIXED'; rows.append([name,u.c,u.rsi,stt,u.volz])
st.dataframe(pd.DataFrame(rows,columns=['TF','Price','RSI','Structure','Volume Z']),hide_index=True,use_container_width=True)

st.subheader('Backtest / Research')
ret,n=backtest(x5); p1,p2,p3=st.columns(3); p1.metric('Simple historical return',f'{ret*100:+.2f}%'); p2.metric('Signal events',n); p3.metric('Test type','Rule-based / 5m')
st.caption('Backtest is illustrative and does not prove future profitability. Costs/slippage matter.')

if alert and signal!='WAIT':
    key=f'{symbol}:{signal}:{round(final/5)}'
    if st.session_state.get('last_alert')!=key:
        if telegram(f'QUANT-X {symbol}\nSignal: {signal}\nScore: {final:.1f}\nPrice: {price:.2f}\nFunding: {funding*100:.4f}%\nOI Δ: {oid*100:+.2f}%',tg_token,tg_chat): st.session_state.last_alert=key; st.success('Telegram alert sent.')

st.markdown('<div class="card"><b>Safety:</b> This build does not place live orders. The ML probability is a small gradient-boosting research model trained on the current historical window; it is not a guarantee of accuracy. Use paper trading and out-of-sample validation before risking capital.</div>',unsafe_allow_html=True)

if auto:
    time.sleep(sec); st.rerun()
