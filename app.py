import streamlit as st
import pandas as pd, numpy as np
from datetime import datetime,timedelta
from kiteconnect import KiteConnect

st.set_page_config(page_title="ALPHA Live v1.1",page_icon="⚡",layout="wide")
st.title("⚡ ALPHA Live v1.1")
st.caption("Zerodha Kite live quotes + historical candles • manual execution only")

try:
    KEY=st.secrets["KITE_API_KEY"]; SECRET=st.secrets["KITE_API_SECRET"]
except Exception:
    st.error("Kite credentials missing in Streamlit Secrets."); st.stop()
kite=KiteConnect(api_key=KEY)

if "access_token" not in st.session_state:
    rt=st.query_params.get("request_token")
    if rt:
        try:
            s=kite.generate_session(rt,api_secret=SECRET)
            st.session_state.access_token=s["access_token"]; st.query_params.clear(); st.rerun()
        except Exception as e: st.error(f"Login exchange failed: {e}")
if "access_token" not in st.session_state:
    st.info("Daily Zerodha login required.")
    st.link_button("Login with Zerodha",kite.login_url(),use_container_width=True); st.stop()

kite.set_access_token(st.session_state.access_token)
try: profile=kite.profile()
except Exception:
    st.session_state.pop("access_token",None); st.rerun()
st.success("Zerodha connected")

SYMS=["RELIANCE","HDFCBANK","ICICIBANK","INFY","TCS","SBIN","ITC","LT","BHARTIARTL","MARUTI","SUNPHARMA","TATAMOTORS","AXISBANK","KOTAKBANK","BAJFINANCE","M&M","NTPC","POWERGRID","TITAN","HCLTECH","WIPRO","TECHM","ADANIPORTS","ONGC","COALINDIA","TATASTEEL","JSWSTEEL","CIPLA","DRREDDY","APOLLOHOSP"]

@st.cache_data(ttl=86400)
def imap():
    return {x["tradingsymbol"]:x["instrument_token"] for x in kite.instruments("NSE")}

def hist(sym):
    tok=imap().get(sym)
    if not tok:return pd.DataFrame()
    x=pd.DataFrame(kite.historical_data(tok,datetime.now()-timedelta(days=550),datetime.now(),"day"))
    if len(x): x=x.rename(columns=str.title).set_index("Date")
    return x

def rsi(c,n=14):
    d=c.diff();g=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();l=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+g/l.replace(0,np.nan))

def analyze(sym,ltp,capital,risk):
    d=hist(sym)
    if len(d)<210:return None
    c=d.Close.astype(float);h=d.High.astype(float);v=d.Volume.astype(float)
    s20=c.rolling(20).mean();s50=c.rolling(50).mean();s200=c.rolling(200).mean()
    rv=float(rsi(c).iloc[-1]); hi=float(h.shift(1).rolling(20).max().iloc[-1])
    pc=c.shift(1); av=float(pd.concat([(d.High-d.Low).abs(),(d.High-pc).abs(),(d.Low-pc).abs()],axis=1).max(axis=1).rolling(14).mean().iloc[-1])
    vr=float(v.iloc[-1]/v.rolling(20).mean().iloc[-1]);score=0;why=[]
    if ltp>s200.iloc[-1]:score+=20;why.append("Above 200-DMA")
    if s20.iloc[-1]>s50.iloc[-1]>s200.iloc[-1]:score+=20;why.append("Trend aligned")
    elif s50.iloc[-1]>s200.iloc[-1]:score+=10
    if ltp>hi:score+=25;why.append("Live 20D breakout")
    elif ltp>=hi*.985:score+=12;why.append("Near breakout")
    if 50<=rv<=68:score+=15;why.append(f"RSI {rv:.0f}")
    elif rv>75:score-=10
    if vr>=1.5:score+=15;why.append("Volume confirmation")
    elif vr>=1.1:score+=8
    mom=ltp/c.iloc[-21]-1
    if mom>.05:score+=10;why.append("20D momentum")
    elif mom>0:score+=5
    score=max(0,min(100,score)); stop=max(ltp-1.5*av,float(s20.iloc[-1])); rps=max(ltp-stop,.01)
    qty=max(0,min(int(capital*risk/100/rps),int(capital/ltp)))
    action="WATCH"
    if score>=75 and ltp>hi and rv<75:action="ENTRY CANDIDATE"
    elif score>=65 and ltp>s20.iloc[-1]:action="HOLD / WATCH"
    elif ltp<s20.iloc[-1]:action="REDUCE / EXIT REVIEW"
    if ltp<stop:action="EXIT CONDITION"
    return {"Symbol":sym,"Live LTP":round(ltp,2),"Score":score,"Action":action,"Stop":round(stop,2),"T1":round(ltp+1.5*rps,2),"T2":round(ltp+2.5*rps,2),"Qty":qty,"RSI":round(rv,1),"Why":" • ".join(why)}

t1,t2,t3=st.tabs(["Live Top Picks","Position Monitor","Connection"])
with t1:
    a,b=st.columns(2);cap=a.number_input("Capital ₹",10000,100000000,100000,10000);risk=b.number_input("Risk per idea %",.25,3.,1.,.25)
    minimum=st.slider("Minimum score",50,90,65,5)
    if st.button("Scan Live NSE"):
        q=kite.ltp([f"NSE:{s}" for s in SYMS]);rows=[]
        for s in SYMS:
            k=f"NSE:{s}"
            if k in q:
                z=analyze(s,float(q[k]["last_price"]),cap,risk)
                if z:rows.append(z)
        df=pd.DataFrame(rows)
        if len(df):
            picks=df[(df.Score>=minimum)&df.Action.isin(["ENTRY CANDIDATE","HOLD / WATCH"])].sort_values("Score",ascending=False).head(5)
            if len(picks):st.dataframe(picks,use_container_width=True,hide_index=True)
            else:st.warning("No qualifying setup now. Do not force a trade.")
            with st.expander("All scanned stocks"):st.dataframe(df.sort_values("Score",ascending=False),use_container_width=True,hide_index=True)
with t2:
    sym=st.selectbox("Stock you hold / paper-track",SYMS)
    if st.button("Check HOLD / EXIT"):
        q=kite.ltp([f"NSE:{sym}"]);z=analyze(sym,float(q[f"NSE:{sym}"]["last_price"]),100000,1)
        if z:
            st.metric("Live LTP",f"₹{z['Live LTP']:,.2f}");st.metric("Score",f"{z['Score']}/100");st.subheader(z["Action"]);st.write(z["Why"]);st.write(f"Stop reference: ₹{z['Stop']:,.2f}")
with t3:
    st.write("Green 'Zerodha connected' means authentication is working.")
    if st.button("Logout ALPHA session"):
        st.session_state.pop("access_token",None);st.rerun()
    st.warning("No automatic orders. News intelligence is not included yet.")
