import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from kiteconnect import KiteConnect

st.set_page_config(page_title="ALPHA Live v1.2", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")
st.title("🎯 ALPHA Live v1.2")
st.caption("Live Zerodha decision-support • LONG / INTRADAY SHORT / NO-TRADE • manual execution")

try:
    KEY = st.secrets["KITE_API_KEY"]
    SECRET = st.secrets["KITE_API_SECRET"]
except Exception:
    st.error("KITE_API_KEY / KITE_API_SECRET missing in Streamlit Secrets.")
    st.stop()

kite = KiteConnect(api_key=KEY)

# ---- Auth ----
if "access_token" not in st.session_state:
    rt = st.query_params.get("request_token")
    if rt:
        try:
            s = kite.generate_session(rt, api_secret=SECRET)
            st.session_state.access_token = s["access_token"]
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Zerodha authentication failed: {e}")

if "access_token" not in st.session_state:
    st.info("Daily Zerodha login required.")
    st.link_button("Login with Zerodha", kite.login_url(), use_container_width=True)
    st.stop()

kite.set_access_token(st.session_state.access_token)
try:
    profile = kite.profile()
except Exception:
    st.session_state.pop("access_token", None)
    st.warning("Session expired. Login again.")
    st.rerun()

st.success(f"Zerodha connected • {profile.get('user_shortname') or profile.get('user_id')}")

SYMS = [
"RELIANCE","HDFCBANK","ICICIBANK","INFY","TCS","SBIN","ITC","LT","BHARTIARTL",
"MARUTI","SUNPHARMA","TATAMOTORS","AXISBANK","KOTAKBANK","BAJFINANCE","M&M",
"NTPC","POWERGRID","TITAN","HCLTECH","WIPRO","TECHM","ADANIPORTS","ONGC",
"COALINDIA","TATASTEEL","JSWSTEEL","CIPLA","DRREDDY","APOLLOHOSP"
]

@st.cache_data(ttl=86400)
def instrument_map():
    return {x["tradingsymbol"]: x["instrument_token"] for x in kite.instruments("NSE")}

def hist(sym, interval="day", days=550):
    tok = instrument_map().get(sym)
    if not tok: return pd.DataFrame()
    raw = kite.historical_data(tok, datetime.now()-timedelta(days=days), datetime.now(), interval)
    d = pd.DataFrame(raw)
    if len(d):
        d = d.rename(columns=str.title).set_index("Date").sort_index()
    return d

def rsi(c, n=14):
    x=c.diff()
    g=x.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l=(-x.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100-(100/(1+g/l.replace(0,np.nan)))

def atr(d,n=14):
    pc=d.Close.shift(1)
    tr=pd.concat([(d.High-d.Low).abs(),(d.High-pc).abs(),(d.Low-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def nifty_regime():
    try:
        d=hist("NIFTY 50","day",365)
        if len(d)<200:
            # index token may not be in NSE cash instrument dump; fallback quote-based neutral
            return "SELECTIVE", 0
        c=d.Close.astype(float); s20=c.rolling(20).mean(); s50=c.rolling(50).mean(); s200=c.rolling(200).mean()
        px=float(c.iloc[-1])
        if px>s20.iloc[-1]>s50.iloc[-1]>s200.iloc[-1]: return "LONG BIAS", 1
        if px<s20.iloc[-1]<s50.iloc[-1]<s200.iloc[-1]: return "SHORT BIAS", -1
        return "SELECTIVE", 0
    except:
        return "SELECTIVE", 0

def score_stock(sym, ltp, capital, risk_pct):
    d=hist(sym,"day",550)
    if len(d)<210:return None
    c=d.Close.astype(float); h=d.High.astype(float); lo=d.Low.astype(float); v=d.Volume.astype(float)
    s20=c.rolling(20).mean(); s50=c.rolling(50).mean(); s200=c.rolling(200).mean()
    rv=float(rsi(c).iloc[-1]); av=float(atr(d).iloc[-1])
    hi=float(h.shift(1).rolling(20).max().iloc[-1]); low20=float(lo.shift(1).rolling(20).min().iloc[-1])
    volavg=float(v.rolling(20).mean().iloc[-1]); vr=float(v.iloc[-1]/volavg) if volavg>0 else 0
    mom=float(ltp/c.iloc[-21]-1)

    long=0; short=0; lw=[]; sw=[]
    if ltp>s200.iloc[-1]: long+=20; lw.append("Above 200-DMA")
    if ltp<s200.iloc[-1]: short+=20; sw.append("Below 200-DMA")
    if s20.iloc[-1]>s50.iloc[-1]>s200.iloc[-1]: long+=20; lw.append("Bull trend aligned")
    if s20.iloc[-1]<s50.iloc[-1]<s200.iloc[-1]: short+=20; sw.append("Bear trend aligned")
    if ltp>hi: long+=25; lw.append("20D breakout")
    elif ltp>=hi*.985: long+=12; lw.append("Near breakout")
    if ltp<low20: short+=25; sw.append("20D breakdown")
    elif ltp<=low20*1.015: short+=12; sw.append("Near breakdown")
    if 50<=rv<=68: long+=15; lw.append(f"RSI {rv:.0f}")
    if 32<=rv<=50: short+=15; sw.append(f"RSI {rv:.0f}")
    if rv>75: long-=10
    if rv<25: short-=10
    if vr>=1.5:
        if mom>=0: long+=15; lw.append("Strong volume")
        else: short+=15; sw.append("Strong selling volume")
    elif vr>=1.1:
        if mom>=0: long+=8
        else: short+=8
    if mom>.05: long+=10; lw.append("20D positive momentum")
    elif mom>0: long+=5
    if mom<-.05: short+=10; sw.append("20D negative momentum")
    elif mom<0: short+=5

    long=max(0,min(100,long)); short=max(0,min(100,short))
    direction="LONG" if long>=short else "SHORT"
    score=max(long,short)
    if direction=="LONG":
        entry=ltp; stop=max(ltp-1.5*av,float(s20.iloc[-1]))
        rps=max(entry-stop,.01); t1=entry+1.5*rps; t2=entry+2.5*rps; why=lw
    else:
        entry=ltp; stop=min(ltp+1.5*av,float(s20.iloc[-1]))
        rps=max(stop-entry,.01); t1=entry-1.5*rps; t2=entry-2.5*rps; why=sw

    qty=max(0,min(int(capital*risk_pct/100/rps), int(capital/max(entry,.01))))
    maxloss=qty*rps
    p1=qty*1.5*rps; p2=qty*2.5*rps
    return {
        "Symbol":sym,"Direction":direction,"Score":int(score),"LONG":int(long),"SHORT":int(short),
        "Live LTP":round(ltp,2),"Entry":round(entry,2),"Stop":round(stop,2),
        "T1":round(t1,2),"T2":round(t2,2),"Qty":qty,
        "Planned loss ₹":round(maxloss,0),"Profit if T1 ₹":round(p1,0),"Profit if T2 ₹":round(p2,0),
        "R:R T1":"1:1.5","R:R T2":"1:2.5","RSI":round(rv,1),"Vol x":round(vr,2),
        "Why":" • ".join(why[:5])
    }

def intraday_confirm(sym, direction):
    # Confirmation only; daily model remains primary.
    try:
        d=hist(sym,"15minute",10)
        if len(d)<30:return False,"Insufficient 15m data"
        c=d.Close.astype(float); v=d.Volume.astype(float)
        ema20=c.ewm(span=20,adjust=False).mean()
        px=float(c.iloc[-1])
        vol_ok=float(v.iloc[-1]) >= float(v.tail(20).mean())*.8
        if direction=="LONG":
            ok=px>ema20.iloc[-1] and c.iloc[-1]>=c.iloc[-2] and vol_ok
            return ok, "15m price above EMA20" if ok else "15m LONG confirmation weak"
        ok=px<ema20.iloc[-1] and c.iloc[-1]<=c.iloc[-2] and vol_ok
        return ok, "15m price below EMA20" if ok else "15m SHORT confirmation weak"
    except Exception:
        return False,"15m confirmation unavailable"

tab1,tab2,tab3,tab4=st.tabs(["Market + Setups","Trade Calculator","Position Monitor","Rules"])

with tab1:
    a,b=st.columns(2)
    capital=a.number_input("Trading capital (₹)",10000,100000000,100000,10000)
    risk=b.number_input("Risk per trade (%)",.25,2.0,1.0,.25)
    minscore=st.slider("Minimum setup score",55,90,70,5)

    if st.button("Run ALPHA Live Decision Engine",use_container_width=True):
        regime,bias=nifty_regime()
        q=kite.ltp([f"NSE:{s}" for s in SYMS])
        rows=[]
        bar=st.progress(0)
        for i,s in enumerate(SYMS):
            k=f"NSE:{s}"
            if k in q:
                z=score_stock(s,float(q[k]["last_price"]),capital,risk)
                if z:
                    ok,msg=intraday_confirm(s,z["Direction"])
                    z["15m Confirm"]=ok; z["Intraday note"]=msg
                    rows.append(z)
            bar.progress((i+1)/len(SYMS))
        df=pd.DataFrame(rows)
        if not len(df):
            st.error("No market data returned."); st.stop()

        # Regime-aware filter. Shorts require intraday confirmation because NSE cash short is an intraday workflow.
        candidates=df[df.Score>=minscore].copy()
        candidates=candidates[candidates["15m Confirm"]==True]
        if regime=="LONG BIAS":
            candidates=candidates[(candidates.Direction=="LONG") | (candidates.Score>=85)]
        elif regime=="SHORT BIAS":
            candidates=candidates[(candidates.Direction=="SHORT") | (candidates.Score>=85)]

        longs=candidates[candidates.Direction=="LONG"]
        shorts=candidates[candidates.Direction=="SHORT"]
        breadth=(df.Direction=="LONG").mean()

        # NO-TRADE is explicit, not forced.
        if len(candidates)==0:
            day="🔴 BAD TRADE TODAY — NO TRADE"
        elif regime=="SELECTIVE" or (0.4<breadth<0.6):
            day="🟡 SELECTIVE DAY — ONLY HIGH-QUALITY SETUPS"
        else:
            day="🟢 TRADEABLE DAY — FOLLOW RISK RULES"

        st.subheader(day)
        st.write(f"**Market regime:** {regime} | **Qualified LONG:** {len(longs)} | **Qualified SHORT:** {len(shorts)}")
        picks=candidates.sort_values("Score",ascending=False).head(5)

        if len(picks):
            for _,r in picks.iterrows():
                icon="🟢" if r.Direction=="LONG" else "🔴"
                st.markdown(f"### {icon} {r.Symbol} — {r.Direction} — {r.Score}/100")
                c1,c2,c3=st.columns(3)
                c1.metric("Entry reference",f"₹{r.Entry:,.2f}")
                c2.metric("Stop",f"₹{r.Stop:,.2f}")
                c3.metric("Qty",f"{int(r.Qty)}")
                st.write(f"**T1:** ₹{r['T1']:,.2f} → profit if hit ≈ ₹{r['Profit if T1 ₹']:,.0f}  |  **T2:** ₹{r['T2']:,.2f} → profit if hit ≈ ₹{r['Profit if T2 ₹']:,.0f}")
                st.write(f"**Planned loss near SL:** ₹{r['Planned loss ₹']:,.0f} | **R:R:** T1 {r['R:R T1']} • T2 {r['R:R T2']}")
                st.write(f"**Strategy evidence:** {r.Why}")
                if r.Direction=="SHORT":
                    st.caption("SHORT = intraday cash-equity research signal only. Confirm broker product/eligibility and square-off rules before execution.")
                st.divider()
        else:
            st.warning("ALPHA found no setup meeting the current filters. Staying in cash is the strategy.")

        with st.expander("Full scanner"):
            st.dataframe(df.sort_values("Score",ascending=False),use_container_width=True,hide_index=True)

with tab2:
    st.subheader("Exact risk / reward calculator")
    direction=st.radio("Direction",["LONG","SHORT"],horizontal=True)
    entry=st.number_input("Entry price",min_value=.01,value=1000.0)
    stop=st.number_input("Stop-loss",min_value=.01,value=980.0 if direction=="LONG" else 1020.0)
    target=st.number_input("Target",min_value=.01,value=1040.0 if direction=="LONG" else 960.0)
    qty=st.number_input("Quantity",min_value=1,value=10)
    riskps=(entry-stop) if direction=="LONG" else (stop-entry)
    rewardps=(target-entry) if direction=="LONG" else (entry-target)
    if riskps<=0 or rewardps<=0:
        st.error("Entry/SL/target geometry is invalid for this direction.")
    else:
        c1,c2,c3=st.columns(3)
        c1.metric("Planned loss",f"₹{riskps*qty:,.0f}")
        c2.metric("Profit if target hits",f"₹{rewardps*qty:,.0f}")
        c3.metric("Risk : Reward",f"1 : {rewardps/riskps:.2f}")

with tab3:
    st.subheader("Live Zerodha positions")
    try:
        pos=kite.positions().get("net",[])
        live=[p for p in pos if p.get("quantity",0)!=0]
        if not live:
            st.info("No open net positions detected.")
        for p in live:
            sym=p["tradingsymbol"]; qty=p["quantity"]; avg=float(p["average_price"])
            try:
                ltp=float(kite.ltp([f"NSE:{sym}"])[f"NSE:{sym}"]["last_price"])
                z=score_stock(sym,ltp,100000,1.0)
            except: z=None; ltp=0
            pnl=(ltp-avg)*qty
            st.markdown(f"### {sym} • Qty {qty}")
            c1,c2=st.columns(2);c1.metric("Live LTP",f"₹{ltp:,.2f}");c2.metric("Approx P&L",f"₹{pnl:,.0f}")
            if z:
                held_dir="LONG" if qty>0 else "SHORT"
                if held_dir==z["Direction"] and z["Score"]>=70:
                    action="HOLD / MANAGE"
                elif z["Score"]<60 or held_dir!=z["Direction"]:
                    action="EXIT / REDUCE REVIEW"
                else: action="TIGHTEN RISK / WATCH"
                st.write(f"**ALPHA:** {action} | Current model: {z['Direction']} {z['Score']}/100")
            st.divider()
    except Exception as e:
        st.error(f"Could not read positions: {e}")

with tab4:
    st.markdown("""
**v1.2 logic**
- Separate LONG and SHORT scores; SHORT is not merely an inverted BUY button.
- Daily trend/breakout/breakdown/RSI/volume/momentum model.
- 15-minute confirmation before a setup qualifies.
- Explicit NO-TRADE state when nothing passes.
- Position sizing from capital and planned % risk.
- Profit is shown as **profit if target hits**, never as guaranteed expected profit.
- Current v1.2 does **not** claim historical expectancy because that requires a proper out-of-sample backtest database.
- No automatic order placement.
- News/event intelligence remains the next independent layer; it should not be faked with simple positive/negative keywords.
""")
    if st.button("Logout Zerodha session"):
        st.session_state.pop("access_token",None);st.rerun()
