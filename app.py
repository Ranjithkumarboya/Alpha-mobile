import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta

st.set_page_config(page_title="ALPHA Daily Scanner v1.0", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container{padding-top:.8rem;padding-bottom:4rem;max-width:1150px}
div[data-testid="stMetric"]{background:rgba(128,128,128,.08);padding:10px;border-radius:12px}
.stButton button{width:100%;min-height:46px;border-radius:12px;font-weight:700}
@media(max-width:700px){.block-container{padding-left:.65rem;padding-right:.65rem}h1{font-size:1.65rem!important}}
</style>
""", unsafe_allow_html=True)

st.title("🎯 ALPHA Daily Scanner v1.0")
st.caption("End-of-day NSE swing scanner • ranks only qualifying setups • 0–5 picks")

UNIVERSE = {
"RELIANCE":"RELIANCE.NS","HDFC BANK":"HDFCBANK.NS","ICICI BANK":"ICICIBANK.NS",
"INFOSYS":"INFY.NS","TCS":"TCS.NS","SBIN":"SBIN.NS","ITC":"ITC.NS",
"LT":"LT.NS","BHARTI AIRTEL":"BHARTIARTL.NS","MARUTI":"MARUTI.NS",
"SUN PHARMA":"SUNPHARMA.NS","TATA MOTORS":"TATAMOTORS.NS","AXIS BANK":"AXISBANK.NS",
"KOTAK BANK":"KOTAKBANK.NS","BAJAJ FINANCE":"BAJFINANCE.NS","M&M":"M&M.NS",
"NTPC":"NTPC.NS","POWER GRID":"POWERGRID.NS","TITAN":"TITAN.NS","ASIAN PAINTS":"ASIANPAINT.NS",
"ULTRATECH":"ULTRACEMCO.NS","HCL TECH":"HCLTECH.NS","WIPRO":"WIPRO.NS",
"TECH MAHINDRA":"TECHM.NS","ADANI PORTS":"ADANIPORTS.NS","ONGC":"ONGC.NS",
"COAL INDIA":"COALINDIA.NS","TATA STEEL":"TATASTEEL.NS","JSW STEEL":"JSWSTEEL.NS",
"GRASIM":"GRASIM.NS","CIPLA":"CIPLA.NS","DR REDDY":"DRREDDY.NS",
"APOLLO HOSPITALS":"APOLLOHOSP.NS","EICHER MOTORS":"EICHERMOT.NS","HINDALCO":"HINDALCO.NS"
}

@st.cache_data(ttl=1800)
def download_one(ticker):
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=550)
    x = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x.columns = [str(c).title() for c in x.columns]
    return x.dropna(subset=["Close"]).sort_index()

def rsi(close, n=14):
    d=close.diff()
    gain=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    loss=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-(100/(1+gain/loss.replace(0,np.nan)))

def atr(df,n=14):
    prev=df.Close.shift(1)
    tr=pd.concat([(df.High-df.Low).abs(),(df.High-prev).abs(),(df.Low-prev).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def analyze(name,ticker,capital,risk_pct,min_score):
    try:
        d=download_one(ticker)
    except:
        return None
    if len(d)<220:
        return None

    c=d.Close.astype(float); h=d.High.astype(float); l=d.Low.astype(float)
    v=d.Volume.astype(float) if "Volume" in d else pd.Series(index=d.index,dtype=float)

    sma20=c.rolling(20).mean()
    sma50=c.rolling(50).mean()
    sma200=c.rolling(200).mean()
    rs=rsi(c)
    at=atr(d)
    high20=h.shift(1).rolling(20).max()
    vol20=v.rolling(20).mean()

    px=float(c.iloc[-1]); atrv=float(at.iloc[-1]) if pd.notna(at.iloc[-1]) else 0
    r=float(rs.iloc[-1]) if pd.notna(rs.iloc[-1]) else 50

    score=0
    reasons=[]

    if px > sma200.iloc[-1]:
        score += 20; reasons.append("Above 200-DMA")
    if sma20.iloc[-1] > sma50.iloc[-1] > sma200.iloc[-1]:
        score += 20; reasons.append("20/50/200 trend aligned")
    elif sma50.iloc[-1] > sma200.iloc[-1]:
        score += 10; reasons.append("Medium-term trend positive")

    breakout = pd.notna(high20.iloc[-1]) and px > float(high20.iloc[-1])
    near_breakout = pd.notna(high20.iloc[-1]) and px >= float(high20.iloc[-1]) * .985
    if breakout:
        score += 25; reasons.append("20-day breakout")
    elif near_breakout:
        score += 12; reasons.append("Near 20-day breakout")

    if 50 <= r <= 68:
        score += 15; reasons.append(f"Healthy RSI {r:.0f}")
    elif 45 <= r < 50:
        score += 7
    elif r > 75:
        score -= 10; reasons.append("RSI stretched")

    vol_ratio=np.nan
    if len(v) and pd.notna(vol20.iloc[-1]) and vol20.iloc[-1]>0:
        vol_ratio=float(v.iloc[-1]/vol20.iloc[-1])
        if vol_ratio >= 1.5:
            score += 15; reasons.append(f"Volume {vol_ratio:.1f}× avg")
        elif vol_ratio >= 1.1:
            score += 8; reasons.append("Volume confirmation")

    mom20=float(c.iloc[-1]/c.iloc[-21]-1) if len(c)>21 else 0
    if mom20 > .05:
        score += 10; reasons.append(f"20D momentum {mom20:.1%}")
    elif mom20 > 0:
        score += 5

    score=max(0,min(100,score))

    # ATR-based research levels for next-session planning
    entry=px
    stop=max(px-1.5*atrv, float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else px-1.5*atrv)
    risk_per_share=max(entry-stop, .01)
    target1=entry+1.5*risk_per_share
    target2=entry+2.5*risk_per_share
    rupees_risk=capital*(risk_pct/100)
    qty=int(rupees_risk/risk_per_share) if risk_per_share>0 else 0
    max_cash_qty=int(capital/entry) if entry>0 else 0
    qty=max(0,min(qty,max_cash_qty))

    qualified = score >= min_score and px > sma200.iloc[-1] and r < 75
    return {
        "Stock":name,"Ticker":ticker,"Score":int(score),"Close":round(px,2),
        "RSI":round(r,1),"Volume x":None if pd.isna(vol_ratio) else round(vol_ratio,2),
        "20D %":round(mom20*100,2),"Entry zone":round(entry,2),"Stop":round(stop,2),
        "Target 1":round(target1,2),"Target 2":round(target2,2),"Qty":qty,
        "Risk ₹":round(qty*risk_per_share,0),"Qualified":qualified,
        "Why":" • ".join(reasons[:5]),"Data date":str(d.index[-1].date())
    }

tab1,tab2,tab3=st.tabs(["Today's Picks","Stock Detail","Rules"])

with tab1:
    st.subheader("Daily Top Opportunities")
    st.info("Run this after market close for next-session swing planning. The app may return fewer than 5 stocks.")
    a,b=st.columns(2)
    capital=a.number_input("Trading capital (₹)",min_value=10000,value=100000,step=10000)
    risk=b.number_input("Max risk per idea (%)",min_value=.25,max_value=3.0,value=1.0,step=.25)
    min_score=st.slider("Minimum ALPHA score",50,90,65,5)
    max_picks=st.slider("Maximum picks",1,5,5)

    if st.button("Scan NSE Universe"):
        bar=st.progress(0)
        rows=[]
        items=list(UNIVERSE.items())
        for i,(name,ticker) in enumerate(items):
            r=analyze(name,ticker,float(capital),float(risk),int(min_score))
            if r: rows.append(r)
            bar.progress((i+1)/len(items))
        df=pd.DataFrame(rows)
        if df.empty:
            st.error("Market data could not be loaded.")
        else:
            picks=df[df.Qualified].sort_values(["Score","20D %"],ascending=False).head(max_picks)
            if picks.empty:
                st.warning("NO TRADE / NO QUALIFYING SETUPS. ALPHA will not force five picks.")
            else:
                st.success(f"{len(picks)} setup(s) passed today's filter.")
                for rank,(_,r) in enumerate(picks.iterrows(),1):
                    st.markdown(f"### #{rank} {r['Stock']} — {r['Score']}/100")
                    a,b,c=st.columns(3)
                    a.metric("Close",f"₹{r['Close']:,.2f}")
                    b.metric("Stop",f"₹{r['Stop']:,.2f}")
                    c.metric("Qty",f"{int(r['Qty'])}")
                    st.write(f"**Planning levels:** Entry around ₹{r['Entry zone']:,.2f} | T1 ₹{r['Target 1']:,.2f} | T2 ₹{r['Target 2']:,.2f}")
                    st.write(f"**Why:** {r['Why']}")
                    st.caption(f"Data: {r['Data date']} • Estimated position risk: ₹{r['Risk ₹']:,.0f}")
                    st.divider()

                st.subheader("Ranking table")
                show=["Stock","Score","Close","RSI","Volume x","20D %","Entry zone","Stop","Target 1","Target 2","Qty","Risk ₹","Data date"]
                st.dataframe(picks[show],use_container_width=True,hide_index=True)
                st.download_button("Download today's shortlist CSV",picks.to_csv(index=False),file_name="alpha_daily_picks.csv",mime="text/csv")

            with st.expander("See all scanned stocks"):
                st.dataframe(df.sort_values("Score",ascending=False),use_container_width=True,hide_index=True)

with tab2:
    st.subheader("Inspect one stock")
    name=st.selectbox("Stock",list(UNIVERSE.keys()),key="detail")
    if st.button("Analyze Stock"):
        r=analyze(name,UNIVERSE[name],100000,1.0,65)
        d=download_one(UNIVERSE[name])
        if r:
            st.metric("ALPHA score",f"{r['Score']}/100")
            st.write(r["Why"])
            st.line_chart(d.Close.tail(180))
            st.json({k:v for k,v in r.items() if k not in ["Why","Qualified"]})

with tab3:
    st.markdown("""
**Current scoring model**
- Long-term trend: price vs 200-DMA
- Trend alignment: 20/50/200-DMA
- 20-day breakout / near-breakout
- RSI quality filter
- Volume confirmation
- 20-day momentum
- ATR-based stop and risk-sized quantity

**Important:** this is an end-of-day swing scanner, not an intraday prediction engine. Entry/stop/targets are mechanical planning levels based on the latest available daily candle, not guaranteed prices.
""")
    st.warning("Do not buy a stock merely because it ranks #1. Gap-ups, news, liquidity and next-session price action can invalidate the setup. This version is for paper trading/controlled research first.")
