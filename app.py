import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date

st.set_page_config(page_title="ALPHA Mobile v0.3", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container{padding-top:1rem;padding-bottom:4rem;max-width:1050px}
.stButton button{width:100%;min-height:46px;border-radius:12px;font-weight:700}
div[data-testid="stMetric"]{padding:10px;border-radius:12px;background:rgba(128,128,128,.08)}
@media(max-width:700px){.block-container{padding-left:.7rem;padding-right:.7rem}h1{font-size:1.8rem!important}}
</style>
""", unsafe_allow_html=True)

st.title("ALPHA Mobile v0.3")
st.caption("Automatic market data • Backtest • Benchmark • Out-of-sample check")

PRESETS = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY MIDCAP 100": "^CNXMDCP",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
}

def fetch_data(ticker, start, end):
    x = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x.columns = [str(c).title() for c in x.columns]
    return x.dropna(subset=["Close"]).sort_index()

def make_signal(df, strategy, p):
    close = df["Close"].astype(float)
    if strategy == "Buy & Hold":
        return pd.Series(1.0, index=df.index)
    if strategy == "SMA Crossover":
        return (close.rolling(p["fast"]).mean() > close.rolling(p["slow"]).mean()).astype(float)
    if strategy == "Donchian Breakout":
        upper = df["High"].astype(float).shift(1).rolling(p["lookback"]).max()
        lower = df["Low"].astype(float).shift(1).rolling(p["exit_lb"]).min()
        pos=0; vals=[]
        for i in range(len(df)):
            if pd.notna(upper.iloc[i]) and close.iloc[i] > upper.iloc[i]: pos=1
            if pd.notna(lower.iloc[i]) and close.iloc[i] < lower.iloc[i]: pos=0
            vals.append(pos)
        return pd.Series(vals,index=df.index,dtype=float)
    delta=close.diff()
    gain=delta.clip(lower=0).ewm(alpha=1/p["rsi_n"],adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/p["rsi_n"],adjust=False).mean()
    rsi=100-(100/(1+gain/loss.replace(0,np.nan)))
    pos=0; vals=[]
    for x in rsi:
        if pd.notna(x) and x < p["entry"]: pos=1
        elif pd.notna(x) and x > p["exit"]: pos=0
        vals.append(pos)
    return pd.Series(vals,index=df.index,dtype=float)

def metrics(df, signal, capital, cost_bps):
    close=df["Close"].astype(float)
    ret=close.pct_change().fillna(0)
    pos=signal.shift(1).fillna(0)
    turnover=pos.diff().abs().fillna(pos.abs())
    sr=pos*ret-turnover*(cost_bps/10000)
    eq=capital*(1+sr).cumprod()
    bh=capital*(1+ret).cumprod()
    dd=eq/eq.cummax()-1
    years=max((df.index[-1]-df.index[0]).days/365.25,1/365.25)
    cagr=(eq.iloc[-1]/capital)**(1/years)-1
    vol=sr.std()*np.sqrt(252)
    sharpe=(sr.mean()*252/vol) if vol>0 else np.nan
    downside=sr[sr<0].std()*np.sqrt(252)
    sortino=(sr.mean()*252/downside) if pd.notna(downside) and downside>0 else np.nan
    trades=int((pos.diff().abs()>0).sum())
    return {"eq":eq,"bh":bh,"dd":dd,"cagr":cagr,"maxdd":dd.min(),"sharpe":sharpe,
            "sortino":sortino,"exposure":pos.mean(),"trades":trades,"ending":eq.iloc[-1]}

tab1, tab2 = st.tabs(["Research", "How to read"])

with tab1:
    mode=st.radio("Market", ["Preset","Custom NSE ticker"], horizontal=True)
    if mode=="Preset":
        label=st.selectbox("Select index / stock", list(PRESETS.keys()))
        ticker=PRESETS[label]
    else:
        raw=st.text_input("NSE symbol", value="ITC", help="Example: ITC, TATAMOTORS, MARUTI")
        ticker=raw.strip().upper()
        if ticker and not ticker.startswith("^") and not ticker.endswith(".NS"): ticker += ".NS"

    c1,c2=st.columns(2)
    start=c1.date_input("Start date", value=date(2015,1,1))
    end=c2.date_input("End date", value=date.today())
    strategy=st.selectbox("Strategy", ["SMA Crossover","Donchian Breakout","RSI Mean Reversion","Buy & Hold"])
    capital=st.number_input("Starting capital (₹)",10000,100000000,100000,10000)
    cost=st.number_input("Estimated cost per position change (bps)",0.0,100.0,10.0,1.0)
    oos=st.slider("Out-of-sample portion", 20, 50, 30, 5, help="Last part of history is shown separately as a reality check.")

    p={}
    if strategy=="SMA Crossover":
        a,b=st.columns(2); p["fast"]=a.number_input("Fast SMA",2,200,20); p["slow"]=b.number_input("Slow SMA",3,500,100)
        if p["fast"]>=p["slow"]: st.warning("Fast SMA should normally be below Slow SMA.")
    elif strategy=="Donchian Breakout":
        a,b=st.columns(2); p["lookback"]=a.number_input("Entry lookback",2,250,55); p["exit_lb"]=b.number_input("Exit lookback",2,250,20)
    elif strategy=="RSI Mean Reversion":
        a,b,c=st.columns(3); p["rsi_n"]=a.number_input("RSI period",2,100,14); p["entry"]=b.number_input("Entry RSI",1,49,30); p["exit"]=c.number_input("Exit RSI",40,99,50)

    if st.button("Fetch Data & Run Research"):
        if start>=end:
            st.error("Start date must be before end date.")
        else:
            with st.spinner("Fetching market data and running backtest..."):
                try:
                    df=fetch_data(ticker,start,end)
                except Exception as e:
                    df=pd.DataFrame()
                    st.error(f"Data fetch failed: {e}")
            if df.empty or len(df)<100:
                st.error("Not enough data returned. Check ticker/date range and try again.")
            else:
                sig=make_signal(df,strategy,p)
                m=metrics(df,sig,float(capital),float(cost))
                split=max(30,int(len(df)*(1-oos/100)))
                oos_df=df.iloc[split:].copy()
                oos_sig=sig.iloc[split:].copy()
                om=metrics(oos_df,oos_sig,float(capital),float(cost)) if len(oos_df)>30 else None

                st.success(f"Loaded {len(df):,} daily bars for {ticker}")
                a,b=st.columns(2); a.metric("Ending capital",f"₹{m['ending']:,.0f}"); b.metric("CAGR",f"{m['cagr']:.2%}")
                c,d=st.columns(2); c.metric("Max drawdown",f"{m['maxdd']:.2%}"); d.metric("Sharpe","—" if np.isnan(m["sharpe"]) else f"{m['sharpe']:.2f}")
                e,f=st.columns(2); e.metric("Trades / position changes",f"{m['trades']}"); f.metric("Exposure",f"{m['exposure']:.1%}")
                st.subheader("Strategy vs Buy & Hold")
                st.line_chart(pd.DataFrame({"Strategy":m["eq"],"Buy & Hold":m["bh"]}))

                if om:
                    st.subheader(f"Out-of-sample check — last {oos}%")
                    a,b,c=st.columns(3)
                    a.metric("OOS CAGR",f"{om['cagr']:.2%}")
                    b.metric("OOS Max DD",f"{om['maxdd']:.2%}")
                    c.metric("OOS Sharpe","—" if np.isnan(om["sharpe"]) else f"{om['sharpe']:.2f}")
                    st.line_chart(pd.DataFrame({"OOS Strategy":om["eq"],"OOS Buy & Hold":om["bh"]}))
                    if om["cagr"] < 0:
                        st.error("OOS return is negative. Treat this strategy as failed until deeper validation says otherwise.")
                    elif pd.notna(om["sharpe"]) and om["sharpe"] < 0.5:
                        st.warning("OOS risk-adjusted performance is weak. Do not treat this as a validated edge.")
                    else:
                        st.info("This is only a first OOS screen, not proof of an edge. Walk-forward and robustness tests are still required.")

                st.caption("Adjusted daily data is fetched automatically. Signals are shifted one bar. Cost model is simplified.")

with tab2:
    st.write("**CAGR** = compounded annual return. **Max drawdown** = worst peak-to-trough fall. **Sharpe** = return relative to volatility. **OOS** = the most recent portion of history kept separate as a basic reality check.")
    st.warning("ALPHA v0.3 is research software. It does not generate a live buy/sell recommendation and should not be used for real-money trading yet.")
