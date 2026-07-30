import streamlit as st
import pandas as pd, numpy as np
st.set_page_config(page_title="ALPHA Mobile",page_icon="📈",layout="wide",initial_sidebar_state="collapsed")
st.markdown("""<style>.block-container{padding-top:1rem;padding-bottom:4rem;max-width:1000px}.stButton button{width:100%;min-height:46px;border-radius:12px}div[data-testid="stMetric"]{padding:10px;border-radius:12px;background:rgba(128,128,128,.08)}@media(max-width:700px){.block-container{padding-left:.7rem;padding-right:.7rem}h1{font-size:1.8rem!important}}</style>""",unsafe_allow_html=True)
st.title("ALPHA Mobile"); st.caption("Quant research lab • research first, real money later")
u=st.file_uploader("Upload daily OHLCV CSV",type=["csv"])
s=st.selectbox("Strategy",["SMA Crossover","Donchian Breakout","RSI Mean Reversion","Buy & Hold"])
capital=st.number_input("Starting capital (₹)",10000,100000000,100000,10000)
cost=st.number_input("Estimated cost per position change (bps)",0.0,100.0,10.0,1.0)
if s=="SMA Crossover":
 c1,c2=st.columns(2); fast=c1.number_input("Fast SMA",2,200,20); slow=c2.number_input("Slow SMA",3,500,100)
elif s=="Donchian Breakout": lookback=st.number_input("Lookback",2,250,55)
elif s=="RSI Mean Reversion":
 c1,c2=st.columns(2); rn=c1.number_input("RSI period",2,100,14); entry=c2.number_input("Entry below RSI",1,49,30)
if u:
 df=pd.read_csv(u); df.columns=[x.strip().title() for x in df.columns]
 if not {"Date","Open","High","Low","Close"}.issubset(df.columns): st.error("Need Date, Open, High, Low, Close columns.")
 else:
  df["Date"]=pd.to_datetime(df["Date"],errors="coerce"); df=df.dropna(subset=["Date","Close"]).sort_values("Date").drop_duplicates("Date").set_index("Date")
  close=df.Close.astype(float); ret=close.pct_change().fillna(0)
  if s=="Buy & Hold": sig=pd.Series(1.,index=df.index)
  elif s=="SMA Crossover": sig=(close.rolling(fast).mean()>close.rolling(slow).mean()).astype(float)
  elif s=="Donchian Breakout":
   upper=df.High.astype(float).shift(1).rolling(lookback).max(); lower=df.Low.astype(float).shift(1).rolling(20).min(); pos=0; a=[]
   for i in range(len(df)):
    if pd.notna(upper.iloc[i]) and close.iloc[i]>upper.iloc[i]: pos=1
    if pd.notna(lower.iloc[i]) and close.iloc[i]<lower.iloc[i]: pos=0
    a.append(pos)
   sig=pd.Series(a,index=df.index,dtype=float)
  else:
   d=close.diff(); g=d.clip(lower=0).ewm(alpha=1/rn,adjust=False).mean(); l=(-d.clip(upper=0)).ewm(alpha=1/rn,adjust=False).mean()
   rsi=100-100/(1+g/l.replace(0,np.nan)); pos=0;a=[]
   for x in rsi:
    if pd.notna(x) and x<entry: pos=1
    elif pd.notna(x) and x>50: pos=0
    a.append(pos)
   sig=pd.Series(a,index=df.index,dtype=float)
  p=sig.shift(1).fillna(0); turn=p.diff().abs().fillna(p.abs()); sr=p*ret-turn*cost/10000
  eq=capital*(1+sr).cumprod(); bh=capital*(1+ret).cumprod(); dd=eq/eq.cummax()-1
  yrs=max((df.index[-1]-df.index[0]).days/365.25,1/365.25); cagr=(eq.iloc[-1]/capital)**(1/yrs)-1
  vol=sr.std()*np.sqrt(252); sharpe=sr.mean()*252/vol if vol>0 else np.nan
  if st.button("Run Backtest"):
   a,b=st.columns(2);a.metric("Ending capital",f"₹{eq.iloc[-1]:,.0f}");b.metric("CAGR",f"{cagr:.2%}")
   c,d=st.columns(2);c.metric("Max drawdown",f"{dd.min():.2%}");d.metric("Sharpe","—" if np.isnan(sharpe) else f"{sharpe:.2f}")
   st.metric("Exposure",f"{p.mean():.1%}");st.line_chart(pd.DataFrame({"Strategy":eq,"Buy & Hold":bh}))
   st.warning("Research MVP only. Do not trade real money from these results yet.")
