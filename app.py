import streamlit as st
import pandas as pd, numpy as np, yfinance as yf, sqlite3, json, hashlib
from datetime import date, datetime
from pathlib import Path

st.set_page_config(page_title="ALPHA Mobile v0.6",page_icon="📈",layout="wide",initial_sidebar_state="collapsed")
st.markdown("""<style>.block-container{padding-top:1rem;padding-bottom:4rem;max-width:1180px}.stButton button{width:100%;min-height:46px;border-radius:12px;font-weight:700}div[data-testid="stMetric"]{padding:10px;border-radius:12px;background:rgba(128,128,128,.08)}@media(max-width:700px){.block-container{padding-left:.6rem;padding-right:.6rem}h1{font-size:1.7rem!important}}</style>""",unsafe_allow_html=True)
st.title("ALPHA Mobile v0.6")
st.caption("Research database • walk-forward library • universe lab • forward paper journal")

DB=Path("alpha.db")
def con(): return sqlite3.connect(DB,check_same_thread=False)
def init():
    c=con()
    c.execute("""create table if not exists experiments(id text primary key,created text,ticker text,strategy text,params text,start text,end text,cost real,holdout_cagr real,holdout_dd real,holdout_sharpe real,status text)""")
    c.execute("""create table if not exists paper_signals(id integer primary key autoincrement,run_time text,data_date text,ticker text,strategy text,params text,state text,close real)""")
    c.commit();c.close()
init()

PRE={"NIFTY 50":"^NSEI","NIFTY BANK":"^NSEBANK","RELIANCE":"RELIANCE.NS","TCS":"TCS.NS","INFOSYS":"INFY.NS","HDFC BANK":"HDFCBANK.NS","ICICI BANK":"ICICIBANK.NS","SBIN":"SBIN.NS","ITC":"ITC.NS","TATA MOTORS":"TATAMOTORS.NS","MARUTI":"MARUTI.NS","LT":"LT.NS","SUN PHARMA":"SUNPHARMA.NS","BHARTI AIRTEL":"BHARTIARTL.NS"}
NIFTY50_SAMPLE=["RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","TCS.NS","LT.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","MARUTI.NS","SUNPHARMA.NS","TATAMOTORS.NS"]

@st.cache_data(ttl=3600)
def fetch(t,s,e):
    x=yf.download(t,start=s,end=e,auto_adjust=True,progress=False,threads=False)
    if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
    x.columns=[str(c).title() for c in x.columns]
    return x.dropna(subset=["Close"]).sort_index()

def sig(df,s,p):
    c=df.Close.astype(float)
    if s=="Buy & Hold":return pd.Series(1.,index=df.index)
    if s=="SMA Crossover":return (c.rolling(int(p["fast"])).mean()>c.rolling(int(p["slow"])).mean()).astype(float)
    if s=="Donchian Breakout":
        u=df.High.astype(float).shift(1).rolling(int(p["entry_lb"])).max();l=df.Low.astype(float).shift(1).rolling(int(p["exit_lb"])).min();pos=0;a=[]
        for i in range(len(df)):
            if pd.notna(u.iloc[i]) and c.iloc[i]>u.iloc[i]:pos=1
            if pd.notna(l.iloc[i]) and c.iloc[i]<l.iloc[i]:pos=0
            a.append(pos)
        return pd.Series(a,index=df.index,dtype=float)
    d=c.diff();g=d.clip(lower=0).ewm(alpha=1/int(p["rsi_n"]),adjust=False).mean();l=(-d.clip(upper=0)).ewm(alpha=1/int(p["rsi_n"]),adjust=False).mean();r=100-100/(1+g/l.replace(0,np.nan));pos=0;a=[]
    for x in r:
        if pd.notna(x) and x<float(p["entry"]):pos=1
        elif pd.notna(x) and x>float(p["exit"]):pos=0
        a.append(pos)
    return pd.Series(a,index=df.index,dtype=float)

def run(df,s,p,cap,cost):
    sg=sig(df,s,p);r=df.Close.astype(float).pct_change().fillna(0);pos=sg.shift(1).fillna(0);chg=pos.diff().fillna(pos);sr=pos*r-chg.abs()*cost/10000
    eq=cap*(1+sr).cumprod();dd=eq/eq.cummax()-1;yrs=max((df.index[-1]-df.index[0]).days/365.25,1/365.25);cg=(eq.iloc[-1]/cap)**(1/yrs)-1;v=sr.std()*np.sqrt(252);sh=sr.mean()*252/v if v>0 else np.nan
    return {"returns":sr,"eq":eq,"cagr":cg,"dd":float(dd.min()),"sharpe":sh,"signal":sg,"close":float(df.Close.iloc[-1])}

def candidates(s,p):
    if s=="SMA Crossover":
        fs=sorted(set([max(2,int(p["fast"]*.75)),int(p["fast"]),int(p["fast"]*1.25)]));ss=sorted(set([max(3,int(p["slow"]*.8)),int(p["slow"]),int(p["slow"]*1.2)]))
        return [{"fast":f,"slow":q} for f in fs for q in ss if f<q]
    if s=="Donchian Breakout":
        es=sorted(set([max(5,int(p["entry_lb"]*.75)),int(p["entry_lb"]),int(p["entry_lb"]*1.25)]));xs=sorted(set([max(3,int(p["exit_lb"]*.75)),int(p["exit_lb"]),int(p["exit_lb"]*1.25)]))
        return [{"entry_lb":e,"exit_lb":x} for e in es for x in xs]
    return [{"rsi_n":n,"entry":e,"exit":p["exit"]} for n in sorted(set([max(2,p["rsi_n"]-3),p["rsi_n"],p["rsi_n"]+3])) for e in sorted(set([max(10,p["entry"]-5),p["entry"],min(45,p["entry"]+5)]))]

def ui_params(s,key):
    if s=="SMA Crossover":
        a,b=st.columns(2);return {"fast":a.number_input("Fast SMA",2,200,20,key="f"+key),"slow":b.number_input("Slow SMA",3,500,100,key="s"+key)}
    if s=="Donchian Breakout":
        a,b=st.columns(2);return {"entry_lb":a.number_input("Entry lookback",2,250,55,key="e"+key),"exit_lb":b.number_input("Exit lookback",2,250,20,key="x"+key)}
    a,b,c=st.columns(3);return {"rsi_n":a.number_input("RSI period",2,100,14,key="r"+key),"entry":b.number_input("Entry RSI",1,49,30,key="i"+key),"exit":c.number_input("Exit RSI",40,99,50,key="o"+key)}

def save_exp(t,s,p,start,end,cost,m,status):
    raw=f"{t}{s}{p}{start}{end}{datetime.utcnow().isoformat()}";eid=hashlib.sha1(raw.encode()).hexdigest()[:10]
    c=con();c.execute("insert into experiments values(?,?,?,?,?,?,?,?,?,?,?,?)",(eid,datetime.utcnow().isoformat(),t,s,json.dumps(p),str(start),str(end),cost,m["cagr"],m["dd"],None if pd.isna(m["sharpe"]) else float(m["sharpe"]),status));c.commit();c.close()

tabs=st.tabs(["Validation Lab","Strategy Library","Universe Lab","Forward Paper","Database","Method"])

with tabs[0]:
    name=st.selectbox("Instrument",list(PRE));ticker=PRE[name];a,b=st.columns(2);start=a.date_input("Start",date(2012,1,1));end=b.date_input("End",date.today())
    s=st.selectbox("Strategy",["SMA Crossover","Donchian Breakout","RSI Mean Reversion"]);p=ui_params(s,"v")
    a,b,c=st.columns(3);br=a.number_input("Brokerage bps",0.,100.,3.,.5);fees=b.number_input("Taxes/fees bps",0.,100.,5.,.5);sl=c.number_input("Slippage bps",0.,100.,5.,.5);cost=br+fees+sl
    hold=st.slider("Final untouched holdout %",20,40,30,5);trainy=st.slider("WF training years",2,8,4);testm=st.slider("WF test months",3,12,6,3)
    if st.button("Validate & Save"):
        d=fetch(ticker,start,end)
        if len(d)<700:st.error("Need more history.")
        else:
            cut=int(len(d)*(1-hold/100));dev=d.iloc[:cut];ho=d.iloc[cut:];hm=run(ho,s,p,100000,cost)
            wf=[];cur=dev.index.min()+pd.DateOffset(years=trainy)
            while cur<dev.index.max():
                tr=dev[(dev.index>=cur-pd.DateOffset(years=trainy))&(dev.index<cur)];te1=min(cur+pd.DateOffset(months=testm),dev.index.max());te=dev[(dev.index>=cur)&(dev.index<=te1)]
                if len(tr)>250 and len(te)>30:
                    bestp=p;best=-999
                    for cp in candidates(s,p):
                        m=run(tr,s,cp,100000,cost);sc=(-999 if pd.isna(m["sharpe"]) else m["sharpe"]-.5*abs(m["dd"]))
                        if sc>best:best=sc;bestp=cp
                    tm=run(te,s,bestp,100000,cost);wf.append([cur.date(),te1.date(),str(bestp),tm["cagr"],tm["dd"],tm["sharpe"]])
                cur=te1
            w=pd.DataFrame(wf,columns=["Start","End","Params","CAGR","Max DD","Sharpe"])
            pos=(w.CAGR>0).mean() if len(w) else 0
            status="PAPER CANDIDATE" if hm["cagr"]>0 and pd.notna(hm["sharpe"]) and hm["sharpe"]>=.5 and hm["dd"]>-.35 and pos>=.6 else "REJECT"
            a,b,c=st.columns(3);a.metric("Holdout CAGR",f"{hm['cagr']:.2%}");b.metric("Holdout DD",f"{hm['dd']:.2%}");c.metric("WF positive windows",f"{pos:.1%}")
            st.dataframe(w,use_container_width=True);st.success(status) if status=="PAPER CANDIDATE" else st.error(status)
            save_exp(ticker,s,p,start,end,cost,hm,status);st.info("Experiment saved to ALPHA database.")

with tabs[1]:
    st.subheader("Walk-forward strategy library")
    st.write("The library is the history of validated experiments—not a leaderboard of highest CAGR.")
    c=con();df=pd.read_sql_query("select * from experiments order by created desc",c);c.close()
    if len(df):
        st.dataframe(df[["created","ticker","strategy","params","holdout_cagr","holdout_dd","holdout_sharpe","status"]],use_container_width=True)
        good=df[df.status=="PAPER CANDIDATE"];st.metric("Paper candidates",len(good))
    else:st.info("No saved experiments yet.")

with tabs[2]:
    st.subheader("Universe Lab")
    st.warning("This built-in list is a CURRENT-stock sample, not survivorship-bias-free historical NIFTY membership. Do not use it to claim historical stock-selection alpha.")
    s=st.selectbox("Universe strategy",["SMA Crossover","Donchian Breakout","RSI Mean Reversion"],key="us");p=ui_params(s,"u")
    if st.button("Scan research universe"):
        rows=[]
        for t in NIFTY50_SAMPLE:
            try:d=fetch(t,date(2018,1,1),date.today())
            except:continue
            if len(d)>500:
                m=run(d,s,p,100000,13);rows.append([t,m["cagr"],m["dd"],m["sharpe"],int(m["signal"].iloc[-1])])
        st.dataframe(pd.DataFrame(rows,columns=["Ticker","CAGR","Max DD","Sharpe","Current state"]),use_container_width=True)

with tabs[3]:
    st.subheader("Forward paper-trading journal")
    s=st.selectbox("Paper strategy",["SMA Crossover","Donchian Breakout","RSI Mean Reversion"],key="ps");p=ui_params(s,"p")
    names=st.multiselect("Paper universe",list(PRE),default=["NIFTY 50","NIFTY BANK","RELIANCE","HDFC BANK","INFOSYS"])
    if st.button("Record today's paper states"):
        c=con();n=0
        for name in names:
            t=PRE[name]
            try:d=fetch(t,date(2022,1,1),date.today())
            except:continue
            if len(d)>100:
                sg=sig(d,s,p);now=int(sg.iloc[-1]);prev=int(sg.iloc[-2]);state="NEW ENTRY" if now and not prev else ("IN" if now else ("NEW EXIT" if prev else "OUT"))
                c.execute("insert into paper_signals(run_time,data_date,ticker,strategy,params,state,close) values(?,?,?,?,?,?,?)",(datetime.utcnow().isoformat(),str(d.index[-1].date()),t,s,json.dumps(p),state,float(d.Close.iloc[-1])));n+=1
        c.commit();c.close();st.success(f"Recorded {n} paper observations.")
    c=con();pj=pd.read_sql_query("select * from paper_signals order by id desc limit 500",c);c.close()
    if len(pj):st.dataframe(pj,use_container_width=True)

with tabs[4]:
    st.subheader("Database backup")
    st.write("Streamlit Community Cloud local disk is not durable infrastructure. Download backups regularly; production should move this database to managed Postgres.")
    if DB.exists():st.download_button("Download alpha.db backup",DB.read_bytes(),file_name="alpha.db",mime="application/octet-stream")
    if st.button("Clear cache (not database)"):st.cache_data.clear();st.success("Market-data cache cleared.")

with tabs[5]:
    st.markdown("""**What v0.6 changes:** research results and forward paper observations are persisted in SQLite; walk-forward results feed a strategy library; a current-stock universe lab is separated from historical claims; and the app explicitly labels survivorship bias.

**Data-source reality:** Yahoo Finance remains a convenience source. For serious deployment, replace it with a licensed/official market-data source.  
**Database reality:** SQLite on free cloud hosting can disappear on restart/redeploy. Download backups; production needs managed Postgres.  
**Execution:** still deliberately disabled. Forward paper evidence should accumulate before broker integration.""")
