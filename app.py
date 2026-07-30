import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
import requests
import sqlite3
from kiteconnect import KiteConnect
from pathlib import Path


def _safe_dataframe_view(df, requested_cols=None, sort_col=None, ascending=False):
    """Return a display-safe DataFrame even when schema/API columns are missing."""
    import pandas as pd
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return pd.DataFrame()
    out = df.copy()
    if requested_cols:
        existing = [c for c in requested_cols if c in out.columns]
        out = out.loc[:, existing] if existing else out
    if sort_col and sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=ascending, na_position="last")
    return out


st.set_page_config(page_title="ALPHA Live v1.8.1", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")
st.title("🎯 ALPHA Live v1.8.1")
st.caption("Live Zerodha decision-support • technicals + 15m confirmation + NSE corporate events • manual execution")

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

def _historical_data_chunked(token, from_dt, to_dt, interval, max_days=1900):
    """
    Fetch Zerodha historical candles in safe chunks and merge them.

    Kite rejects overly large date ranges for many intervals.  Using 1900-day
    chunks keeps daily research below the observed 2000-day ceiling while
    preserving the full requested research window.
    """
    if from_dt >= to_dt:
        return []

    rows = []
    cursor = from_dt
    while cursor < to_dt:
        chunk_to = min(cursor + timedelta(days=max_days), to_dt)
        part = kite.historical_data(token, cursor, chunk_to, interval)
        if part:
            rows.extend(part)

        # Advance past the previous boundary. Daily candles are deduplicated
        # below, so a small overlap/boundary difference cannot duplicate bars.
        cursor = chunk_to + timedelta(seconds=1)

    return rows


def _history_frame(raw):
    """Normalize and deduplicate Kite historical-data output."""
    d = pd.DataFrame(raw)
    if d.empty:
        return d
    d = d.rename(columns=str.title)
    if "Date" not in d.columns:
        return pd.DataFrame()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).drop_duplicates(subset=["Date"], keep="last")
    return d.set_index("Date").sort_index()


def hist(sym, interval="day", days=550):
    tok = instrument_map().get(sym)
    if not tok:
        return pd.DataFrame()

    end = datetime.now()
    start = end - timedelta(days=int(days))

    # Daily research/backtests can request >2000 calendar days. Fetch those
    # windows in chunks rather than silently shortening the test.
    if interval == "day" and int(days) > 1900:
        raw = _historical_data_chunked(tok, start, end, interval, max_days=1900)
    else:
        raw = kite.historical_data(tok, start, end, interval)

    return _history_frame(raw)

def rsi(c, n=14):
    x=c.diff()
    g=x.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l=(-x.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100-(100/(1+g/l.replace(0,np.nan)))

def atr(d,n=14):
    pc=d.Close.shift(1)
    tr=pd.concat([(d.High-d.Low).abs(),(d.High-pc).abs(),(d.Low-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()


IST = ZoneInfo("Asia/Kolkata")
NIFTY50_TOKEN = 256265

def nifty_hist(days=365):
    """NIFTY 50 history, with chunked fetching for long evidence windows."""
    try:
        end = datetime.now(IST)
        start = end - timedelta(days=int(days))
        if int(days) > 1900:
            raw = _historical_data_chunked(
                NIFTY50_TOKEN, start, end, "day", max_days=1900
            )
        else:
            raw = kite.historical_data(NIFTY50_TOKEN, start, end, "day")
        return _history_frame(raw)
    except Exception:
        return pd.DataFrame()

# ---- NSE official corporate-information layer ----
# This is an event-risk layer, NOT a directional news-prediction engine.
NSE_HOME = "https://www.nseindia.com"
NSE_ANN = NSE_HOME + "/api/corporate-announcements?index=equities"
NSE_BOARD = NSE_HOME + "/api/corporate-board-meetings?index=equities"
NSE_ACTIONS = NSE_HOME + "/api/corporates-corporateActions?index=equities"

def _nse_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": NSE_HOME + "/companies-listing/corporate-filings-application"
    })
    try:
        s.get(NSE_HOME, timeout=8)
    except Exception:
        pass
    return s

def _json_rows(url):
    try:
        r = _nse_session().get(url, timeout=10)
        if r.status_code != 200:
            return []
        x = r.json()
        if isinstance(x, list):
            return x
        if isinstance(x, dict):
            for key in ("data", "records", "result"):
                if isinstance(x.get(key), list):
                    return x[key]
        return []
    except Exception:
        return []

def _first(d, *keys):
    for k in keys:
        v = d.get(k)
        if v not in (None, "", "-"):
            return str(v)
    return ""

def _event_class(text):
    t = (text or "").lower()
    high = (
        "financial result", "results", "merger", "acquisition", "order", "contract",
        "fund raising", "buyback", "bonus", "split", "default", "fraud",
        "resignation", "regulatory", "penalty", "insolvency", "dividend"
    )
    medium = (
        "board meeting", "investor", "analyst", "press release", "record date",
        "shareholder", "allotment", "appointment"
    )
    if any(x in t for x in high): return "HIGH"
    if any(x in t for x in medium): return "MEDIUM"
    return "LOW"

@st.cache_data(ttl=300, show_spinner=False)
def nse_event_book():
    """Build a symbol -> recent/upcoming official NSE event map."""
    book = {}
    sources = [
        ("ANNOUNCEMENT", NSE_ANN),
        ("BOARD MEETING", NSE_BOARD),
        ("CORPORATE ACTION", NSE_ACTIONS),
    ]
    for kind, url in sources:
        for x in _json_rows(url):
            sym = _first(x, "symbol", "sm_symbol", "symbolName", "SYMBOL").upper().strip()
            if not sym:
                continue
            subject = _first(x, "subject", "desc", "purpose", "bm_purpose", "attchmntText", "details")
            when = _first(
                x, "an_dt", "broadcastDate", "broadcast_date", "bm_date",
                "meetingDate", "exDate", "ex_date", "recordDate"
            )
            text = " ".join([kind, subject, when]).strip()
            item = {
                "type": kind,
                "headline": subject or kind.title(),
                "when": when,
                "impact": _event_class(text),
                "source": "NSE"
            }
            book.setdefault(sym, []).append(item)
    return book

def stock_event_summary(sym, book):
    items = book.get(sym, [])
    if not items:
        return "None detected from current NSE event feed", "NONE", []
    rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    items = sorted(items, key=lambda x: rank.get(x["impact"], 0), reverse=True)[:3]
    top = items[0]
    label = f'{top["type"]}: {top["headline"]}'
    if top["when"]:
        label += f' • {top["when"]}'
    return label, top["impact"], items

def event_risk_adjustment(direction, score, impact):
    """
    Corporate events are treated as uncertainty/risk, not guessed as bullish/bearish.
    HIGH-impact event => require a stronger technical score.
    """
    required = score
    if impact == "HIGH":
        required = max(required, 80)
    elif impact == "MEDIUM":
        required = max(required, 75)
    return required

def nifty_regime():
    try:
        d = nifty_hist(365)
        if len(d) < 200:
            return "UNKNOWN", None
        c = d.Close.astype(float)
        s20 = c.rolling(20).mean()
        s50 = c.rolling(50).mean()
        s200 = c.rolling(200).mean()
        px = float(c.iloc[-1])
        ret5 = float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) >= 6 else 0
        if px > s20.iloc[-1] > s50.iloc[-1] > s200.iloc[-1] and ret5 > 0:
            return "LONG BIAS", 1
        if px < s20.iloc[-1] < s50.iloc[-1] < s200.iloc[-1] and ret5 < 0:
            return "SHORT BIAS", -1
        return "SELECTIVE", 0
    except Exception:
        return "UNKNOWN", None

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
    """Four-factor 15m confirmation: EMA20, session VWAP, candle direction, relative volume."""
    try:
        d = hist(sym, "15minute", 10)
        if len(d) < 30:
            return False, "Insufficient 15m data", 0, ["Insufficient 15m history"]

        c = d.Close.astype(float)
        h = d.High.astype(float)
        lo = d.Low.astype(float)
        v = d.Volume.astype(float)
        ema20 = c.ewm(span=20, adjust=False).mean()
        px = float(c.iloc[-1])

        idx = pd.to_datetime(d.index)
        latest_day = idx[-1].date()
        mask = pd.Series([x.date() == latest_day for x in idx], index=d.index)
        sd = d.loc[mask.values].copy()
        typical = (sd.High.astype(float) + sd.Low.astype(float) + sd.Close.astype(float)) / 3
        cumv = sd.Volume.astype(float).cumsum()
        vwap = float((typical * sd.Volume.astype(float)).cumsum().iloc[-1] / max(cumv.iloc[-1], 1))

        volavg = float(v.tail(20).mean())
        checks = {}
        if direction == "LONG":
            checks["Above 15m EMA20"] = px > float(ema20.iloc[-1])
            checks["Above session VWAP"] = px > vwap
            checks["Latest candle bullish"] = float(c.iloc[-1]) >= float(d.Open.astype(float).iloc[-1])
        else:
            checks["Below 15m EMA20"] = px < float(ema20.iloc[-1])
            checks["Below session VWAP"] = px < vwap
            checks["Latest candle bearish"] = float(c.iloc[-1]) <= float(d.Open.astype(float).iloc[-1])
        checks["Volume >= 0.8x 20-bar avg"] = float(v.iloc[-1]) >= volavg * .8

        passed = sum(bool(x) for x in checks.values())
        failed = [name for name, ok in checks.items() if not ok]
        ok = passed >= 3
        msg = f"{passed}/4 checks passed"
        return ok, msg, passed, failed
    except Exception as e:
        return False, "15m confirmation unavailable", 0, [str(e)[:100]]


# ==================== v1.5 Strategy + Calendar + Tracked Trade Layer ====================

TRADE_STORE = "alpha_trades.json"

def now_ist():
    return datetime.now(IST)

def market_clock():
    n = now_ist()
    wd = n.strftime("%A")
    mins = n.hour * 60 + n.minute
    is_weekday = n.weekday() < 5
    if not is_weekday:
        session = "CLOSED / WEEKEND"
    elif mins < 9*60+15:
        session = "PRE-MARKET / BEFORE CASH SESSION"
    elif mins <= 15*60+30:
        session = "MARKET OPEN"
    else:
        session = "MARKET CLOSED"
    return n, wd, session

@st.cache_data(ttl=86400, show_spinner=False)
def nfo_instruments():
    try:
        return pd.DataFrame(kite.instruments("NFO"))
    except Exception:
        return pd.DataFrame()

def expiry_context(underlying=None):
    """
    Uses live Zerodha NFO instrument master rather than hard-coding weekday expiry rules.
    This automatically adapts when exchange expiry conventions change.
    """
    df = nfo_instruments()
    if df.empty or "expiry" not in df.columns:
        return {"label":"UNAVAILABLE","dte":None,"expiry":None,"note":"NFO instrument master unavailable"}
    x = df.copy()
    x["expiry"] = pd.to_datetime(x["expiry"], errors="coerce").dt.date
    x = x[x["expiry"].notna()]
    today = now_ist().date()
    x = x[x["expiry"] >= today]
    if underlying and "name" in x.columns:
        ux = x[x["name"].astype(str).str.upper() == underlying.upper()]
        if len(ux): x = ux
    if not len(x):
        return {"label":"NO CONTRACT","dte":None,"expiry":None,"note":"No future expiry found"}
    exp = min(x["expiry"])
    dte = (exp - today).days
    if dte == 0: label = "EXPIRY DAY"
    elif dte == 1: label = "1 DAY TO EXPIRY"
    elif dte <= 3: label = "NEAR EXPIRY"
    else: label = f"{dte} DAYS TO EXPIRY"
    return {"label":label,"dte":dte,"expiry":exp,"note":f"Nearest NFO expiry: {exp}"}

def confidence_label(score):
    if score >= 85: return "VERY HIGH"
    if score >= 75: return "HIGH"
    if score >= 65: return "MEDIUM"
    return "LOW"

def strategy_scores(row, intraday_passed, event_impact, regime, expiry):
    """
    These are SETUP QUALITY scores, not win probabilities.
    They are intentionally displayed as confidence until validated outcome history exists.
    """
    base = int(row["Score"])
    direction = row["Direction"]
    regime_bonus = 0
    if regime == "LONG BIAS" and direction == "LONG": regime_bonus = 7
    elif regime == "SHORT BIAS" and direction == "SHORT": regime_bonus = 7
    elif regime == "SELECTIVE": regime_bonus = -3
    elif regime == "UNKNOWN": regime_bonus = -8

    event_penalty = {"HIGH":10,"MEDIUM":5,"LOW":2,"NONE":0}.get(event_impact,0)
    intra = max(0,min(100, base + (12 if intraday_passed else -12) + regime_bonus - event_penalty))
    swing = max(0,min(100, base + regime_bonus - round(event_penalty*.7)))

    # Options buying requires a stronger underlying setup and intraday timing.
    option = base + (15 if intraday_passed else -18) + regime_bonus - event_penalty
    if expiry.get("dte") is not None:
        if expiry["dte"] == 0: option -= 12
        elif expiry["dte"] <= 2: option -= 7
        elif expiry["dte"] >= 4: option += 3
    option = max(0,min(100,option))

    # Long-term is deliberately NOT called a true investment score because fundamentals are absent.
    longterm_technical = max(0,min(100, base + (8 if direction=="LONG" else -20) - event_penalty))
    return {"INTRADAY":intra,"SWING":swing,"OPTIONS":option,"LONGTERM_TECH":longterm_technical}

def choose_strategy(scores, direction, expiry):
    # Do not recommend options merely because leverage is available.
    eligible = {
        "INTRADAY STOCK": scores["INTRADAY"] if scores["INTRADAY"] >= 78 else -1,
        "SWING STOCK": scores["SWING"] if scores["SWING"] >= 72 and direction=="LONG" else -1,
        "OPTION BUY": scores["OPTIONS"] if scores["OPTIONS"] >= 88 else -1,
    }
    best = max(eligible, key=eligible.get)
    if eligible[best] < 0:
        return "NO TRADE"
    return best

def option_contract_hint(sym, direction, ltp, expiry):
    """
    Selects a liquid-ish near-ATM contract from Zerodha's live NFO instrument master.
    This is a contract candidate, not an options valuation model.
    """
    df = nfo_instruments()
    if df.empty or expiry.get("expiry") is None:
        return None
    x = df.copy()
    x["expiry"] = pd.to_datetime(x["expiry"], errors="coerce").dt.date
    if "name" not in x.columns or "instrument_type" not in x.columns:
        return None
    opt_type = "CE" if direction=="LONG" else "PE"
    x = x[(x["name"].astype(str).str.upper()==sym.upper()) &
          (x["expiry"]==expiry["expiry"]) &
          (x["instrument_type"].astype(str).str.upper()==opt_type)]
    if not len(x): return None
    x["strike_dist"] = (pd.to_numeric(x["strike"], errors="coerce") - float(ltp)).abs()
    x = x.sort_values("strike_dist")
    r = x.iloc[0]
    return {"tradingsymbol":r.get("tradingsymbol"),"strike":float(r.get("strike",0)),
            "type":opt_type,"expiry":str(r.get("expiry")),"lot_size":int(r.get("lot_size",1))}

def _db():
    # Local SQLite persistence. Survives normal reruns/restarts on a stable host,
    # but Streamlit Community Cloud can replace the filesystem on redeploy.
    path = Path("/tmp/alpha_trades.db")
    con = sqlite3.connect(path, check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS trades(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, direction TEXT, strategy TEXT, entry REAL, qty INTEGER,
        stop REAL, t1 REAL, t2 REAL, entered_at TEXT, status TEXT,
        original_score INTEGER, expiry TEXT, event_at_entry TEXT,
        option_symbol TEXT, option_entry REAL, option_qty INTEGER,
        exit_price REAL, exit_time TEXT, exit_reason TEXT
    )""")
    con.commit()
    return con

def _load_trades():
    con=_db()
    df=pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC",con)
    con.close()
    return df.to_dict("records") if len(df) else []

def _save_trade(t):
    con=_db()
    cols=["symbol","direction","strategy","entry","qty","stop","t1","t2","entered_at","status",
          "original_score","expiry","event_at_entry","option_symbol","option_entry","option_qty"]
    vals=[t.get(c) for c in cols]
    con.execute(f"INSERT INTO trades ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})",vals)
    con.commit();con.close()

def _close_trade(trade_id, exit_price, reason):
    con=_db()
    con.execute("UPDATE trades SET status='CLOSED',exit_price=?,exit_time=?,exit_reason=? WHERE id=?",
                (float(exit_price),now_ist().isoformat(),reason,int(trade_id)))
    con.commit();con.close()

def option_live_snapshot(option_symbol):
    if not option_symbol:return None
    try:
        q=kite.quote([f"NFO:{option_symbol}"])[f"NFO:{option_symbol}"]
        depth=q.get("depth",{})
        buy=depth.get("buy",[]) or []; sell=depth.get("sell",[]) or []
        bid=float(buy[0]["price"]) if buy else np.nan
        ask=float(sell[0]["price"]) if sell else np.nan
        ltp=float(q.get("last_price",0))
        oi=float(q.get("oi",0) or 0)
        vol=float(q.get("volume",0) or 0)
        spread=(ask-bid) if np.isfinite(bid) and np.isfinite(ask) else np.nan
        spread_pct=(spread/ltp*100) if ltp>0 and np.isfinite(spread) else np.nan
        return {"ltp":ltp,"bid":bid,"ask":ask,"spread":spread,"spread_pct":spread_pct,"oi":oi,"volume":vol}
    except Exception:
        return None

def option_management(t, underlying_ltp):
    snap=option_live_snapshot(t.get("option_symbol"))
    base_action,r=trade_monitor_decision_underlying(t,underlying_ltp)
    if not snap:return base_action,r,None
    prem_entry=float(t.get("option_entry") or 0)
    prem=snap["ltp"]
    prem_ret=(prem/prem_entry-1) if prem_entry>0 else 0
    exp=datetime.fromisoformat(t["expiry"]).date() if t.get("expiry") else None
    dte=(exp-now_ist().date()).days if exp else None

    # Premium-aware risk management. This does not fabricate Greeks/IV.
    if prem_entry>0 and prem <= prem_entry*.70:
        action="EXIT — OPTION PREMIUM RISK LIMIT"
    elif prem_entry>0 and prem >= prem_entry*1.50:
        action="PARTIAL EXIT / TRAIL OPTION PROFIT"
    elif dte is not None and dte<=0 and now_ist().time() >= time(14,45):
        action="EXIT REVIEW — EXPIRY / THETA RISK"
    elif snap.get("spread_pct") is not None and np.isfinite(snap["spread_pct"]) and snap["spread_pct"]>2:
        action="TIGHTEN RISK — OPTION SPREAD WIDE"
    else:
        action=base_action
    return action,r,{"snapshot":snap,"premium_return":prem_ret,"dte":dte}

def trade_monitor_decision_underlying(t, ltp):
    direction=t["direction"]; entry=float(t["entry"]); stop=float(t["stop"])
    t1=float(t["t1"]); t2=float(t["t2"]); strategy=t["strategy"]
    risk=max(abs(entry-stop),0.01)
    if direction=="LONG":
        r=(ltp-entry)/risk
        if ltp <= stop:return "EXIT — STOP/INVALIDATION",r
        if ltp >= t2:return "PROTECT PROFIT / EXIT REMAINDER",r
        if ltp >= t1:return "PARTIAL EXIT / TRAIL STOP",r
    else:
        r=(entry-ltp)/risk
        if ltp >= stop:return "EXIT — STOP/INVALIDATION",r
        if ltp <= t2:return "PROTECT PROFIT / EXIT REMAINDER",r
        if ltp <= t1:return "PARTIAL EXIT / TRAIL STOP",r
    if strategy=="INTRADAY STOCK":
        ok,msg,passed,failed=intraday_confirm(t["symbol"],direction)
        if not ok and r<0:return "EXIT / REDUCE REVIEW — INTRADAY SETUP WEAK",r
        if passed<3:return "TIGHTEN RISK / WATCH",r
    elif strategy=="SWING STOCK":
        z=score_stock(t["symbol"],ltp,100000,1)
        if z and (z["Direction"]!=direction or z["Score"]<60):
            return "EXIT / REDUCE REVIEW — SWING THESIS WEAK",r
    if r>=1:return "HOLD / PROTECT PROFIT",r
    return "HOLD / MANAGE",r

def trade_monitor_decision(t, ltp):
    if t["strategy"]=="OPTION BUY":
        return option_management(t,ltp)
    action,r=trade_monitor_decision_underlying(t,ltp)
    return action,r,None

def get_underlying_ltp(sym):
    try:
        return float(kite.ltp([f"NSE:{sym}"])[f"NSE:{sym}"]["last_price"])
    except Exception:
        return None



# ==================== v1.7 Historical Validation Layer ====================

def _score_at_bar(d, i):
    """Rebuild the daily setup using only information available at bar i (no look-ahead)."""
    if i < 200 or i >= len(d):
        return None
    x=d.iloc[:i+1].copy()
    c=x.Close.astype(float); h=x.High.astype(float); lo=x.Low.astype(float); v=x.Volume.astype(float)
    s20=c.rolling(20).mean(); s50=c.rolling(50).mean(); s200=c.rolling(200).mean()
    rv=float(rsi(c).iloc[-1]); av=float(atr(x).iloc[-1])
    if not np.isfinite(rv) or not np.isfinite(av) or av <= 0: return None
    px=float(c.iloc[-1])
    hi=float(h.shift(1).rolling(20).max().iloc[-1]); low20=float(lo.shift(1).rolling(20).min().iloc[-1])
    volavg=float(v.rolling(20).mean().iloc[-1]); vr=float(v.iloc[-1]/volavg) if volavg>0 else 0
    mom=float(px/c.iloc[-21]-1)
    long=0; short=0
    if px>s200.iloc[-1]: long+=20
    if px<s200.iloc[-1]: short+=20
    if s20.iloc[-1]>s50.iloc[-1]>s200.iloc[-1]: long+=20
    if s20.iloc[-1]<s50.iloc[-1]<s200.iloc[-1]: short+=20
    if px>hi: long+=25
    elif px>=hi*.985: long+=12
    if px<low20: short+=25
    elif px<=low20*1.015: short+=12
    if 50<=rv<=68: long+=15
    if 32<=rv<=50: short+=15
    if rv>75: long-=10
    if rv<25: short-=10
    if vr>=1.5:
        if mom>=0: long+=15
        else: short+=15
    elif vr>=1.1:
        if mom>=0: long+=8
        else: short+=8
    if mom>.05: long+=10
    elif mom>0: long+=5
    if mom<-.05: short+=10
    elif mom<0: short+=5
    long=max(0,min(100,long)); short=max(0,min(100,short))
    direction='LONG' if long>=short else 'SHORT'; score=max(long,short)
    if direction=='LONG':
        stop=max(px-1.5*av,float(s20.iloc[-1])); risk=max(px-stop,.01)
        t1=px+1.5*risk; t2=px+2.5*risk
    else:
        stop=min(px+1.5*av,float(s20.iloc[-1])); risk=max(stop-px,.01)
        t1=px-1.5*risk; t2=px-2.5*risk
    return {'direction':direction,'score':int(score),'entry':px,'stop':float(stop),'t1':float(t1),'t2':float(t2),'risk':risk}

def _walk_forward_trade(d, signal_i, sig, max_hold=20):
    """Enter next daily open. Conservative rule: if stop and target touch in same bar, stop wins."""
    entry_i=signal_i+1
    if entry_i>=len(d): return None
    entry=float(d.Open.iloc[entry_i]); risk=float(sig['risk'])
    if risk<=0: return None
    # Preserve signal-day risk distance but anchor levels to executable next-open entry.
    if sig['direction']=='LONG':
        stop=entry-risk; t1=entry+1.5*risk; t2=entry+2.5*risk
    else:
        stop=entry+risk; t1=entry-1.5*risk; t2=entry-2.5*risk
    end=min(len(d)-1,entry_i+max_hold-1); t1_hit=False
    for j in range(entry_i,end+1):
        hi=float(d.High.iloc[j]); lo=float(d.Low.iloc[j])
        if sig['direction']=='LONG':
            if lo<=stop: return j,-1.0,'SL'
            if hi>=t2: return j,2.5,'T2'
            if hi>=t1: t1_hit=True
        else:
            if hi>=stop: return j,-1.0,'SL'
            if lo<=t2: return j,2.5,'T2'
            if lo<=t1: t1_hit=True
    exit_px=float(d.Close.iloc[end])
    r=((exit_px-entry)/risk) if sig['direction']=='LONG' else ((entry-exit_px)/risk)
    # T1 is recorded as evidence, but unresolved trades exit at max-hold close to avoid invented fills.
    return end,float(r),'TIME'+(' / T1 TOUCHED' if t1_hit else '')

def backtest_symbol(sym, min_score=70, years=3, max_hold=20):
    days=max(550,int(years*365)+260)
    d=hist(sym,'day',days)
    if len(d)<230: return pd.DataFrame()
    d=d.copy().dropna(subset=['Open','High','Low','Close'])
    start=max(200,len(d)-int(years*252)-max_hold-5)
    rows=[]; next_free=start
    for i in range(start,len(d)-1):
        if i<next_free: continue
        sig=_score_at_bar(d,i)
        if not sig or sig['score']<min_score: continue
        out=_walk_forward_trade(d,i,sig,max_hold)
        if not out: continue
        exit_i,r,outcome=out
        rows.append({'Symbol':sym,'Signal Date':str(pd.to_datetime(d.index[i]).date()),
                     'Entry Date':str(pd.to_datetime(d.index[i+1]).date()),'Direction':sig['direction'],
                     'Score':sig['score'],'R':round(r,3),'Outcome':outcome,
                     'Exit Date':str(pd.to_datetime(d.index[exit_i]).date())})
        next_free=exit_i+1
    return pd.DataFrame(rows)

def backtest_metrics(bt):
    if bt is None or bt.empty: return None
    r=pd.to_numeric(bt['R'],errors='coerce').dropna()
    if not len(r): return None
    eq=r.cumsum(); peak=eq.cummax(); dd=eq-peak
    gross_win=float(r[r>0].sum()); gross_loss=abs(float(r[r<0].sum()))
    return {'Trades':len(r),'Win rate %':float((r>0).mean()*100),'Expectancy R':float(r.mean()),
            'Profit factor':(gross_win/gross_loss if gross_loss>0 else np.inf),
            'Net R':float(r.sum()),'Max drawdown R':float(dd.min())}



# ==================== v1.8 Evidence + Calibration Layer ====================

def classify_setup_at_bar(d, i, sig=None):
    """Classify the historical daily setup without using future bars."""
    if sig is None:
        sig = _score_at_bar(d, i)
    if not sig or i < 200:
        return "NO VALID SETUP"
    x=d.iloc[:i+1]
    c=x.Close.astype(float); h=x.High.astype(float); lo=x.Low.astype(float)
    s20=c.rolling(20).mean(); s50=c.rolling(50).mean(); s200=c.rolling(200).mean()
    px=float(c.iloc[-1])
    hi=float(h.shift(1).rolling(20).max().iloc[-1])
    low20=float(lo.shift(1).rolling(20).min().iloc[-1])
    direction=sig["direction"]
    if direction=="LONG":
        if px > hi: return "TREND BREAKOUT"
        if px > s200.iloc[-1] and s20.iloc[-1] > s50.iloc[-1] and abs(px/s20.iloc[-1]-1) <= .03:
            return "TREND PULLBACK"
        if px > s50.iloc[-1] > s200.iloc[-1]: return "MOMENTUM CONTINUATION"
    else:
        if px < low20: return "BREAKDOWN / SHORT"
        if px < s200.iloc[-1] and s20.iloc[-1] < s50.iloc[-1] and abs(px/s20.iloc[-1]-1) <= .03:
            return "BEAR TREND PULLBACK"
        if px < s50.iloc[-1] < s200.iloc[-1]: return "BEAR MOMENTUM"
    return "MIXED TECHNICAL SETUP"

def historical_regime_series(days=2200):
    """Point-in-time NIFTY daily regime map used by historical validation."""
    d=nifty_hist(days)
    if d.empty or len(d)<210: return {}
    c=d.Close.astype(float)
    s20=c.rolling(20).mean(); s50=c.rolling(50).mean(); s200=c.rolling(200).mean()
    out={}
    for i in range(200,len(d)):
        px=float(c.iloc[i]); ret5=float(c.iloc[i]/c.iloc[i-5]-1) if i>=5 else 0
        if px>s20.iloc[i]>s50.iloc[i]>s200.iloc[i] and ret5>0: rg="LONG BIAS"
        elif px<s20.iloc[i]<s50.iloc[i]<s200.iloc[i] and ret5<0: rg="SHORT BIAS"
        else: rg="SELECTIVE"
        out[pd.to_datetime(d.index[i]).date()]=rg
    return out

def backtest_symbol_v18(sym, min_score=55, years=5, max_hold=20, regime_map=None):
    """Research dataset: signal features + future R, using only point-in-time information."""
    days=max(800,int(years*365)+300)
    d=hist(sym,'day',days)
    if len(d)<230: return pd.DataFrame()
    d=d.copy().dropna(subset=['Open','High','Low','Close'])
    start=max(200,len(d)-int(years*252)-max_hold-5)
    rows=[]; next_free=start
    for i in range(start,len(d)-1):
        if i<next_free: continue
        sig=_score_at_bar(d,i)
        if not sig or sig["score"]<min_score: continue
        out=_walk_forward_trade(d,i,sig,max_hold)
        if not out: continue
        exit_i,r,outcome=out
        dt=pd.to_datetime(d.index[i]).date()
        setup=classify_setup_at_bar(d,i,sig)
        rg=(regime_map or {}).get(dt,"UNKNOWN")
        rows.append({
            "Symbol":sym,"Signal Date":str(dt),"Entry Date":str(pd.to_datetime(d.index[i+1]).date()),
            "Direction":sig["direction"],"Score":sig["score"],"Setup":setup,"Regime":rg,
            "R":round(float(r),3),"Outcome":outcome,
            "Exit Date":str(pd.to_datetime(d.index[exit_i]).date())
        })
        next_free=exit_i+1
    return pd.DataFrame(rows)

def evidence_metrics(bt):
    m=backtest_metrics(bt)
    if not m: return None
    # Practical robustness diagnostics, not claims of statistical certainty.
    g=bt.copy()
    g["Year"]=pd.to_datetime(g["Signal Date"],errors="coerce").dt.year
    yearly=[]
    for y,yy in g.groupby("Year"):
        mm=backtest_metrics(yy)
        if mm: yearly.append(mm["Expectancy R"])
    positive_year_share=(sum(x>0 for x in yearly)/len(yearly)) if yearly else 0
    return {**m,"Positive year share":positive_year_share,"Years":len(yearly)}

def _date_split(bt, train_frac=.70):
    if bt is None or bt.empty: return pd.DataFrame(),pd.DataFrame()
    x=bt.copy()
    x["_dt"]=pd.to_datetime(x["Signal Date"],errors="coerce")
    x=x.sort_values("_dt").dropna(subset=["_dt"])
    dates=sorted(x["_dt"].dt.date.unique())
    if len(dates)<2:return x.iloc[0:0],x
    cut=dates[max(1,min(len(dates)-1,int(len(dates)*train_frac)))-1]
    return x[x["_dt"].dt.date<=cut].drop(columns=["_dt"]), x[x["_dt"].dt.date>cut].drop(columns=["_dt"])

def comparable_evidence(research_bt, symbol, direction, setup, regime, score, min_n=20):
    """
    Prefer same symbol+direction+setup+regime. Relax progressively if sample is too small.
    Evidence is descriptive historical/OOS evidence, not a probability guarantee.
    """
    if research_bt is None or research_bt.empty:
        return None
    train,oos=_date_split(research_bt,.70)
    if oos.empty:return None

    score_lo=max(55,int(score)-10); score_hi=min(100,int(score)+10)
    filters=[
        ("EXACT", lambda x:(x.Symbol==symbol)&(x.Direction==direction)&(x.Setup==setup)&(x.Regime==regime)&x.Score.between(score_lo,score_hi)),
        ("SETUP+REGIME", lambda x:(x.Direction==direction)&(x.Setup==setup)&(x.Regime==regime)&x.Score.between(score_lo,score_hi)),
        ("SETUP", lambda x:(x.Direction==direction)&(x.Setup==setup)&x.Score.between(score_lo,score_hi)),
        ("DIRECTION+SCORE", lambda x:(x.Direction==direction)&x.Score.between(score_lo,score_hi)),
    ]
    chosen=None; level=None
    for name,fn in filters:
        z=oos[fn(oos)]
        if len(z)>=min_n:
            chosen=z;level=name;break
    if chosen is None:
        name,fn=filters[-1]; chosen=oos[fn(oos)]; level="LOW SAMPLE"
    m=evidence_metrics(chosen)
    if not m:return None
    # Hard evidence gates. Designed to reject weak/unstable edges rather than maximize trade count.
    sample_ok=m["Trades"]>=min_n
    exp_ok=m["Expectancy R"]>=0.15
    pf_ok=m["Profit factor"]>=1.30
    dd_ok=m["Max drawdown R"]>=-12
    stability_ok=(m["Years"]<2) or (m["Positive year share"]>=0.50)
    passed=sum([sample_ok,exp_ok,pf_ok,dd_ok,stability_ok])
    if passed==5: strength="STRONG"
    elif passed>=3 and sample_ok and m["Expectancy R"]>0 and m["Profit factor"]>1: strength="MIXED"
    else: strength="WEAK"
    return {
        "level":level,"strength":strength,"sample":m["Trades"],"win_rate":m["Win rate %"],
        "expectancy":m["Expectancy R"],"pf":m["Profit factor"],"max_dd":m["Max drawdown R"],
        "positive_year_share":m["Positive year share"],"gate_pass":strength=="STRONG"
    }

def live_setup_class(sym, direction):
    d=hist(sym,"day",550)
    if len(d)<210:return "NO VALID SETUP"
    sig=_score_at_bar(d,len(d)-1)
    return classify_setup_at_bar(d,len(d)-1,sig) if sig else "NO VALID SETUP"

def final_alpha_verdict(row, evidence, intraday_ok, event_impact, regime):
    """Evidence first. A high technical score cannot override failed historical evidence."""
    if evidence is None:
        return "RESEARCH / WAIT", "No OOS historical evidence loaded"
    if evidence["sample"] < 20:
        return "RESEARCH / WAIT", f"Only {evidence['sample']} comparable OOS trades"
    if evidence["strength"]=="WEAK":
        return "REJECT", f"Weak OOS edge: {evidence['expectancy']:+.2f}R expectancy, PF {evidence['pf']:.2f}"
    if event_impact=="HIGH":
        return "WAIT", "High-impact corporate event risk"
    if row["Recommended Strategy"]=="INTRADAY STOCK" and not intraday_ok:
        return "WAIT", "Daily setup exists but intraday entry is not confirmed"
    if regime=="LONG BIAS" and row["Direction"]=="SHORT":
        return "WAIT", "Short conflicts with broad-market regime"
    if regime=="SHORT BIAS" and row["Direction"]=="LONG":
        return "WAIT", "Long conflicts with broad-market regime"
    if evidence["strength"]=="STRONG":
        return "TRADE", "OOS evidence + current execution gates passed"
    return "WAIT", "Historical edge is positive but robustness is not strong enough"

def build_research_database(symbols, years=5, max_hold=20):
    rg=historical_regime_series(max(2200,years*400+300))
    all_bt=[]
    for sym in symbols:
        try:
            z=backtest_symbol_v18(sym,55,years,max_hold,rg)
            if len(z):all_bt.append(z)
        except Exception:
            pass
    return pd.concat(all_bt,ignore_index=True) if all_bt else pd.DataFrame()


# ==================== v1.5 UI ====================

tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs(
    ["Today's Playbook","Evidence Scanner","My Tracked Trades","Zerodha Positions","Evidence Lab","Backtest Lab","Rules"]
)

with tab1:
    n, weekday, session = market_clock()
    exp = expiry_context()
    st.subheader("ALPHA Today's Playbook")
    c1,c2,c3 = st.columns(3)
    c1.metric("Trading day", f"{weekday} • {n.strftime('%d %b %Y')}")
    c2.metric("Cash market", session)
    c3.metric("Expiry context", exp["label"])
    st.caption(exp["note"])
    st.info("ALPHA does not assume expiry day = option-buying day. Near expiry makes the options filter stricter.")

    if "last_v15_scan" in st.session_state:
        x=st.session_state.last_v15_scan
        df=x["df"]
        if len(df):
            best=df.sort_values("BestScore",ascending=False).iloc[0]
            st.markdown(f"### Best current opportunity: {best.Symbol} • {best['Recommended Strategy']} • {best.Direction}")
            st.write(f"**Setup confidence:** {confidence_label(best.BestScore)} ({int(best.BestScore)}/100)")
            st.write(f"Intraday {int(best.IntradayScore)}/100 • Swing {int(best.SwingScore)}/100 • Options {int(best.OptionsScore)}/100")
    else:
        st.caption("Run the scanner once to generate today's strategy playbook.")

with tab2:
    a,b=st.columns(2)
    capital=a.number_input("Trading capital ₹",10000,100000000,100000,10000,key="v15_cap")
    risk=b.number_input("Risk per trade %",.25,2.0,1.0,.25,key="v15_risk")
    minscore=st.slider("Minimum underlying setup score",55,90,70,5,key="v15_min")

    if st.button("Run ALPHA v1.8 Evidence Scanner",use_container_width=True):
        scan_time=now_ist()
        regime,bias=nifty_regime()
        event_book=nse_event_book()
        exp=expiry_context()
        q=kite.ltp([f"NSE:{s}" for s in SYMS])
        rows=[]
        bar=st.progress(0)
        for i,s in enumerate(SYMS):
            k=f"NSE:{s}"
            if k in q:
                z=score_stock(s,float(q[k]["last_price"]),capital,risk)
                if z:
                    ok,msg,passed,failed=intraday_confirm(s,z["Direction"])
                    ev,impact,items=stock_event_summary(s,event_book)
                    scores=strategy_scores(z,ok,impact,regime,exp)
                    strat=choose_strategy(scores,z["Direction"],exp)
                    bestscore=max(scores["INTRADAY"],scores["SWING"],scores["OPTIONS"])
                    z.update({
                        "15m Confirm":ok,"15m checks":f"{passed}/4",
                        "15m failed":" • ".join(failed) if failed else "None",
                        "News / Event":ev,"Event impact":impact,
                        "IntradayScore":scores["INTRADAY"],"SwingScore":scores["SWING"],
                        "OptionsScore":scores["OPTIONS"],"LongTermTechnical":scores["LONGTERM_TECH"],
                        "Recommended Strategy":strat,"BestScore":bestscore
                    })
                    setup=live_setup_class(s,z["Direction"])
                    evidence=comparable_evidence(st.session_state.get("alpha_research_bt",pd.DataFrame()),
                                                 s,z["Direction"],setup,regime,z["Score"])
                    verdict,verdict_reason=final_alpha_verdict(z,evidence,ok,impact,regime)
                    z.update({"Setup Type":setup,"Evidence":evidence,"ALPHA Verdict":verdict,
                              "Verdict Reason":verdict_reason})
                    rows.append(z)
            bar.progress((i+1)/len(SYMS))
        df=pd.DataFrame(rows)
        st.session_state.last_v15_scan={"df":df,"time":scan_time,"regime":regime,"expiry":exp}
        st.rerun()

    if "last_v15_scan" in st.session_state:
        x=st.session_state.last_v15_scan
        df=x["df"]; regime=x["regime"]; exp=x["expiry"]
        st.write(f"**Market regime:** {regime} • **Scan:** {x['time'].strftime('%d-%b-%Y %I:%M:%S %p IST')} • **Expiry:** {exp['label']}")
        actionable=df[(df.Score>=minscore)&(df["Recommended Strategy"]!="NO TRADE")].copy()
        if "ALPHA Verdict" in actionable.columns:
            order={"TRADE":0,"WAIT":1,"RESEARCH / WAIT":2,"REJECT":3}
            actionable["_vorder"]=actionable["ALPHA Verdict"].map(order).fillna(9)
            actionable=actionable.sort_values(["_vorder","BestScore"],ascending=[True,False]).head(5)
        else:
            actionable=actionable.sort_values("BestScore",ascending=False).head(5)

        if not len(actionable):
            st.warning("NO TRADE NOW — no strategy clears the current quality thresholds.")
        for idx,r in actionable.iterrows():
            st.markdown(f"### {r.Symbol} — {r['Recommended Strategy']} — {r.Direction}")
            verdict=r.get("ALPHA Verdict","RESEARCH / WAIT")
            if verdict=="TRADE": st.success(f"ALPHA VERDICT: TRADE — {r.get('Verdict Reason','')}")
            elif verdict=="REJECT": st.error(f"ALPHA VERDICT: REJECT — {r.get('Verdict Reason','')}")
            else: st.warning(f"ALPHA VERDICT: {verdict} — {r.get('Verdict Reason','')}")
            st.write(f"**Setup:** {r.get('Setup Type','Unknown')}")
            evd=r.get("Evidence")
            if isinstance(evd,dict):
                st.write(f"**OOS historical evidence:** {evd['strength']} • Comparable trades {evd['sample']} • "
                         f"Win {evd['win_rate']:.1f}% • Expectancy {evd['expectancy']:+.2f}R • "
                         f"PF {evd['pf']:.2f} • Max DD {evd['max_dd']:.1f}R • Match {evd['level']}")
            else:
                st.caption("No Evidence Lab database loaded yet; ALPHA will not call this evidence-validated.")
            st.write(
                f"**Setup confidence (not historical win probability):** "
                f"Intraday {confidence_label(r.IntradayScore)} {int(r.IntradayScore)}/100 • "
                f"Swing {confidence_label(r.SwingScore)} {int(r.SwingScore)}/100 • "
                f"Options {confidence_label(r.OptionsScore)} {int(r.OptionsScore)}/100"
            )
            c1,c2,c3=st.columns(3)
            c1.metric("Entry reference",f"₹{r.Entry:,.2f}")
            c2.metric("Stop",f"₹{r.Stop:,.2f}")
            c3.metric("Qty",int(r.Qty))
            st.write(f"**T1:** ₹{r['T1']:,.2f} • **T2:** ₹{r['T2']:,.2f} • **Planned SL risk:** ~₹{r['Planned loss ₹']:,.0f}")
            st.write(f"**News / Event:** {r['News / Event']} • Risk: {r['Event impact']}")
            st.write(f"**15m:** {r['15m checks']} • Failed: {r['15m failed']}")
            st.caption(f"Underlying evidence: {r.Why}")

            if r["Recommended Strategy"]=="OPTION BUY":
                oc=option_contract_hint(r.Symbol,r.Direction,r["Live LTP"],exp)
                if oc:
                    st.write(f"**Option candidate:** {oc['tradingsymbol']} • {oc['type']} • Strike {oc['strike']:.0f} • Expiry {oc['expiry']} • Lot {oc['lot_size']}")
                    st.warning("Contract selection is preliminary: v1.5 does not yet model IV/Greeks or option-premium stop/target, so do not treat this as a fully validated options entry.")
                else:
                    st.warning("Underlying/options setup scored well, but no tradable option contract passed the contract filters (expiry/strike/liquidity/quote checks). Do not force an options trade.")

            if r.get("ALPHA Verdict")!="TRADE":
                st.caption("Trade-entry journal is locked because the evidence/execution gates have not produced a TRADE verdict.")
                st.divider()
                continue
            with st.expander("I ENTERED THIS TRADE"):
                actual=st.number_input("Actual entry price",min_value=.01,value=float(r.Entry),key=f"entry_{r.Symbol}_{idx}")
                qty=st.number_input("Actual quantity",min_value=1,value=max(1,int(r.Qty)),key=f"qty_{r.Symbol}_{idx}")
                option_symbol=None; option_entry=None; option_qty=None
                if r["Recommended Strategy"]=="OPTION BUY":
                    oc2=option_contract_hint(r.Symbol,r.Direction,r["Live LTP"],exp)
                    default_opt=oc2["tradingsymbol"] if oc2 else ""
                    option_symbol=st.text_input("Actual option contract",value=default_opt,key=f"optsym_{r.Symbol}_{idx}")
                    option_entry=st.number_input("Actual option premium paid",min_value=0.01,value=1.0,key=f"optentry_{r.Symbol}_{idx}")
                    option_qty=st.number_input("Actual option quantity",min_value=1,value=(oc2["lot_size"] if oc2 else 1),key=f"optqty_{r.Symbol}_{idx}")
                if st.button("Save entered trade",key=f"save_{r.Symbol}_{idx}"):
                    _save_trade({
                        "symbol":r.Symbol,"direction":r.Direction,"strategy":r["Recommended Strategy"],
                        "entry":float(actual),"qty":int(qty),"stop":float(r.Stop),"t1":float(r["T1"]),"t2":float(r["T2"]),
                        "entered_at":now_ist().isoformat(),"status":"OPEN","original_score":int(r.Score),
                        "expiry":str(exp["expiry"]) if r["Recommended Strategy"]=="OPTION BUY" and exp.get("expiry") else None,
                        "event_at_entry":r["News / Event"],
                        "option_symbol":option_symbol,"option_entry":float(option_entry) if option_entry else None,
                        "option_qty":int(option_qty) if option_qty else None
                    })
                    st.success("Trade saved. Open 'My Tracked Trades' and refresh the app for live HOLD/EXIT guidance.")
            st.divider()

        with st.expander("All strategy scores"):
            cols=["Symbol","Direction","Setup Type","Score","IntradayScore","SwingScore","OptionsScore",
                  "Recommended Strategy","ALPHA Verdict","Verdict Reason","15m checks","News / Event","Event impact"]
            st.dataframe(_safe_dataframe_view(df, cols, "BestScore", ascending=False),use_container_width=True,hide_index=True)

        st.caption("LongTermTechnical is only a technical suitability indicator. ALPHA will not call a stock a long-term investment until a fundamentals/valuation data layer is added.")

with tab3:
    st.subheader("My Tracked Trades")
    st.caption("Refresh the page/app to pull fresh Zerodha prices and recalculate management guidance.")
    trades=_load_trades()
    open_count=0
    for t in trades:
        if t.get("status")!="OPEN": continue
        open_count+=1
        ltp=get_underlying_ltp(t["symbol"])
        st.markdown(f"### {t['symbol']} • {t['strategy']} • {t['direction']}")
        st.write(f"Entered: {t['entered_at']} • Entry ₹{t['entry']:,.2f} • Qty {t['qty']}")
        if ltp is None:
            st.error("Could not fetch current underlying LTP.")
            continue
        action,r_mult,optmeta=trade_monitor_decision(t,ltp)
        pnl=(ltp-t["entry"])*t["qty"] if t["direction"]=="LONG" else (t["entry"]-ltp)*t["qty"]
        c1,c2,c3=st.columns(3)
        c1.metric("Current underlying",f"₹{ltp:,.2f}")
        c2.metric("Approx underlying P&L",f"₹{pnl:,.0f}")
        c3.metric("R multiple",f"{r_mult:+.2f}R")
        st.subheader(action)
        st.write(f"Original SL ₹{t['stop']:,.2f} • T1 ₹{t['t1']:,.2f} • T2 ₹{t['t2']:,.2f}")
        if t["strategy"]=="OPTION BUY":
            if optmeta and optmeta.get("snapshot"):
                s=optmeta["snapshot"]
                st.write(f"**Option:** {t.get('option_symbol')} • Premium ₹{s['ltp']:,.2f} • Entry premium ₹{float(t.get('option_entry') or 0):,.2f}")
                st.write(f"**Option P&L %:** {optmeta['premium_return']*100:+.1f}% • OI {s['oi']:,.0f} • Volume {s['volume']:,.0f}")
                if np.isfinite(s.get("spread_pct",np.nan)):
                    st.write(f"**Bid/Ask:** ₹{s['bid']:,.2f} / ₹{s['ask']:,.2f} • Spread {s['spread_pct']:.2f}%")
                st.caption("Premium, liquidity and expiry are monitored live. IV/Greeks are not fabricated because Kite quote data does not directly provide them.")
            else:
                st.warning("Could not fetch the saved NFO option contract. Underlying-thesis management only.")
        reason=st.text_input("Exit note (optional)",key=f"reason_{t['id']}")
        if st.button("Mark trade CLOSED at current underlying price",key=f"close_{t['id']}"):
            _close_trade(t["id"],ltp,reason or action)
            st.rerun()
        st.divider()
    if open_count==0:
        st.info("No manually tracked open trades in this active Streamlit session.")

    closed=[t for t in trades if t.get("status")=="CLOSED"]
    if closed:
        with st.expander("Closed tracked trades"):
            st.dataframe(pd.DataFrame(closed),use_container_width=True,hide_index=True)

with tab4:
    st.subheader("Live Zerodha Positions")
    try:
        pos=kite.positions().get("net",[])
        live=[p for p in pos if p.get("quantity",0)!=0]
        if not live: st.info("No open net positions detected.")
        for p in live:
            st.write(f"**{p.get('tradingsymbol')}** • Qty {p.get('quantity')} • Avg ₹{float(p.get('average_price',0)):,.2f} • P&L ₹{float(p.get('pnl',0)):,.0f}")
    except Exception as e:
        st.error(f"Could not read Zerodha positions: {e}")

with tab5:
    st.subheader("Evidence Lab — build OOS research database")
    st.caption("Build this first. ALPHA uses the resulting historical research database as a gate in the live Evidence Scanner.")
    c1,c2=st.columns(2)
    ev_years=c1.selectbox("Research history",[3,4,5],index=2,key="ev_years")
    ev_hold=c2.selectbox("Research max holding days",[10,15,20,30],index=2,key="ev_hold")
    ev_syms=st.multiselect("Research universe",SYMS,default=SYMS,key="ev_syms")
    st.info("The database is split chronologically: earlier ~70% is treated as development history and later ~30% as out-of-sample evidence. Live recommendations query the OOS portion only.")
    if st.button("Build / refresh evidence database",use_container_width=True,key="build_evidence"):
        prog=st.progress(0)
        rg=historical_regime_series(max(2200,ev_years*400+300))
        all_bt=[]
        failures=[]
        completed=0
        for i,sym in enumerate(ev_syms):
            try:
                z=backtest_symbol_v18(sym,55,ev_years,ev_hold,rg)
                completed += 1
                if len(z):
                    all_bt.append(z)
                else:
                    st.caption(f"{sym}: history fetched, but no qualifying research trades were produced.")
            except Exception as e:
                failures.append((sym, str(e)))
                st.caption(f"{sym}: research unavailable — {str(e)[:160]}")
            prog.progress((i+1)/max(len(ev_syms),1))

        st.session_state.alpha_research_bt=pd.concat(all_bt,ignore_index=True) if all_bt else pd.DataFrame()

        if all_bt:
            st.success(
                f"Evidence database refreshed • {completed}/{len(ev_syms)} symbols processed • "
                f"{len(st.session_state.alpha_research_bt)} research trades created. "
                "Re-run the Evidence Scanner so live candidates can be gated by OOS evidence."
            )
        elif failures:
            st.error(
                "Evidence database was NOT built because historical research failed. "
                "Review the per-symbol errors above; ALPHA will keep live candidates at RESEARCH / WAIT."
            )
        else:
            st.warning(
                "Historical data was processed, but no qualifying research trades were produced "
                "for the selected universe/settings."
            )
    research=st.session_state.get("alpha_research_bt",pd.DataFrame())
    if len(research):
        train,oos=_date_split(research,.70)
        st.write(f"**Research signals:** {len(research)} • **Development portion:** {len(train)} • **OOS evidence portion:** {len(oos)}")
        m=evidence_metrics(oos)
        if m:
            a,b,c,d,e=st.columns(5)
            a.metric("OOS trades",m["Trades"]); b.metric("OOS expectancy",f"{m['Expectancy R']:+.2f}R")
            c.metric("OOS PF",f"{m['Profit factor']:.2f}"); d.metric("OOS max DD",f"{m['Max drawdown R']:.1f}R")
            e.metric("Positive-year share",f"{m['Positive year share']*100:.0f}%")
        st.markdown("#### OOS by setup")
        rows=[]
        for name,g in oos.groupby("Setup"):
            mm=evidence_metrics(g)
            if mm: rows.append({"Setup":name,**mm})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        st.markdown("#### OOS by regime + direction")
        rows=[]
        for (rg,dr),g in oos.groupby(["Regime","Direction"]):
            mm=evidence_metrics(g)
            if mm: rows.append({"Regime":rg,"Direction":dr,**mm})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    else:
        st.warning("No evidence database exists in this session yet. Until you build it, live candidates remain RESEARCH / WAIT.")

with tab6:
    st.subheader("Historical Backtest Lab")
    st.caption("Walk-forward daily validation using Zerodha historical OHLC. Signals use only data known on the signal date; execution is modeled at the next day open.")
    c1,c2,c3=st.columns(3)
    bt_years=c1.selectbox("History",[1,2,3,4,5],index=2,key="bt_years")
    bt_score=c2.slider("Minimum historical score",55,90,70,5,key="bt_score")
    bt_hold=c3.selectbox("Maximum holding days",[5,10,15,20,30],index=3,key="bt_hold")
    bt_syms=st.multiselect("Stocks to validate",SYMS,default=SYMS[:10],key="bt_syms")
    st.warning("This validates the DAILY technical engine, not the live 15-minute confirmation, NSE-event filter, or option-premium logic. Mixing those into this result without point-in-time historical data would create fake precision.")
    if st.button("Run historical validation",use_container_width=True,key="run_bt"):
        all_bt=[]; prog=st.progress(0)
        for i,sym in enumerate(bt_syms):
            try:
                z=backtest_symbol(sym,bt_score,bt_years,bt_hold)
                if len(z): all_bt.append(z)
            except Exception as e:
                st.caption(f"{sym}: backtest unavailable — {str(e)[:100]}")
            prog.progress((i+1)/max(len(bt_syms),1))
        st.session_state.alpha_bt=pd.concat(all_bt,ignore_index=True) if all_bt else pd.DataFrame()
    bt=st.session_state.get("alpha_bt",pd.DataFrame())
    if len(bt):
        m=backtest_metrics(bt)
        a,b,c,d,e,f=st.columns(6)
        a.metric("Trades",m['Trades']); b.metric("Win rate",f"{m['Win rate %']:.1f}%")
        c.metric("Expectancy",f"{m['Expectancy R']:+.2f}R"); d.metric("Profit factor",f"{m['Profit factor']:.2f}")
        e.metric("Net result",f"{m['Net R']:+.1f}R"); f.metric("Max drawdown",f"{m['Max drawdown R']:.1f}R")
        if m['Trades']<50: st.warning("Sample is small. Do not treat this as proof of an edge yet.")
        elif m['Expectancy R']<=0 or m['Profit factor']<=1: st.error("The tested daily engine has not demonstrated a positive historical edge under these assumptions.")
        else: st.success("The tested daily engine shows positive historical expectancy under these assumptions. It still needs out-of-sample/live validation before risking meaningful capital.")
        st.markdown("#### By direction")
        direction_rows=[]
        for name,g in bt.groupby('Direction'):
            mm=backtest_metrics(g); direction_rows.append({'Direction':name,**mm})
        st.dataframe(pd.DataFrame(direction_rows),use_container_width=True,hide_index=True)
        st.markdown("#### By score band")
        tmp=bt.copy(); tmp['Score band']=pd.cut(tmp.Score,[54,64,69,74,79,84,100],labels=['55-64','65-69','70-74','75-79','80-84','85+'])
        score_rows=[]
        for name,g in tmp.groupby('Score band',observed=True):
            mm=backtest_metrics(g); score_rows.append({'Score band':str(name),**mm})
        st.dataframe(pd.DataFrame(score_rows),use_container_width=True,hide_index=True)
        with st.expander("All historical simulated trades"):
            st.dataframe(bt.sort_values('Signal Date',ascending=False),use_container_width=True,hide_index=True)
    else:
        st.info("Choose stocks and run validation. No historical result is claimed until this test actually runs against your Zerodha data.")

with tab7:
    st.markdown("""
### ALPHA v1.8 rules

- **Evidence first:** a high technical score can no longer override weak out-of-sample historical evidence.
- Build the **Evidence Lab** database first; the live scanner then queries comparable OOS setups.
- Live candidates are classified as **TRADE / WAIT / RESEARCH-WAIT / REJECT**.
- Evidence matching prefers same stock + direction + setup + regime, then relaxes only when the OOS sample is too small.
- A STRONG evidence gate currently requires a meaningful sample plus positive expectancy, PF, drawdown and time-stability checks.
- Trade-entry journaling is locked unless the final evidence/execution verdict is **TRADE**.
- This remains decision-support, not a guarantee of profitability and not automatic order placement.

- The app knows the current India date, weekday and cash-market session.
- Expiry is derived from Zerodha's current NFO instrument master instead of hard-coding a weekday.
- Every stock gets separate **Intraday / Swing / Options** setup-quality scores.
- These numbers are **NOT win probabilities**. Real win probability will only be displayed after enough completed strategy-specific trades exist.
- Options require a substantially higher threshold than stock trades, and near-expiry conditions make the filter stricter.
- **I ENTERED THIS TRADE** records the actual entry and changes the workflow from scanning to position management.
- Tracked trades use strategy-specific HOLD / PROTECT PROFIT / PARTIAL EXIT / EXIT logic on refresh.
- Long-term investing is not recommended from technicals alone. Fundamentals and valuation are still missing.
- News/events remain an official-NSE event-risk layer; “None detected” is not proof that no relevant public news exists.
- No automatic order placement.
- The v1.8 Evidence Lab database is cached in Streamlit session memory in this build; rebuild it after a full app/session restart. This avoids silently treating stale research as current evidence.
- The Backtest Lab validates the daily technical engine with next-session execution and conservative same-bar stop/target handling; it does not pretend to validate historical 15m/event/options layers without point-in-time data.

### Storage and options notes
- v1.6 stores tracked trades in a local SQLite database rather than session state, so normal page refreshes and app reruns keep the trades.
- Streamlit Community Cloud can still replace its local filesystem during redeploy/rebuild. Truly durable cloud persistence requires an external database; the app cannot guarantee that with local SQLite alone.
- OPTION BUY trades now monitor the **actual saved NFO contract premium**, bid/ask spread, volume, OI, expiry context and the underlying thesis.
- ALPHA does **not fabricate IV or Greeks**. Kite's standard quote response does not directly supply a full Greeks/IV model; a proper option analytics model/data source is required before those are used for decisions.
""")
    if st.button("Logout Zerodha session"):
        st.session_state.pop("access_token",None);st.rerun()
