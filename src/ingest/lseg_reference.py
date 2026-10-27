"""
Reference data from LSEG: daily market cap, dividends, Treasury yields.
Run inside lseg_session() (ING-01).
"""
import lseg.data as ld
import pandas as pd


def fetch_market_cap(ric, start="2014-01-01", end=None):
    """[ING-20] daily company market cap for one RIC (TR.CompanyMarketCap), date index; one request covers 2015-2026"""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    df = ld.get_data(ric, ["TR.CompanyMarketCap", "TR.CompanyMarketCap.date"], {"SDate": start, "EDate": end, "Frq": "D"})
    if df is None or df.empty:
        return pd.Series(dtype=float, name="market_cap")
    df = df.dropna()
    s = pd.Series(df.iloc[:, 1].astype(float).values, index=pd.to_datetime(df.iloc[:, 2]).dt.normalize(), name="market_cap")
    s.index.name = "date"
    return s[~s.index.duplicated(keep="last")].sort_index()


def fetch_dividends(rics, start="2014-01-01", end=None, batch=50):
    """[ING-21] cash dividends (ex-date, gross amount per share, type) for many RICs, 50 per request"""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    frames = []
    for i in range(0, len(rics), batch):
        chunk = list(rics[i:i + batch])
        try:
            frames.append(ld.get_data(chunk, ["TR.DivExDate", "TR.DivUnadjustedGross", "TR.DivType"], {"SDate": start, "EDate": end}))
        except Exception as e:
            print(f"  dividends batch {i} failed: {type(e).__name__}: {e}")
    if not frames:
        return pd.DataFrame(columns=["ric", "ex_date", "amount", "div_type"])
    out = pd.concat(frames, ignore_index=True)
    out.columns = ["ric", "ex_date", "amount", "div_type"]
    out = out.dropna(subset=["ex_date", "amount"])
    out["ex_date"] = pd.to_datetime(out["ex_date"]).dt.normalize()
    return out.drop_duplicates().sort_values(["ric", "ex_date"]).reset_index(drop=True)


def fetch_treasury_yields(rics=("US1MT=RR", "US3MT=RR", "US6MT=RR"), start="2014-01-01", end=None):
    """[ING-22] daily Treasury bill yields in DECIMALS (B_YLD_1 / 100), one column per RIC, date index"""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cols = {}
    for r in rics:
        h = ld.get_history(r, interval="daily", start=start, end=end)
        s = h["B_YLD_1"].astype(float) / 100
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        cols[r] = s[~s.index.duplicated(keep="last")]
    out = pd.DataFrame(cols).sort_index()
    out.index.name = "date"
    return out
