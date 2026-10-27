"""
Security master: one permanent company ID per stock, plus the Massive ticker valid on each date.

Why (tested Oct 2026):
- GOOG and GOOGL are one company: LSEG TR.OrganizationID is the same (5030853586).
- LSEG uses today's RIC / option root for all history (META.OQ, METAR...^F22 work for 2022,
  when the ticker was FB). Massive uses the ticker valid at the time (FB before 2022-06-09).
- Massive ticker history comes from /vX/reference/tickers/{CUSIP}/events. Looked up by CUSIP,
  not ticker, because delisted tickers get reused by other companies.
- Delisted companies have no Massive events, so they fall back to the ticker in the RIC.
"""
import os
import time
from pathlib import Path

import lseg.data as ld
import pandas as pd
import requests
from dotenv import load_dotenv

from src.data.security_master import massive_ticker_from_ric, lseg_option_root

load_dotenv()

RAW = Path("data/raw/universe")
MASSIVE = "https://api.massive.com"


def _retry(fn, retries=3, wait=5):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} after {type(e).__name__}, waiting {wait * 2 ** attempt}s")
            time.sleep(wait * 2 ** attempt)


def fetch_company_ids(rics, batch=100):
    """[ING-16] LSEG company ID (TR.OrganizationID), name, ticker and CUSIP for each RIC"""
    frames = []
    for i in range(0, len(rics), batch):
        chunk = list(rics[i:i + batch])
        df = _retry(lambda: ld.get_data(chunk, ["TR.OrganizationID", "TR.CommonName", "TR.TickerSymbol", "TR.CUSIP"]))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.columns = ["ric", "company_id", "name", "ticker", "cusip"]
    out = out.drop_duplicates("ric")
    for c in ["company_id", "name", "ticker", "cusip"]:
        out[c] = out[c].astype("string").replace("", pd.NA)
    # RICs LSEG can't resolve keep a unique placeholder ID so they're never merged with anything
    missing = out["company_id"].isna()
    out.loc[missing, "company_id"] = "RIC:" + out.loc[missing, "ric"]
    return out.reset_index(drop=True)


def fetch_ticker_history(cusip):
    """[ING-17] Massive ticker history for one CUSIP: list of (ticker, valid_from), oldest first; [] if none"""
    if cusip is None or pd.isna(cusip) or not str(cusip).strip():
        return []
    headers = {"Authorization": f"Bearer {os.getenv('POLYGON_API_KEY')}"}
    r = _retry(lambda: requests.get(f"{MASSIVE}/vX/reference/tickers/{cusip}/events", headers=headers, timeout=60))
    if r.status_code == 404:
        return []
    r.raise_for_status()
    events = (r.json().get("results") or {}).get("events", [])
    hist = [(e["ticker_change"]["ticker"], pd.Timestamp(e["date"]))
            for e in events if e.get("type") == "ticker_change" and e.get("ticker_change", {}).get("ticker")]
    return sorted(hist, key=lambda x: x[1])


def build_security_master(membership):
    """
    [ING-18] security master + Massive ticker history for every RIC ever in the membership table

    membership: DataFrame with columns date, ric (from ING-12 build_membership)
    saves data/raw/universe/security_master.parquet and massive_tickers.parquet, returns both
    massive_tickers.source: "events" (Massive history), "ric_fallback" (ticker from the RIC),
    "events_overridden" (Massive history disagreed with the current ticker, fallback used)
    """
    spans = membership.groupby("ric")["date"].agg(first_member="min", last_member="max").reset_index()
    ids = fetch_company_ids(spans["ric"].tolist())
    master = spans.merge(ids, on="ric", how="left")
    master["delisted"] = master["ric"].str.contains("^", regex=False)
    master["lseg_option_root"] = master["ric"].map(lseg_option_root)

    rows = []
    for i, m in enumerate(master.itertuples(index=False)):
        fallback = massive_ticker_from_ric(m.ric)
        hist = fetch_ticker_history(m.cusip)
        source = "events" if hist else "ric_fallback"
        # a live stock's latest Massive ticker must be its current ticker, otherwise don't trust the history
        if hist and not m.delisted and hist[-1][0] != fallback:
            hist, source = [], "events_overridden"
        if not hist:
            hist = [(fallback, pd.NaT)]
        rows += [(m.ric, t, start, source) for t, start in hist]
        if (i + 1) % 100 == 0:
            print(f"  ticker history {i + 1}/{len(master)}")
    tickers = pd.DataFrame(rows, columns=["ric", "massive_ticker", "valid_from", "source"])

    RAW.mkdir(parents=True, exist_ok=True)
    master.to_parquet(RAW / "security_master.parquet", index=False)
    tickers.to_parquet(RAW / "massive_tickers.parquet", index=False)
    return master, tickers
