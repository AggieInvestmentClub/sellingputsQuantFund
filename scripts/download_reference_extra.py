"""
Reference data still missing before the backtest (LSEG, ~15 requests):
  1. GICS sector / industry group / industry for every RIC in the security master
  2. Daily prices for SPY's 11 sector ETFs (sector-relative shocks)
  3. Daily VIX
  4. How far back LSEG news headlines go (one stock, a few dates)

Needs: LSEG Workspace open, data/raw/universe/security_master.parquet
Run from the repo root:  py -3.12 -u scripts/download_reference_extra.py
Writes data/raw/reference/*.parquet. Safe to re-run: finished files are skipped.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import lseg.data as ld
import pandas as pd

from src.ingest.lseg_client import lseg_session
from src.ingest.lseg_prices import fetch_daily_history

START = "2014-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")
OUT = Path("data/raw/reference")
OUT.mkdir(parents=True, exist_ok=True)
SECTOR_ETFS = {"XLB": "Materials", "XLC": "Communication Services", "XLE": "Energy", "XLF": "Financials",
               "XLI": "Industrials", "XLK": "Information Technology", "XLP": "Consumer Staples",
               "XLRE": "Real Estate", "XLU": "Utilities", "XLV": "Health Care", "XLY": "Consumer Discretionary"}

master = pd.read_parquet("data/raw/universe/security_master.parquet")

with lseg_session():
    # 1) GICS classification. Only RICs not yet in gics.parquet are requested, 20 per request;
    #    a request LSEG rejects (one bad RIC fails the whole batch) is retried one RIC at a time
    path = OUT / "gics.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["ric", "sector", "industry_group", "industry"])
    todo = [r for r in master["ric"] if r not in set(have["ric"])]
    fields = ["TR.GICSSector", "TR.GICSIndustryGroup", "TR.GICSIndustry"]
    frames, bad = [have], []
    for i in range(0, len(todo), 20):
        chunk = todo[i:i + 20]
        try:
            frames.append(ld.get_data(chunk, fields).set_axis(have.columns, axis=1))
        except Exception:
            for r in chunk:
                try:
                    frames.append(ld.get_data(r, fields).set_axis(have.columns, axis=1))
                except Exception as e:
                    bad.append(r)
                    frames.append(pd.DataFrame({"ric": [r]}))
    g = pd.concat(frames, ignore_index=True).drop_duplicates("ric", keep="last")
    for c in ["sector", "industry_group", "industry"]:
        g[c] = g[c].astype("string").replace("", pd.NA)
    g.to_parquet(path, index=False)
    print(f"GICS: {g['sector'].notna().sum()} of {len(master)} RICs have a sector "
          f"({len(todo)} requested this run, LSEG rejected: {bad})")
    print(g["sector"].value_counts().to_string())
    live = g[g["sector"].isna() & ~g["ric"].str.contains("^", regex=False)]
    print("live RICs with no sector:", live["ric"].tolist())

    # 2) sector ETFs: plain ticker, then the NYSE Arca RIC
    for name in SECTOR_ETFS:
        path = OUT / f"{name}.parquet"
        if path.exists():
            continue
        for ric in (name, f"{name}.P"):
            try:
                px = fetch_daily_history(ric, START, END)
            except Exception as e:
                print(f"  {ric}: {type(e).__name__}")
                continue
            if len(px):
                px.to_parquet(path)
                print(f"{name}: {ric} {len(px)} rows {px.index.min().date()}..{px.index.max().date()}")
                break
        else:
            print(f"{name}: NOT FOUND")

    # 3) VIX: index RICs often need explicit fields, so try three ways
    path = OUT / "VIX.parquet"
    if not path.exists():
        attempts = [
            ("get_history .VIX TRDPRC_1", lambda: ld.get_history(".VIX", fields=["TRDPRC_1", "HIGH_1", "LOW_1", "OPEN_PRC"],
                                                              interval="daily", start=START, end=END)),
            ("get_history .VIX default", lambda: ld.get_history(".VIX", interval="daily", start=START, end=END)),
            ("get_data .VIX TR.PriceClose", lambda: ld.get_data(".VIX", ["TR.PriceClose.date", "TR.PriceClose"],
                                                                {"SDate": START, "EDate": END, "Frq": "D"})),
        ]
        for label, fn in attempts:
            try:
                df = fn()
            except Exception as e:
                print(f"  VIX {label}: {type(e).__name__}: {str(e)[:120]}")
                continue
            if df is None or df.empty:
                print(f"  VIX {label}: empty")
                continue
            if "Date" in df.columns:          # get_data shape: Instrument, Date, Price Close
                df = df.dropna().set_index(pd.to_datetime(df["Date"]).dt.normalize())[["Price Close"]]
                df.columns = ["TRDPRC_1"]
            else:
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df.index.name = "date"
            df.columns.name = None
            df.astype(float).to_parquet(path)
            print(f"VIX: {label} -> {len(df)} rows {df.index.min().date()}..{df.index.max().date()}")
            break
        else:
            print("VIX: NOT FOUND (all three ways failed)")

    # 4) news depth: count Reuters/all headlines for AAPL on a few dates
    print("\nnews depth (AAPL.OQ, headlines in one day):")
    for d in ["2024-04-15"]:
        s, e = f"{d}T00:00:00", f"{d}T23:59:59"
        try:
            h = ld.news.get_headlines("R:AAPL.OQ AND Language:LEN", count=100, start=s, end=e)
            n = 0 if h is None else len(h)
            src = h["sourceCode"].value_counts().to_dict() if n and "sourceCode" in h else {}
            print(f"  {d}: {n} headlines, Reuters {src.get('NS:RTRS', 0)}")
        except Exception as ex:
            print(f"  {d}: FAILED {type(ex).__name__}: {str(ex)[:100]}")

print(f"\nsaved to {OUT}/")
