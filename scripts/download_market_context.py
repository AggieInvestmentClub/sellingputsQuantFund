"""
Market-wide context from LSEG for the index branch (AG-27). Run with LSEG Workspace open, ~2-5 minutes.

    py -3.12 -u scripts/download_market_context.py

Daily history 1990 -> today for each RIC below (whatever LSEG returns; failures are listed, not fatal), saved to
data/raw/market_context/<name>.parquet with all columns LSEG gives, plus a summary of what came back.
Used for: volatility regime (VIX level, term structure VIX9D/VIX/VIX3M/VIX6M), tail-risk pricing (SKEW, VVIX),
trend (SPX), credit stress (HY spread), rates (2y/10y), so the strategy can avoid entering into a crash.
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import pandas as pd

from src.ingest.lseg_client import lseg_session

OUT = Path("data/raw/market_context")
RICS = {                      # name: candidate RICs (first that works wins)
    "spx": [".SPX"], "spy": ["SPY"], "vix": [".VIX"], "vix9d": [".VIX9D"], "vix3m": [".VIX3M", ".VXV"],
    "vix6m": [".VIX6M", ".VXMT"], "vvix": [".VVIX"], "skew": [".SKEW"], "vxn": [".VXN"],
    "move": [".MOVE"], "us2y": ["US2YT=RR"], "us10y": ["US10YT=RR"],
    "hy_oas": ["BAMLH0A0HYM2", "aUSHYOAS"], "putcall": [".CPC", ".PCRATIO"],
}


def fetch(ld, ric, start, end):
    frames, s = [], pd.Timestamp(start)
    while s < pd.Timestamp(end):                       # chunks: get_history caps points per request
        e = min(s + pd.DateOffset(years=8), pd.Timestamp(end))
        df = ld.get_history(ric, interval="daily", start=s.strftime("%Y-%m-%d"), end=e.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            frames.append(df)
        s = e + pd.Timedelta(days=1)
    if not frames:
        return None
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df


def main():
    import lseg.data as ld
    OUT.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    rows = []
    with lseg_session():
        for name, cands in RICS.items():
            got = None
            for ric in cands:
                try:
                    df = fetch(ld, ric, "1990-01-01", end)
                except Exception as e:
                    rows.append({"name": name, "ric": ric, "rows": 0, "note": f"{type(e).__name__}: {str(e)[:80]}"})
                    continue
                if df is not None and len(df):
                    df.columns = [str(c) for c in df.columns]
                    df.to_parquet(OUT / f"{name}.parquet")
                    got = ric
                    rows.append({"name": name, "ric": ric, "rows": len(df), "first": df.index.min().date(),
                                 "last": df.index.max().date(), "columns": ",".join(df.columns[:6])})
                    break
                rows.append({"name": name, "ric": ric, "rows": 0, "note": "empty"})
                time.sleep(0.3)
            print(f"{name:8s} {'OK ' + got if got else 'missing'}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
