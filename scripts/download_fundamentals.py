"""
Download I/B/E/S consensus history (AG-30) for every stock that had a shock, plus their GICS industry peers
(needed for the P/E-vs-industry signal). Needs LSEG Workspace open.

    py -3.12 -u scripts/download_fundamentals.py --limit 3        # smoke test: prints which fields came back
    py -3.12 -u scripts/download_fundamentals.py                  # all (skips RICs already downloaded)
    py -3.12 -u scripts/download_fundamentals.py --refresh        # re-download everything (e.g. weekly, for live use)

Writes data/raw/fundamentals/<RIC>.parquet (field, date, value) and data/raw/fundamentals/failures.csv.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.insert(0, ".")
import pandas as pd

from src.agent.evidence import fname
from src.ingest.lseg_client import lseg_session
from src.ingest.lseg_fundamentals import FIELDS, fetch_fundamentals

OUT = Path("data/raw/fundamentals")


def universe():
    ev = pd.read_csv("data/results/event_study/events.csv")
    rics = set(ev.loc[ev["date"] >= "2024-01-01", "ric"])
    g = pd.read_parquet("data/raw/reference/gics.parquet")
    inds = set(g.loc[g["ric"].isin(rics), "industry"].dropna())
    peers = set(g.loc[g["industry"].isin(inds), "ric"])
    cur = set(pd.read_csv("data/raw/universe/sp500_current.csv")["Instrument"].dropna())
    return sorted(rics | (peers & (cur | rics)) | cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--frq", default="W", help="W (weekly, default) or D")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rics = universe()
    todo = [r for r in rics if a.refresh or not (OUT / f"{fname(r)}.parquet").exists()]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(rics)} RICs, {len(todo)} to download, fields: {[f[0] for f in FIELDS]}", flush=True)
    fails, t0 = [], time.time()
    with lseg_session():
        for i, r in enumerate(todo, 1):
            try:
                df = fetch_fundamentals(r, a.start, frq=a.frq)
            except Exception as e:
                fails.append({"ric": r, "error": f"{type(e).__name__}: {str(e)[:150]}"})
                continue
            tmp = OUT / f"{fname(r)}.parquet.tmp"
            df.to_parquet(tmp, index=False)
            tmp.replace(OUT / f"{fname(r)}.parquet")
            if a.limit:
                got = df.groupby("field").size().to_dict()
                print(f"  {r}: rows per field {got}; empty: {[f[0] for f in FIELDS if f[0] not in got]}")
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(fails, columns=["ric", "error"]).to_csv(OUT / "failures.csv", index=False)
    print(f"done: {len(todo) - len(fails)} ok, {len(fails)} failed (see {OUT / 'failures.csv'})")


if __name__ == "__main__":
    main()
