"""
Download Reuters headlines around every shock in the event study (ING-23), for the LLM filter and
the news / no-news test. Needs LSEG Workspace open and data/results/event_study/events.csv.

Run from the repo root:
    python -u scripts/download_event_reuters.py --start 2024-04-08 --end 2026-09-15
    python -u scripts/download_event_reuters.py --limit 20          (smoke test)
Writes data/raw/news/reuters_events/<RIC>_<YYYYMMDD>.parquet (one per event, empty if no Reuters story)
and data/raw/news/reuters_events/manifest.csv. Safe to re-run: finished events are skipped.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import lseg.data as ld
import pandas as pd

from src.data.news import before_decision, first_versions
from src.ingest.lseg_client import lseg_session
from src.ingest.lseg_news import fetch_event_headlines

OUT = Path("data/raw/news/reuters_events")
parser = argparse.ArgumentParser()
parser.add_argument("--start", default="2024-04-08")
parser.add_argument("--end", default="2026-09-15")
parser.add_argument("--budget", type=int, default=4000, help="max LSEG news requests this run")
parser.add_argument("--limit", type=int, default=None, help="only the first N events (smoke test)")
args = parser.parse_args()


class Budget(BaseException):
    # BaseException so the retry loop inside ING-04 can't swallow it
    pass


calls = {"n": 0}
_get = ld.news.get_headlines


def counted(*a, **k):
    if calls["n"] >= args.budget:
        raise Budget()
    calls["n"] += 1
    return _get(*a, **k)


ld.news.get_headlines = counted


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} [{calls['n']:5d} req] {msg}", flush=True)


def path_for(ric, date):
    return OUT / f"{ric.replace('.', '_').replace('^', '~')}_{date:%Y%m%d}.parquet"


events = pd.read_csv("data/results/event_study/events.csv", parse_dates=["date"])
events = events[(events["date"] >= args.start) & (events["date"] <= args.end)].sort_values("date")
if args.limit:
    events = events.head(args.limit)
OUT.mkdir(parents=True, exist_ok=True)

failed = []
with lseg_session():
    try:
        log(f"start: {len(events)} events {args.start}..{args.end}, budget {args.budget}")
        for i, e in enumerate(events.itertuples(), 1):
            p = path_for(e.ric, e.date)
            if p.exists():
                continue
            try:
                fetch_event_headlines(e.ric, e.date).to_parquet(p, index=False)
            except Budget:
                raise
            except Exception as ex:
                failed.append((e.ric, e.date.date(), f"{type(ex).__name__}: {str(ex)[:80]}"))
            if i % 100 == 0:
                log(f"events {i}/{len(events)}")
        log("all events done")
    except Budget:
        log(f"STOPPED: budget of {args.budget} requests reached. Re-run to continue.")

rows = []
for e in events.itertuples():
    p = path_for(e.ric, e.date)
    if not p.exists():
        continue
    n = pd.read_parquet(p)
    stories = first_versions(n)
    rows.append({"ric": e.ric, "date": e.date.date(), "headlines": len(n), "stories": len(stories),
                 "stories_before_close": len(before_decision(stories, e.date))})
man = pd.DataFrame(rows)
man.to_csv(OUT / "manifest.csv", index=False)
if failed:
    pd.DataFrame(failed, columns=["ric", "date", "error"]).to_csv(OUT / "failures.csv", index=False)
if len(man):
    log(f"{len(man)} events saved | with >=1 Reuters story: {(man.stories > 0).mean():.0%} | "
        f"with >=1 before the close on the shock day: {(man.stories_before_close > 0).mean():.0%} | "
        f"failed: {len(failed)}")
