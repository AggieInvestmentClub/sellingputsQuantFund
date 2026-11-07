"""
Reuters headlines and story texts around every shock, input for the LLM filter (spec 2.3 / 2.4)
and the news / no-news test. Needs LSEG Workspace open and data/results/event_study/events.csv.

Per event:
  1. headlines (ING-23): Reuters only (Source:RTRS in the query), shock day -3 .. +1. One request.
  2. story texts (ING-24): the first `--stories` Reuters stories that broke between the previous
     close and the shock-day close (DAT-08 first version, DAT-10 window); if none, the latest ones
     before the shock-day close (DAT-09). One request per story.
  Nothing published after the shock-day close (New York time) is fetched as a story text.

Run from the repo root:
    python -u scripts/download_event_news.py --start 2024-04-08 --end 2026-09-15
    python -u scripts/download_event_news.py --limit 20              (smoke test)
    python -u scripts/download_event_news.py --stories 0             (headlines only)
Writes, one file per event (<RIC>_<YYYYMMDD>.parquet):
    data/raw/news/reuters_events/   headlines (empty file if no Reuters story)
    data/raw/news/reuters_stories/  story texts: ric, shock_date, storyId, event_time, headline, text
    data/raw/news/reuters_events/manifest.csv (+ failures.csv)
Safe to re-run: finished files are skipped; stops cleanly at --budget requests.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import lseg.data as ld
import pandas as pd

from src.data.news import before_decision, first_versions, on_shock_day
from src.ingest.lseg_client import lseg_session
from src.ingest.lseg_news import fetch_event_headlines, fetch_story_text

HEAD = Path("data/raw/news/reuters_events")
STORY = Path("data/raw/news/reuters_stories")
STORY_COLS = ["ric", "shock_date", "storyId", "event_time", "headline", "text"]

parser = argparse.ArgumentParser()
parser.add_argument("--start", default="2024-04-08")
parser.add_argument("--end", default="2026-09-15")
parser.add_argument("--stories", type=int, default=2, help="story texts per event (0 = headlines only)")
parser.add_argument("--budget", type=int, default=6000, help="max LSEG news requests this run")
parser.add_argument("--limit", type=int, default=None, help="only the first N events (smoke test)")
args = parser.parse_args()


class Budget(BaseException):
    # BaseException so the retry loops in ING-04 / ING-24 can't swallow it
    pass


calls = {"n": 0}


def counted(fn):
    def wrapper(*a, **k):
        if calls["n"] >= args.budget:
            raise Budget()
        calls["n"] += 1
        return fn(*a, **k)
    return wrapper


ld.news.get_headlines = counted(ld.news.get_headlines)
ld.news.get_story = counted(ld.news.get_story)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} [{calls['n']:5d} req] {msg}", flush=True)


def name(ric, date):
    return f"{ric.replace('.', '_').replace('^', '~')}_{date:%Y%m%d}.parquet"


def pick_stories(headlines, date, n):
    """first versions; shock-day stories first, else the latest ones before the close"""
    stories = before_decision(first_versions(headlines), date)
    day = on_shock_day(stories, date)
    return day.head(n) if len(day) else stories.tail(n)


events = pd.read_csv("data/results/event_study/events.csv", parse_dates=["date"])
events = events[(events["date"] >= args.start) & (events["date"] <= args.end)].sort_values("date")
if args.limit:
    events = events.head(args.limit)
for p in (HEAD, STORY):
    p.mkdir(parents=True, exist_ok=True)

failed = []
with lseg_session():
    try:
        log(f"start: {len(events)} events {args.start}..{args.end}, {args.stories} stories each, "
            f"budget {args.budget}")
        for i, e in enumerate(events.itertuples(), 1):
            h_path, s_path = HEAD / name(e.ric, e.date), STORY / name(e.ric, e.date)
            try:
                if not h_path.exists():
                    fetch_event_headlines(e.ric, e.date).to_parquet(h_path, index=False)
                if args.stories and not s_path.exists():
                    pick = pick_stories(pd.read_parquet(h_path), e.date, args.stories)
                    rows = [{"ric": e.ric, "shock_date": e.date, "storyId": s.storyId,
                             "event_time": s.event_time, "headline": s.headline,
                             "text": fetch_story_text(s.storyId)} for s in pick.itertuples()]
                    pd.DataFrame(rows, columns=STORY_COLS).to_parquet(s_path, index=False)
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
    h_path, s_path = HEAD / name(e.ric, e.date), STORY / name(e.ric, e.date)
    if not h_path.exists():
        continue
    stories = before_decision(first_versions(pd.read_parquet(h_path)), e.date)
    texts = pd.read_parquet(s_path) if s_path.exists() else pd.DataFrame(columns=STORY_COLS)
    rows.append({"ric": e.ric, "date": e.date.date(), "stories_before_close": len(stories),
                 "stories_on_shock_day": len(on_shock_day(stories, e.date)),
                 "texts": len(texts), "texts_nonempty": int((texts["text"].str.len() > 0).sum())})
man = pd.DataFrame(rows)
man.to_csv(HEAD / "manifest.csv", index=False)
(HEAD / "failures.csv").unlink(missing_ok=True)        # only this run's failures
if failed:
    pd.DataFrame(failed, columns=["ric", "date", "error"]).to_csv(HEAD / "failures.csv", index=False)
if len(man):
    log(f"{len(man)} events | Reuters story before the close: {(man.stories_before_close > 0).mean():.0%} | "
        f"on the shock day: {(man.stories_on_shock_day > 0).mean():.0%} | "
        f"with story text: {(man.texts_nonempty > 0).mean():.0%} | failed: {len(failed)}")
