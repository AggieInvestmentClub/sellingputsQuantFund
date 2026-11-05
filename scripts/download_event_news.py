"""
Headlines (and a few Reuters story texts) around every shock, as input for the LLM filter (spec 2.3 / 2.4).

Window per event: previous trading day 20:00 UTC (US close) .. day +1 21:00 UTC.
The LLM decision at the shock-day close must only use headlines up to the shock day 21:00 UTC;
day +1 is kept for the entry-day re-check.

Needs LSEG Workspace open, data/results/event_study/events.csv.
Run from the repo root:  py -3.12 -u scripts/download_event_news.py [--since 2024-04-01] [--stories 2]
  --since    first shock date to fetch (default: start of the options data, 2024-04-01)
  --stories  Reuters story texts per event (0 = headlines only); each text is one extra request
Writes data/raw/news/events/<event>.parquet (headlines), data/raw/news/stories/<event>.parquet
(story texts), and data/raw/news/event_news_summary.csv. Resumable: finished events are skipped.
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import lseg.data as ld
import pandas as pd

from src.ingest.lseg_client import lseg_session
from src.ingest.lseg_news import fetch_headlines

ap = argparse.ArgumentParser()
ap.add_argument("--since", default="2024-04-01")
ap.add_argument("--stories", type=int, default=2)
ap.add_argument("--budget", type=int, default=6000, help="stop after this many LSEG requests")
args = ap.parse_args()

HEAD = Path("data/raw/news/events")
STORY = Path("data/raw/news/stories")
for p in (HEAD, STORY):
    p.mkdir(parents=True, exist_ok=True)

events = pd.read_csv("data/results/event_study/events.csv", parse_dates=["date"])
events = events[events["date"] >= pd.Timestamp(args.since)].sort_values("date").reset_index(drop=True)
bday = pd.offsets.BDay


def safe(event_id):
    return event_id.replace(".", "_").replace("^", "~").replace("|", "__")


def story_text(story_id):
    try:
        raw = ld.news.get_story(story_id, format=ld.news.Format.TEXT)
    except (AttributeError, TypeError):
        raw = ld.news.get_story(story_id)
    text = re.sub(r"<[^>]+>", " ", str(raw or ""))
    return re.sub(r"\s+", " ", text).strip()[:8000]


requests_used, t0, rows = 0, time.time(), []
with lseg_session():
    for i, ev in events.iterrows():
        event_id = f"{ev['ric']}|{ev['date']:%Y-%m-%d}"
        h_path, s_path = HEAD / f"{safe(event_id)}.parquet", STORY / f"{safe(event_id)}.parquet"
        if h_path.exists() and (args.stories == 0 or s_path.exists()):
            continue
        if requests_used >= args.budget:
            print(f"STOPPED: request budget {args.budget} used; re-run tomorrow to continue")
            break
        start = (ev["date"] - bday(1)).strftime("%Y-%m-%dT20:00:00")
        end = (ev["date"] + bday(1)).strftime("%Y-%m-%dT21:00:00")
        try:
            h = fetch_headlines(ev["ric"], start, end, window_days=7)
            requests_used += 1
        except Exception as e:
            print(f"  {event_id}: headlines failed ({type(e).__name__}: {str(e)[:80]})")
            requests_used += 1
            continue
        if h.empty:
            h = pd.DataFrame(columns=["event_time", "headline", "storyId", "sourceCode", "ric"])
        h = h.assign(event_id=event_id,
                     before_decision=pd.to_datetime(h["event_time"], utc=True)
                     <= pd.Timestamp(ev["date"].strftime("%Y-%m-%dT21:00:00"), tz="UTC"))
        h.to_parquet(h_path, index=False)

        reuters = h[(h["sourceCode"] == "NS:RTRS") & h["before_decision"]]
        reuters = reuters[~reuters["storyId"].astype(str).str.contains(r":\d+$") |
                          reuters["storyId"].astype(str).str.endswith(":1")]       # first version only
        texts = []
        for sid, head in reuters[["storyId", "headline"]].head(args.stories).itertuples(index=False):
            try:
                texts.append({"event_id": event_id, "storyId": sid, "headline": head, "text": story_text(sid)})
            except Exception as e:
                texts.append({"event_id": event_id, "storyId": sid, "headline": head, "text": f"FAILED {type(e).__name__}"})
            requests_used += 1
        if args.stories:
            pd.DataFrame(texts, columns=["event_id", "storyId", "headline", "text"]).to_parquet(s_path, index=False)
        rows.append({"event_id": event_id, "headlines": len(h), "before_decision": int(h["before_decision"].sum()),
                     "reuters_before_decision": len(reuters), "stories": len(texts)})
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(events)} events | {requests_used} requests | {el / 60:.0f} min")

# summary over everything on disk (this run and earlier ones)
summ = []
for ev in events.itertuples():
    event_id = f"{ev.ric}|{ev.date:%Y-%m-%d}"
    p = HEAD / f"{safe(event_id)}.parquet"
    if not p.exists():
        summ.append({"event_id": event_id, "fetched": False})
        continue
    h = pd.read_parquet(p)
    summ.append({"event_id": event_id, "fetched": True, "headlines": len(h),
                 "before_decision": int(h["before_decision"].sum()),
                 "reuters_before_decision": int(((h["sourceCode"] == "NS:RTRS") & h["before_decision"]).sum())})
s = pd.DataFrame(summ)
s.to_csv("data/raw/news/event_news_summary.csv", index=False)
f = s[s["fetched"]]
print(f"\nevents since {args.since}: {len(s)} | fetched: {len(f)} | requests this run: {requests_used}")
if len(f):
    print(f"  with any headline before the decision:  {(f['before_decision'] > 0).mean():.0%}")
    print(f"  with a Reuters headline before it:      {(f['reuters_before_decision'] > 0).mean():.0%}")
    print(f"  median headlines before the decision:   {f['before_decision'].median():.0f}")
print("saved data/raw/news/events/, data/raw/news/stories/, data/raw/news/event_news_summary.csv")
