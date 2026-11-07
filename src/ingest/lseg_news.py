import re
import time
from pathlib import Path
import pandas as pd
import lseg.data as ld

RAW = Path("data/raw/news")


def _get_page(query, page, start_iso, end_iso, retries=4):
    """[ING-04] one page (up to 100) of headlines, retries with backoff on errors"""
    for attempt in range(retries):
        try:
            return ld.news.get_headlines(query, count=page,
                                         start=start_iso, end=end_iso)
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 5 * 2 ** attempt
            print(f"  retry {attempt + 1} after {type(e).__name__}, waiting {wait}s")
            time.sleep(wait)


def _fetch_window(query, start_ts, end_ts, page):
    """[ING-05] all headlines in one time window, paging backward 100 at a time"""
    frames = []
    cur_end = end_ts
    while True:
        df = _get_page(query, page, start_ts.isoformat(), cur_end.isoformat())
        if df is None or df.empty:
            break
        frames.append(df)
        oldest = pd.to_datetime(df.index.min(), utc=True)
        if len(df) < page or oldest <= start_ts:
            break
        cur_end = oldest - pd.Timedelta(seconds=1)
    return frames


def fetch_headlines(ric: str, start: str, end: str, page: int = 100,
                    lang: str = "LEN", window_days: int = 7) -> pd.DataFrame:
    """[ING-06] all English headlines for one RIC, fetched in 7-day windows, deduped by storyId"""
    query = f"R:{ric} AND Language:{lang}"
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    frames = []
    w_start = start_ts
    while w_start < end_ts:
        w_end = min(w_start + pd.Timedelta(days=window_days), end_ts)
        frames += _fetch_window(query, w_start, w_end, page)
        w_start = w_end

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    out.index = pd.to_datetime(out.index, utc=True)
    out = out.reset_index().rename(columns={out.index.name or "index": "event_time"})
    out["data_available_time"] = out["event_time"]
    out["ric"] = ric
    return out.drop_duplicates("storyId").sort_values("event_time")


def fetch_event_headlines(ric: str, date, days_before: int = 3, days_after: int = 1,
                          reuters_only: bool = True) -> pd.DataFrame:
    """
    [ING-23] headlines around one shock: from `days_before` calendar days before the shock day to
    the end of `days_after` days after it (UTC). Reuters only by default (Source:RTRS in the query,
    so LSEG filters it and one event is usually one request)

    ric: delisted RICs ("MRO.N^K24") are queried by their base RIC ("MRO.N")
    returns ING-06's columns plus `shock_date`; empty DataFrame (same columns) if nothing found
    """
    base = ric.split("^")[0]
    query = f"R:{base} AND Language:LEN" + (" AND Source:RTRS" if reuters_only else "")
    day = pd.Timestamp(date).normalize()
    start_ts = (day - pd.Timedelta(days=days_before)).tz_localize("UTC")
    end_ts = (day + pd.Timedelta(days=days_after + 1)).tz_localize("UTC") - pd.Timedelta(seconds=1)
    frames = _fetch_window(query, start_ts, end_ts, 100)
    cols = ["event_time", "headline", "storyId", "sourceCode", "data_available_time", "ric", "shock_date"]
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames)
    out.index = pd.to_datetime(out.index, utc=True)
    out = out.reset_index().rename(columns={out.index.name or "index": "event_time"})
    out["data_available_time"] = out["event_time"]
    out["ric"], out["shock_date"] = ric, day
    return out.drop_duplicates("storyId").sort_values("event_time")[cols].reset_index(drop=True)


def fetch_story_text(story_id: str, max_chars: int = 8000) -> str:
    """
    [ING-24] full text of one story (one LSEG request), HTML tags and extra spaces removed

    story_id: storyId from ING-06 / ING-23, e.g. "urn:newsml:reuters.com:20250514:nL1N3G4:0"
    returns plain text cut to max_chars ("" if LSEG returns nothing); retries like ING-04
    """
    for attempt in range(4):
        try:
            try:
                raw = ld.news.get_story(story_id, format=ld.news.Format.TEXT)
            except (AttributeError, TypeError):      # older lseg-data versions have no format option
                raw = ld.news.get_story(story_id)
            break
        except Exception as e:
            if attempt == 3:
                raise
            print(f"  retry {attempt + 1} after {type(e).__name__}, waiting {5 * 2 ** attempt}s")
            time.sleep(5 * 2 ** attempt)
    text = re.sub(r"<[^>]+>", " ", str(raw or ""))
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def save_news(ric: str, df: pd.DataFrame) -> Path:
    """[ING-07] save headlines to data/raw/news/<RIC>.parquet"""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{ric.replace('.', '_')}.parquet"
    df.to_parquet(path, index=False)
    return path