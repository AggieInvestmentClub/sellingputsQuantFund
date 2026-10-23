import time
from pathlib import Path
import pandas as pd
import lseg.data as ld

RAW = Path("data/raw/news")


def _get_page(query, page, start_iso, end_iso, retries=4):
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


def save_news(ric: str, df: pd.DataFrame) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{ric.replace('.', '_')}.parquet"
    df.to_parquet(path, index=False)
    return path