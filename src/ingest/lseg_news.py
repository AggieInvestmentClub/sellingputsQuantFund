from pathlib import Path
import pandas as pd
import lseg.data as ld

RAW = Path("data/raw/news")


def fetch_headlines(ric: str, start: str, end: str,
                    page: int = 100, lang: str = "LEN") -> pd.DataFrame:
    query = f"R:{ric} AND Language:{lang}"
    start_ts = pd.Timestamp(start, tz="UTC")
    cur_end = pd.Timestamp(end, tz="UTC")
    frames = []
    while True:
        df = ld.news.get_headlines(query, count=page,
                                   start=start_ts.isoformat(),
                                   end=cur_end.isoformat())
        if df is None or df.empty:
            break
        frames.append(df)
        oldest = pd.to_datetime(df.index.min(), utc=True)
        if len(df) < page or oldest <= start_ts:
            break
        cur_end = oldest - pd.Timedelta(seconds=1)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    out.index = pd.to_datetime(out.index, utc=True)
    out = out.reset_index().rename(columns={out.index.name or "index": "event_time"})
    out["data_available_time"] = out["event_time"]
    out["ric"] = ric
    out = out.drop_duplicates("storyId").sort_values("event_time")
    return out


def save_news(ric: str, df: pd.DataFrame) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{ric.replace('.', '_')}.parquet"
    df.to_parquet(path, index=False)
    return path