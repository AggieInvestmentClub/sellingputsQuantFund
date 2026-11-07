"""
Cleaning downloaded headlines (ING-06 / ING-23 output) before anything reads them.
Pure functions: no downloads.
"""
import pandas as pd


def first_versions(news):
    """
    [DAT-08] one row per Reuters story: LSEG lists each update of a story separately
    ("...:nL1N3G4:0", "...:nL1N3G4:1"); keep the earliest, so the time is when the news broke

    news: DataFrame with storyId and event_time. returns the same columns, sorted by event_time
    """
    if news.empty:
        return news
    base = news["storyId"].astype(str).str.replace(r":\d+$", "", regex=True)
    return (news.assign(_story=base).sort_values("event_time")
                .drop_duplicates("_story").drop(columns="_story").reset_index(drop=True))


def before_decision(news, shock_date, close="16:00", tz="America/New_York"):
    """
    [DAT-09] headlines published before the decision time: the market close on the shock day
    (New York time, so daylight saving is handled). Anything later wasn't known when we decide.

    news: DataFrame with event_time (UTC). returns the rows strictly before the close
    """
    if news.empty:
        return news
    cutoff = pd.Timestamp(f"{pd.Timestamp(shock_date).date()} {close}", tz=tz).tz_convert("UTC")
    t = pd.to_datetime(news["event_time"], utc=True)
    return news[t < cutoff].reset_index(drop=True)


def on_shock_day(news, shock_date, close="16:00", tz="America/New_York"):
    """
    [DAT-10] headlines from the previous trading day's close to the shock day's close (New York
    time): the news that can explain the drop. Previous trading day = previous weekday (holidays
    are not skipped, so after a holiday the window is one day longer)

    news: DataFrame with event_time (UTC). returns the rows in [previous close, shock close)
    """
    if news.empty:
        return news
    day = pd.Timestamp(shock_date).normalize()
    prev = day - pd.offsets.BDay(1)
    start = pd.Timestamp(f"{prev.date()} {close}", tz=tz).tz_convert("UTC")
    end = pd.Timestamp(f"{day.date()} {close}", tz=tz).tz_convert("UTC")
    t = pd.to_datetime(news["event_time"], utc=True)
    return news[(t >= start) & (t < end)].reset_index(drop=True)

