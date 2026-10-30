from pathlib import Path
import pandas as pd
import lseg.data as ld

RAW = Path("data/raw/prices")


def fetch_prices(ric: str, start: str, end: str, interval: str = "daily") -> pd.DataFrame:
    """[ING-02] daily price bars for one RIC, starting 1 day early so the first return can be computed"""
    start_adj = (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = ld.get_history(ric, interval=interval, start=start_adj, end=end)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "event_time"
    df["ric"] = ric
    return df.reset_index()

def save_prices(ric: str, interval: str, df: pd.DataFrame) -> Path:
    """[ING-03] save price bars to data/raw/prices/<RIC>_<interval>.parquet"""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{ric.replace('.', '_')}_{interval}.parquet"
    df.to_parquet(path, index=False)
    return path

def fetch_daily_history(ric: str, start: str, end: str, adjustments: str = None) -> pd.DataFrame:
    """
    [ING-19] daily bars for one RIC with a plain date index (no timezone), ready for STR / EVT code

    adjustments: None = split-adjusted (use for returns), "unadjusted" = as traded (price floor, strikes)
    LSEG's adjusted prices cover splits and spin-offs, NOT regular dividends (AAPL 2020: 506.09 / 4 = 126.52;
    A.N 2014-11-03 Keysight spin-off: factor 1.40, see DAT-07)
    one request covers 2014-2026 (3,223 rows tested)
    """
    kwargs = {"interval": "daily", "start": start, "end": end}
    if adjustments:
        kwargs["adjustments"] = adjustments
    df = ld.get_history(ric, **kwargs)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df.columns.name = None
    return df[~df.index.duplicated(keep="last")]
