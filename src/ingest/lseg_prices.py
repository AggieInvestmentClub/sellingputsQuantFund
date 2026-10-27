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