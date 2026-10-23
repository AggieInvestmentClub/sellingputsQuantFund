from pathlib import Path
import pandas as pd
import lseg.data as ld

RAW = Path("data/raw/earnings")


def fetch_earnings_dates(rics, start="2017-01-01", end="2027-01-31", batch=50):
    frames = []
    for i in range(0, len(rics), batch):
        chunk = list(rics[i:i + batch])
        try:
            df = ld.get_data(chunk, ["TR.EPSActReportDate"],
                             {"SDate": start, "EDate": end,
                              "Period": "FQ0", "Frq": "FQ"})
            frames.append(df)
        except Exception as e:
            print(f"batch starting {i} failed: {type(e).__name__}: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).rename(columns={"Instrument": "ric",
                                            "Report Date": "report_time"})
    out = out.dropna(subset=["report_time"]).drop_duplicates()
    out["report_time"] = pd.to_datetime(out["report_time"])
    out["report_date"] = out["report_time"].dt.normalize()
    out["after_close"] = out["report_time"].dt.hour >= 16   # assumes ET
    return out.sort_values(["ric", "report_time"]).reset_index(drop=True)


def save_earnings(df):
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW / "earnings_dates.parquet", index=False)