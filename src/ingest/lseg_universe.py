from pathlib import Path
import pandas as pd
import lseg.data as ld
import time 

RAW = Path("data/raw/universe")

def fetch_members(as_of, index=".SPX"):
    """[ING-10] S&P 500 member RICs as of one date"""
    df = ld.get_data(index, ["TR.IndexConstituentRIC"], {"SDate": as_of})
    col = [c for c in df.columns if c != "Instrument"][0]
    return sorted(df[col].dropna().astype(str).unique())


def save_membership(df):
    """[ING-11] save membership snapshots to data/raw/universe/sp500_membership.parquet"""
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW / "sp500_membership.parquet", index=False)


def build_membership(start="2000-01-31", end="2026-09-30", index=".SPX", retries=3):
    """[ING-12] Month-end snapshots of index membership, with retries."""
    rows = []
    for d in pd.date_range(start, end, freq="ME"):
        rics = []
        for attempt in range(retries):
            try:
                rics = fetch_members(d.strftime("%Y-%m-%d"), index)
                break
            except Exception as e:
                if attempt == retries - 1:
                    print(f"{d.date()} FAILED: {type(e).__name__}")
                else:
                    time.sleep(5 * 2 ** attempt)
        rows += [(d, r) for r in rics]
        if d.month == 12:
            print("done", d.year, "| size", len(rics))
    out = pd.DataFrame(rows, columns=["date", "ric"])
    save_membership(out)
    return out


def members_before(snapshots, shock_date):
    """[ING-13] Members from the latest snapshot strictly before the shock date."""
    d = snapshots.loc[snapshots["date"] < pd.Timestamp(shock_date), "date"].max()
    return set(snapshots.loc[snapshots["date"] == d, "ric"])

def fetch_index_changes(index=".SPX", start="2000-01-01", end="2026-10-22"):
    """[ING-14] every S&P 500 join / leave event with its date"""
    df = ld.get_data(
        index,
        ["TR.IndexJLConstituentRIC", "TR.IndexJLConstituentName",
         "TR.IndexJLConstituentRIC.change", "TR.IndexJLConstituentRIC.date"],
        {"SDate": start, "EDate": end})
    df = df.rename(columns={"Constituent RIC": "ric", "Constituent Name": "name",
                            "Change": "change", "Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[["ric", "name", "change", "date"]].dropna(subset=["ric", "date"])
    return df.drop_duplicates().sort_values("date").reset_index(drop=True)


def save_changes(df):
    """[ING-15] save join / leave events to data/raw/universe/sp500_changes.parquet"""
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW / "sp500_changes.parquet", index=False)


