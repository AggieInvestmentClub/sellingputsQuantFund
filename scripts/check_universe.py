import sys
sys.path.insert(0, ".")
import pandas as pd
from src.ingest.lseg_client import lseg_session
from src.ingest.lseg_universe import fetch_index_changes, save_changes, members_on
from src.ingest.lseg_prices import fetch_prices
from src.ingest.lseg_earnings import fetch_earnings_dates

current_df = pd.read_csv("data/raw/universe/sp500_current.csv")
print(current_df.head(3).to_string())
current = set(current_df["Instrument"])

with lseg_session():
    ch = fetch_index_changes(start="2000-01-01")
    save_changes(ch)
    print("\nrows:", len(ch), "| range:", ch["date"].min().date(), "to", ch["date"].max().date())
    print(ch.groupby([ch["date"].dt.year, "change"]).size().unstack(fill_value=0).to_string())

    print("\nreconstructed index size (should stay near 500):")
    for d in ["2018-02-01", "2020-01-01", "2022-01-01", "2024-01-01", "2026-01-01"]:
        print(d, len(members_on(current, ch, d)))

    print("\nprices for delisted RICs:")
    leavers = ch[(ch["change"] == "Leaver") & ch["ric"].str.contains(r"\^")].head(3)
    for _, r in leavers.iterrows():
        s = (r["date"] - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
        e = (r["date"] - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        try:
            print(r["ric"], "bars:", len(fetch_prices(r["ric"], s, e)))
        except Exception as ex:
            print(r["ric"], "FAILED:", type(ex).__name__)

    print("\nearnings from 2018 for AAPL:")
    er = fetch_earnings_dates(["AAPL.O"], start="2018-01-01")
    print(er.groupby(er["report_date"].dt.year).size().to_string())