import sys
sys.path.insert(0, ".")
import pandas as pd
from src.ingest.lseg_client import lseg_session
from src.ingest.lseg_universe import fetch_members, build_membership
from src.ingest.lseg_prices import fetch_prices
from src.ingest.lseg_earnings import fetch_earnings_dates

with lseg_session():
    m = fetch_members("2018-02-01")
    print("2018-02-01 size", len(m))
    print("sample RICs:", m[:8])
    picks = [r for r in m if r.startswith(("TWX", "MON"))]
    print("TWX/MON RICs:", picks)
    for r in picks:
        try:
            print(r, "bars Jan 2018:", len(fetch_prices(r, "2018-01-02", "2018-01-31")))
        except Exception as e:
            print(r, "price FAILED:", type(e).__name__)
    if picks:
        er = fetch_earnings_dates(picks[:1], start="2016-01-01", end="2018-12-31")
        print("earnings rows for", picks[0], ":", len(er))

    for d in ["2000-06-30", "2005-06-30", "2010-06-30", "2015-06-30", "2020-06-30", "2026-09-30"]:
        print(d, "size", len(fetch_members(d)))

    snaps = build_membership(start="2015-04-30")
    print("\nsnapshots:", snaps["date"].nunique(), "| rows:", len(snaps))
    print(snaps.groupby("date").size().describe().to_string())