import sys
sys.path.insert(0, ".")
import pandas as pd
from src.ingest.lseg_client import lseg_session
from src.ingest.lseg_news import fetch_headlines
from src.ingest.lseg_prices import fetch_prices

RIC = "AAPL.O"

with lseg_session():
    # 1) Are prices split-adjusted? AAPL did a 4-for-1 split on Aug 31, 2020
    px = fetch_prices(RIC, "2020-08-26", "2020-09-02", "daily")
    print(px[["event_time", "TRDPRC_1", "ACVOL_UNS"]].to_string())

    # 2) How far back do prices go?
    for d in ["2015-06-01", "2010-06-01", "2005-06-01", "2000-06-01"]:
        end = (pd.Timestamp(d) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        try:
            print(d, "price bars:", len(fetch_prices(RIC, d, end, "daily")))
        except Exception as e:
            print(d, "price error:", type(e).__name__)

    # 3) News quality by year
    for d in ["2026-07-01", "2025-06-02", "2024-06-03", "2023-06-01", "2022-06-01",
              "2021-06-01", "2020-06-01", "2019-06-03", "2018-06-01"]:
        end = (pd.Timestamp(d) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        n = fetch_headlines(RIC, d, end)
        if n.empty:
            print(d, "no news")
            continue
        rt = n[n["sourceCode"] == "NS:RTRS"]
        intraday = (rt["event_time"] != rt["event_time"].dt.normalize()).sum()
        print(f"{d}: total={len(n)}, reuters={len(rt)}, reuters_with_intraday_time={intraday}")