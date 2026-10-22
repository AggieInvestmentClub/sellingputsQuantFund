import sys
sys.path.insert(0, ".")
import pandas as pd
from src.ingest.lseg_client import lseg_session
from src.ingest.lseg_news import fetch_headlines
from src.ingest.lseg_prices import fetch_prices

RIC = "AAPL.O"
DATES = ["2026-07-01", "2026-01-05", "2025-06-02", "2024-06-03",
         "2022-06-01", "2020-06-01", "2018-06-01"]

with lseg_session():
    for d in DATES:
        end = (pd.Timestamp(d) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        try:
            n = len(fetch_headlines(RIC, d, end))
        except Exception as e:
            n = f"error: {type(e).__name__}"
        try:
            p = len(fetch_prices(RIC, d, end, "daily"))
        except Exception as e:
            p = f"error: {type(e).__name__}"
        print(f"{d}: news={n}, daily prices={p}")