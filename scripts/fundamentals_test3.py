import sys
sys.path.insert(0, ".")
import pandas as pd
from src.ingest.lseg_client import lseg_session
import lseg.data as ld

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

with lseg_session():
    # 1) Quarterly earnings dates with fiscal period labels
    try:
        df = ld.get_data(
            "AAPL.O",
            ["TR.EPSActReportDate", "TR.FiscalPeriod", "TR.PeriodEndDate"],
            {"SDate": "2022-01-01", "EDate": "2026-10-01", "Frq": "FQ"})
        print("\nearnings w/ fiscal period:", df.shape)
        print(df.to_string())
    except Exception as e:
        print("earnings test FAILED:", type(e).__name__, e)

    # 2) Daily market cap
    try:
        df = ld.get_data(
            "AAPL.O", ["TR.CompanyMarketCap", "TR.CompanyMarketCap.date"],
            {"SDate": "2026-09-01", "EDate": "2026-09-15", "Frq": "D"})
        print("\ndaily market cap:", df.shape)
        print(df.to_string())
    except Exception as e:
        print("daily mktcap FAILED:", type(e).__name__, e)