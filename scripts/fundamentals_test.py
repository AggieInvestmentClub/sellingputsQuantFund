import sys
sys.path.insert(0, ".")
from src.ingest.lseg_client import lseg_session
import lseg.data as ld

with lseg_session():
    tests = {
        "earnings dates": ["TR.EPSActReportDate"],
        "market cap": ["TR.CompanyMarketCap"],
    }
    for name, fields in tests.items():
        try:
            df = ld.get_data("AAPL.O", fields,
                             {"SDate": "2020-01-01", "EDate": "2026-10-01", "Frq": "FQ"})
            print(name, "->", df.shape)
            print(df.head(8).to_string())
        except Exception as e:
            print(name, "FAILED:", type(e).__name__, e)

    try:
        df = ld.get_data("0#.SPX", ["TR.CommonName"])
        print("current S&P 500 members:", len(df))
    except Exception as e:
        print("constituents FAILED:", type(e).__name__, e)