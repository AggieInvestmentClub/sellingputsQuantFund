import sys
sys.path.insert(0, ".")
import pandas as pd
from src.ingest.lseg_client import lseg_session
import lseg.data as ld

pd.set_option("display.width", 200)
params = {"SDate": "2020-01-01", "EDate": "2026-10-01", "Frq": "FQ"}

attempts = {
    "earnings": ["TR.EPSActReportDate", "TR.EPSActReportDate.date"],
    "mktcap": ["TR.CompanyMarketCap", "TR.CompanyMarketCap.date"],
}

with lseg_session():
    for name, fields in attempts.items():
        try:
            df = ld.get_data("AAPL.O", fields, params)
            print(f"\n{name}: {df.shape}")
            print(df.to_string())
        except Exception as e:
            print(f"{name} FAILED: {type(e).__name__}: {e}")

    # save the current S&P 500 list
    spx = ld.get_data("0#.SPX", ["TR.CommonName"])
    import os
    os.makedirs("data/raw/universe", exist_ok=True)
    spx.to_csv("data/raw/universe/sp500_current.csv", index=False)
    print("\nsaved", len(spx), "constituents")