import sys
sys.path.insert(0, ".")
import pandas as pd
from src.ingest.lseg_client import lseg_session
from src.ingest.lseg_prices import fetch_prices
import lseg.data as ld

pd.set_option("display.width", 200)

with lseg_session():
    # 1) Historical index joins/leaves
    try:
        df = ld.get_data(
            ".SPX",
            ["TR.IndexJLConstituentRIC", "TR.IndexJLConstituentName",
             "TR.IndexJLConstituentRIC.change", "TR.IndexJLConstituentRIC.date"],
            {"SDate": "2018-01-01", "EDate": "2026-10-01"})
        print("joins/leaves:", df.shape)
        print(df.head(15).to_string())
    except Exception as e:
        print("joins/leaves FAILED:", type(e).__name__, e)

    # 2) Unadjusted prices
    try:
        px = ld.get_history("NVDA.O", interval="daily", start="2018-06-01",
                            end="2018-06-06", adjustments="unadjusted")
        print("\nunadjusted NVDA 2018:\n", px.head().to_string())
    except Exception as e:
        print("unadjusted FAILED:", type(e).__name__, e)

    # 3) Delisted names (SVB and First Republic, removed from the S&P 500 in 2023)
    for ric in ["SIVB.O", "FRC.N"]:
        try:
            print(ric, "bars:", len(fetch_prices(ric, "2022-06-01", "2022-06-10")))
        except Exception as e:
            print(ric, "FAILED:", type(e).__name__)