import sys
sys.path.insert(0, ".")
import pandas as pd
from src.ingest.lseg_client import lseg_session
import lseg.data as ld

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)

variants = {
    "A relative FQ0": (["TR.EPSActReportDate", "TR.EPSActual"],
                       {"SDate": 0, "EDate": -12, "Period": "FQ0", "Frq": "FQ"}),
    "B dated + Period": (["TR.EPSActReportDate", "TR.EPSActual"],
                         {"SDate": "2022-01-01", "EDate": "2026-10-01",
                          "Period": "FQ0", "Frq": "FQ"}),
    "C quarter list": (["TR.EPSActual", "TR.EPSActual.date"],
                       {"SDate": "2022-01-01", "EDate": "2026-10-01", "Frq": "FQ"}),
}

with lseg_session():
    for ric in ["AAPL.O", "MSFT.O"]:
        for name, (fields, params) in variants.items():
            try:
                df = ld.get_data(ric, fields, params)
                print(f"\n=== {ric} | {name} | shape {df.shape} | columns {list(df.columns)}")
                date_col = next((c for c in df.columns if "Date" in c or "date" in c), None)
                if date_col:
                    print("distinct dates:", df[date_col].nunique())
                print(df.head(12).to_string())
            except Exception as e:
                print(f"\n=== {ric} | {name} FAILED: {type(e).__name__}: {e}")