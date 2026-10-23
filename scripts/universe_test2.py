import sys
sys.path.insert(0, ".")
import pandas as pd
from src.ingest.lseg_client import lseg_session
from src.ingest.lseg_prices import fetch_prices
import lseg.data as ld

variants = {
    "A SDate": lambda d: {"SDate": d},
    "B SDate=EDate": lambda d: {"SDate": d, "EDate": d},
}

with lseg_session():
    for d in ["2018-02-01", "2022-01-01"]:
        for name, params in variants.items():
            try:
                df = ld.get_data(".SPX", ["TR.IndexConstituentRIC"], params(d))
                rics = df[df.columns[-1]].dropna().astype(str)
                has = lambda p: rics.str.startswith(p).any()
                print(d, name, "size", len(rics),
                      "| TWX:", has("TWX"), "MON:", has("MON"), "TSLA:", has("TSLA"))
            except Exception as e:
                print(d, name, "FAILED:", type(e).__name__, str(e)[:120])

    try:
        print("MSFT.OQ bars:", len(fetch_prices("MSFT.OQ", "2026-09-01", "2026-09-15")))
    except Exception as e:
        print("MSFT.OQ FAILED:", type(e).__name__)