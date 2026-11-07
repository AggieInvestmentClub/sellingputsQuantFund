"""
Option contract names from Massive / Polygon flat files (OPRA day aggregates).
Pricing, implied vol and delta for whole chains are in greeks.py (OPT-07..09).

Flat-file contracts are OCC symbols: O:AAPL250321P00200000 = AAPL, expires 2025-03-21, put, $200 strike.
Day aggregates have open/high/low/close/volume per contract, no bid/ask and no open interest.
"""
import numpy as np
import pandas as pd

OCC_PATTERN = r"^O:([A-Z0-9.]+?)(\d{6})([CP])(\d{8})$"


def parse_occ(tickers):
    """
    [OPT-05] split OCC option symbols into root, expiry, kind and strike

    tickers: Series of symbols like "O:AAPL250321P00200000"
    returns a DataFrame (same index): root, expiry (Timestamp), kind ("put"/"call"), strike (float);
    rows that don't match the OCC pattern come back NaN
    """
    p = pd.Series(tickers, dtype="string").str.extract(OCC_PATTERN)
    out = pd.DataFrame(index=p.index)
    out["root"] = p[0]
    out["expiry"] = pd.to_datetime(p[1], format="%y%m%d", errors="coerce")
    out["kind"] = p[2].map({"P": "put", "C": "call"})
    out["strike"] = pd.to_numeric(p[3], errors="coerce") / 1000
    return out


def occ_root(ticker):
    """
    [OPT-06] OCC option root for a stock ticker (Massive ticker on that date, DAT-03)

    share classes lose the dot: BRK.B -> BRKB, BF.B -> BFB; returns None for a missing ticker
    """
    if ticker is None or (isinstance(ticker, float) and np.isnan(ticker)):
        return None
    return str(ticker).replace(".", "").replace("/", "").upper()
