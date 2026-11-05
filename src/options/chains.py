"""
Option contracts from Massive / Polygon flat files (OPRA day aggregates) and implied vol from price.

Flat-file contracts are OCC symbols: O:AAPL250321P00200000 = AAPL, expires 2025-03-21, put, $200 strike.
Day aggregates have open/high/low/close/volume per contract, no bid/ask and no open interest.

Conventions as in greeks.py: decimals, time in years (calendar days / 365), discrete dividends
handled by subtracting their present value from the stock price (OPT-02).
"""
import numpy as np
import pandas as pd
from scipy.special import ndtr

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


def bs_price(S, K, T, r, vol, kind="put"):
    """
    [OPT-07] Black-Scholes price of a European option; works on numbers or numpy arrays

    S: stock price net of dividends' present value (OPT-02), K: strike, T: years, r and vol: decimals
    T <= 0 or vol <= 0 gives the discounted intrinsic value
    """
    S, K, T, r, vol = (np.asarray(x, dtype=float) for x in (S, K, T, r, vol))
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = vol * np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / sq
        d2 = d1 - sq
        disc = K * np.exp(-r * np.maximum(T, 0))
        if kind == "put":
            px = disc * ndtr(-d2) - S * ndtr(-d1)
            intrinsic = np.maximum(disc - S, 0)
        elif kind == "call":
            px = S * ndtr(d1) - disc * ndtr(d2)
            intrinsic = np.maximum(S - disc, 0)
        else:
            raise ValueError(f"unknown kind: {kind}")
    return np.where((T > 0) & (vol > 0), px, intrinsic)


def implied_vol(price, S, K, T, r, kind="put", lo=0.01, hi=5.0, iters=60):
    """
    [OPT-08] implied vol that makes bs_price equal the option price (bisection, numpy arrays)

    price: option price per share; S: stock price net of dividends' PV (OPT-02); K, T, r as in OPT-07
    returns vols as decimals (0.35 = 35%); NaN where no vol in [lo, hi] fits: price at or below
    intrinsic, above the vol-hi price, T <= 0, or missing inputs
    """
    price, S, K, T, r = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (price, S, K, T, r)))
    low, high = np.full(price.shape, lo), np.full(price.shape, hi)
    p_lo, p_hi = bs_price(S, K, T, r, low, kind), bs_price(S, K, T, r, high, kind)
    ok = (T > 0) & (price > 0) & np.isfinite(price + S + K + T + r) & (price > p_lo) & (price < p_hi)
    for _ in range(iters):
        mid = 0.5 * (low + high)
        above = bs_price(S, K, T, r, mid, kind) > price
        high = np.where(above, mid, high)
        low = np.where(above, low, mid)
    iv = 0.5 * (low + high)
    iv = np.where(ok, iv, np.nan)
    return float(iv) if iv.ndim == 0 else iv


def bs_delta_vec(S, K, T, r, vol, kind="put"):
    """
    [OPT-09] Black-Scholes delta on numpy arrays (same model as OPT-03, vectorized for whole chains)

    S net of dividends' PV; returns NaN where vol or T is missing / <= 0
    """
    S, K, T, r, vol = (np.asarray(x, dtype=float) for x in (S, K, T, r, vol))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d = ndtr(d1) if kind == "call" else ndtr(d1) - 1
    return np.where((T > 0) & (vol > 0), d, np.nan)
