"""
Black-Scholes option math: one model for the whole repo.
OPT-01..04: single options (filling LSEG's missing DELTA, 2018-2021).
OPT-07..09: numpy versions for whole chains (option panel, EVT-12).

Method validated against LSEG's own DELTA on 2022 data (COF + AAPL puts):
mean error ~0.001, max ~0.04 (at-the-money in the last days before expiry).

Conventions (see CONTRIBUTING.md):
- all rates, vols and yields are DECIMALS: 0.25 = 25%, 0.0191 = 1.91%
  (LSEG IMP_VOLT and B_YLD_1 are in percent, divide by 100 first)
- time is in years, calendar days / 365
- dividends are discrete cash amounts per share, handled by subtracting their
  present value from the stock price (escrowed dividend model)
"""
import math

import numpy as np
import pandas as pd
from scipy.special import ndtr


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def years_to_expiry(as_of, expiry):
    """[OPT-01] time to expiry in years (calendar days / 365), 0 if already expired"""
    days = (pd.Timestamp(expiry).normalize() - pd.Timestamp(as_of).normalize()).days
    return max(days, 0) / 365


def pv_dividends(as_of, expiry, dividends, r):
    """
    [OPT-02] present value of cash dividends going ex between as_of (exclusive) and expiry (inclusive)

    dividends: iterable of (ex_date, cash_amount) per share, e.g. [("2022-05-12", 0.60)]
    r: risk-free rate (decimal)
    """
    start, end = pd.Timestamp(as_of).normalize(), pd.Timestamp(expiry).normalize()
    pv = 0.0
    for ex_date, amount in dividends:
        ex = pd.Timestamp(ex_date).normalize()
        if start < ex <= end:
            pv += amount * math.exp(-r * (ex - start).days / 365)
    return pv


def bs_delta(S, K, T, r, vol, q=0.0, kind="put"):
    """
    [OPT-03] Black-Scholes delta of a European option

    S: stock price (already dividend-adjusted if using discrete dividends)
    K: strike, T: years to expiry, r: risk-free rate, vol: implied vol, q: continuous dividend yield
    kind: "put" (negative delta) or "call"
    at or after expiry (T <= 0) returns the expiry delta: put -1 if S < K else 0
    returns None if any input is unusable (missing, S/K/vol <= 0)
    """
    if kind not in ("put", "call"):
        raise ValueError(f"unknown kind: {kind}")
    if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in (S, K, T, r, vol)):
        return None
    if S <= 0 or K <= 0:
        return None
    if T <= 0:
        if kind == "put":
            return -1.0 if S < K else 0.0
        return 1.0 if S > K else 0.0
    if vol <= 0:
        return None

    d1 = (math.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * math.sqrt(T))
    if kind == "call":
        return math.exp(-q * T) * _norm_cdf(d1)
    return -math.exp(-q * T) * _norm_cdf(-d1)


def delta_from_iv(S, K, as_of, expiry, iv, r, dividends=(), kind="put"):
    """
    [OPT-04] delta from an option's implied vol, the way we fill LSEG's missing DELTA

    S: UNADJUSTED stock close on as_of (strikes are as-traded, so the price must be too)
    iv, r: decimals (LSEG IMP_VOLT / 100, US1MT=RR B_YLD_1 / 100)
    dividends: (ex_date, cash_amount) pairs, subtracted from S at present value
    ex: COF $115 put on 2022-05-18 -> about -0.489 (LSEG reported -0.4884)
    returns None if inputs are unusable
    raises ValueError if iv or r look like percents (iv > 5 or r > 0.25), the most common unit mistake
    (can't catch every case: a 0.05% yield passed as 0.05 looks like a valid 5%)
    """
    if iv is not None and not (isinstance(iv, float) and math.isnan(iv)) and iv > 5:
        raise ValueError(f"iv={iv} looks like a percent, pass decimals (divide LSEG IMP_VOLT by 100)")
    if r is not None and r > 0.25:
        raise ValueError(f"r={r} looks like a percent, pass decimals (divide B_YLD_1 by 100)")

    T = years_to_expiry(as_of, expiry)
    S_adj = S - pv_dividends(as_of, expiry, dividends, r or 0.0)
    return bs_delta(S_adj, K, T, r, iv, q=0.0, kind=kind)


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
