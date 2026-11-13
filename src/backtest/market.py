"""
Market data the backtest needs per event: stock bars and put prices.

Two put pricers with the same methods (expiries, chain, atm_iv, leg_marks):
- PanelPricer: real OPRA day-aggregate closes from scripts/build_option_panel.py
  (data/processed/options/event_puts.parquet). Day aggregates have no quotes, so bid / ask are
  modeled as close -/+ a half spread.
- ProxyPricer: Black-Scholes with IV = iv_mult x the stock's 60-day realized vol before the shock,
  plus a simple put skew. For wiring and sanity checks only: it has no real IV crush or skew,
  so its P&L says nothing about the real strategy.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.options.greeks import bs_delta_vec, bs_price

RAW = Path("data/raw")


def fname(ric):
    return ric.replace(".", "_").replace("^", "~")


def load_stock(ric):
    """
    [BT-01] daily bars for one stock: unadjusted close / low / high / volume (strikes are as traded)
    and the split-adjusted return (for realized vol)

    returns DataFrame indexed by date: close, low, high, volume, ret. None if no price file
    """
    pu = RAW / "prices" / "daily" / f"{fname(ric)}_unadj.parquet"
    pa = RAW / "prices" / "daily" / f"{fname(ric)}_adj.parquet"
    if not pu.exists() or not pa.exists():
        return None
    u, a = pd.read_parquet(pu), pd.read_parquet(pa)
    df = pd.DataFrame({"close": pd.to_numeric(u["TRDPRC_1"], errors="coerce"),
                       "low": pd.to_numeric(u["LOW_1"], errors="coerce"),
                       "high": pd.to_numeric(u["HIGH_1"], errors="coerce"),
                       "volume": pd.to_numeric(u["ACVOL_UNS"], errors="coerce")})
    df["ret"] = pd.to_numeric(a["TRDPRC_1"], errors="coerce").pct_change(fill_method=None)
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated()].sort_index()
    df["low"] = df["low"].fillna(df["close"])
    return df.dropna(subset=["close"])


def load_rates():
    r = pd.read_parquet(RAW / "rates" / "treasury_yields.parquet")["US1MT=RR"]
    r.index = pd.to_datetime(r.index).normalize()
    return r.sort_index().ffill()


def _quote(mid, half_spread, min_half=0.025):
    """modeled bid / ask around a price (no quotes in day aggregates)"""
    h = np.maximum(np.asarray(mid, dtype=float) * half_spread, min_half)
    return np.maximum(mid - h, 0.0), mid + h


class ProxyPricer:
    def __init__(self, rates, iv_mult=1.3, skew=1.0, iv_floor=0.15, half_spread=0.05):
        self.rates, self.iv_mult, self.skew = rates, iv_mult, skew
        self.iv_floor, self.half_spread = iv_floor, half_spread
        self._base = {}

    def start(self, event_id, stock, shock_date):
        """
        [BT-02] set up the proxy for one event: ATM IV = iv_mult x 60-day realized vol before the shock
        (floored at iv_floor). Call once per event before the other methods
        """
        pre = stock["ret"][stock.index < pd.Timestamp(shock_date)].dropna().iloc[-60:]
        rv = pre.std() * math.sqrt(252) if len(pre) >= 20 else np.nan
        self._base[event_id] = max(self.iv_floor, self.iv_mult * rv) if np.isfinite(rv) else self.iv_floor
        self._stock = stock

    def _r(self, date):
        r = self.rates.asof(pd.Timestamp(date))
        return 0.04 if pd.isna(r) else float(r)

    def _iv(self, event_id, S, K):
        return self._base[event_id] * np.clip(1 + self.skew * np.log(S / np.asarray(K, dtype=float)), 0.5, 3)

    def expiries(self, event_id, date):
        d = pd.Timestamp(date)
        first = d + pd.Timedelta(days=int((4 - d.weekday()) % 7 or 7))
        return list(pd.date_range(first, d + pd.Timedelta(days=60), freq="7D"))

    def atm_iv(self, event_id, date):
        return self._base[event_id]

    def chain(self, event_id, date, expiry):
        S = float(self._stock.at[pd.Timestamp(date), "close"])
        step = 0.5 if S < 25 else 1 if S < 50 else 2.5 if S < 200 else 5 if S < 500 else 10
        K = np.arange(math.floor(S * 0.6 / step) * step, S * 1.05, step)
        K = K[K > 0]
        T = max((pd.Timestamp(expiry) - pd.Timestamp(date)).days, 0) / 365
        iv, r = self._iv(event_id, S, K), self._r(date)
        mid = bs_price(S, K, T, r, iv)
        bid, ask = _quote(mid, self.half_spread)
        return pd.DataFrame({"strike": K, "delta": bs_delta_vec(S, K, T, r, iv), "mid": mid,
                             "bid": bid, "ask": ask, "volume": 1.0, "iv": iv})

    def leg_marks(self, event_id, date, expiry, strikes):
        S = float(self._stock.at[pd.Timestamp(date), "close"])
        K = np.asarray(strikes, dtype=float)
        T = max((pd.Timestamp(expiry) - pd.Timestamp(date)).days, 0) / 365
        iv, r = self._iv(event_id, S, K), self._r(date)
        mid = bs_price(S, K, T, r, iv)
        bid, ask = _quote(mid, self.half_spread)
        return mid, bid, ask, bs_delta_vec(S, K, T, r, iv)


class PanelPricer:
    def __init__(self, panel, rates, half_spread=0.05):
        p = panel.copy()
        p["date"] = pd.to_datetime(p["date"]).dt.normalize()
        p["expiry"] = pd.to_datetime(p["expiry"]).dt.normalize()
        self.by_event = dict(tuple(p.groupby("event_id")))
        self.rates, self.half_spread = rates, half_spread

    def start(self, event_id, stock, shock_date):
        """
        [BT-03] set up the real-panel pricer for one event (its puts from event_puts.parquet);
        legs with no print on a day are re-priced with Black-Scholes at their last implied vol
        """
        self._p = self.by_event.get(event_id, pd.DataFrame(columns=["date", "expiry", "strike"]))
        self._stock = stock

    def expiries(self, event_id, date):
        d = self._p[self._p["date"] == pd.Timestamp(date)]
        return sorted(d["expiry"].unique())

    def atm_iv(self, event_id, date):
        d = self._p[self._p["date"] == pd.Timestamp(date)]
        dte = (d["expiry"] - pd.Timestamp(date)).dt.days
        atm = d[(dte >= 14) & (dte <= 60) & d["delta"].abs().between(0.35, 0.65)]["iv"].dropna()
        return float(atm.median()) if len(atm) else None

    def chain(self, event_id, date, expiry):
        d = self._p[(self._p["date"] == pd.Timestamp(date)) & (self._p["expiry"] == pd.Timestamp(expiry))]
        d = d.dropna(subset=["close", "delta"])
        bid, ask = _quote(d["close"].to_numpy(), self.half_spread)
        return pd.DataFrame({"strike": d["strike"].to_numpy(), "delta": d["delta"].to_numpy(),
                             "mid": d["close"].to_numpy(), "bid": bid, "ask": ask,
                             "volume": d["volume"].to_numpy(), "iv": d["iv"].to_numpy()})

    def leg_marks(self, event_id, date, expiry, strikes):
        date, expiry = pd.Timestamp(date), pd.Timestamp(expiry)
        S = float(self._stock.at[date, "close"])
        T = max((expiry - date).days, 0) / 365
        r = self.rates.asof(date)
        r = 0.04 if pd.isna(r) else float(r)
        legs = self._p[(self._p["expiry"] == expiry) & (self._p["date"] <= date)]
        mids, deltas = [], []
        for K in strikes:
            rows = legs[legs["strike"] == K].sort_values("date")
            today = rows[rows["date"] == date]
            if len(today) and pd.notna(today["close"].iloc[0]):
                mids.append(float(today["close"].iloc[0]))
                deltas.append(today["delta"].iloc[0])
                continue
            iv = rows["iv"].dropna()
            vol = float(iv.iloc[-1]) if len(iv) else 0.0
            mids.append(float(bs_price(S, K, T, r, vol)))
            deltas.append(float(bs_delta_vec(S, K, T, r, vol)) if vol > 0 else (-1.0 if S < K else 0.0))
        mid = np.array(mids)
        bid, ask = _quote(mid, self.half_spread)
        return mid, bid, ask, np.array(deltas, dtype=float)
