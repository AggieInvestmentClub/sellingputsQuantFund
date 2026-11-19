"""
AG-14: fair-value evidence for one shock, point-in-time (every value published before the shock close).

Inputs: data/raw/fundamentals/<RIC>.parquet (AG-30, I/B/E/S history) and the daily price files.
Signals (see docs/code_reference/AG_agent.md, "Fair-value gate"):
  fwd_pe          shock-day close / consensus EPS next 12 months
  pe_pctile_3y    where that P/E sits in the stock's own last 3 years (0 = cheapest)
  pe_vs_sector    fwd P/E / median fwd P/E of the same GICS industry on the shock day
  rev_30d         % change in NTM EPS consensus over the 30 days BEFORE the shock (cuts already under way?)
  target_gap      consensus price target / shock-day close - 1
  n_est           analysts in the FY1 consensus
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent.evidence import fname, load_close


@lru_cache(maxsize=1024)
def _fund(raw, ric):
    p = Path(raw) / "fundamentals" / f"{fname(ric)}.parquet"
    if not p.exists():
        return None
    f = pd.read_parquet(p)
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    return {k: g.set_index("date")["value"].sort_index() for k, g in f.groupby("field")}


def _asof(s, date):
    """last value dated strictly before the shock day (estimates published that day may come after the close)"""
    if s is None:
        return np.nan
    s = s[s.index < pd.Timestamp(date).normalize()].dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def fwd_pe_on(raw, ric, date):
    f, px = _fund(str(raw), ric), load_close(raw, ric)
    if f is None or px is None or pd.Timestamp(date) not in px.index:
        return np.nan
    eps = _asof(f.get("eps_ntm"), date)
    return float(px.loc[pd.Timestamp(date), "close"] / eps) if eps and eps > 0 else np.nan


def valuation_context(raw, ric, date, gics=None, members=None, max_peers=40):
    """[AG-14] dict of the signals above (NaN where data is missing), or None if no fundamentals file"""
    date = pd.Timestamp(date).normalize()
    f, px = _fund(str(raw), ric), load_close(raw, ric)
    if f is None or px is None or date not in px.index:
        return None
    S = float(px.loc[date, "close"])
    eps = f.get("eps_ntm")
    pe = fwd_pe_on(raw, ric, date)
    pct = np.nan
    if eps is not None and np.isfinite(pe):
        e = eps[(eps.index < date) & (eps.index >= date - pd.Timedelta(days=3 * 365))]
        c = px["close"].reindex(e.index, method="ffill")
        hist = (c / e.where(e > 0)).dropna()
        if len(hist) >= 26:
            pct = float((hist < pe).mean())
    e_now, e_30 = _asof(eps, date), _asof(eps, date - pd.Timedelta(days=30))
    rev = (e_now / e_30 - 1) if np.isfinite(e_now) and np.isfinite(e_30) and e_30 > 0 else np.nan
    tgt = _asof(f.get("target"), date)
    sector = np.nan
    if gics is not None and np.isfinite(pe):
        g = gics.set_index("ric")
        if ric in g.index and pd.notna(g.loc[ric, "industry"]):
            peers = [p for p in g.index[g["industry"] == g.loc[ric, "industry"]] if p != ric
                     and (members is None or p in members)][:max_peers]
            pes = [fwd_pe_on(raw, p, date) for p in peers]
            pes = [x for x in pes if np.isfinite(x) and 0 < x < 200]
            if len(pes) >= 3:
                sector = pe / float(np.median(pes))
    return {"fwd_pe": pe, "pe_pctile_3y": pct, "pe_vs_sector": sector, "rev_30d": rev,
            "target_gap": tgt / S - 1 if np.isfinite(tgt) else np.nan, "n_est": _asof(f.get("n_est"), date)}


def gate(v, max_pctile=0.5, max_vs_sector=1.2, max_cut=-0.02):
    """
    [AG-15] fair-value gate (thresholds from the literature, not tuned on our data)
    returns (passes, reasons). Missing data never fails the gate on its own; it is reported instead.
    """
    if not v:
        return True, ["no fundamentals file"]
    if not np.isfinite(v["fwd_pe"]) and not np.isfinite(v["n_est"]):
        return True, ["no estimates"]
    why = []
    if np.isfinite(v["pe_pctile_3y"]) and v["pe_pctile_3y"] > max_pctile:
        why.append(f"forward P/E above its 3-year median (percentile {v['pe_pctile_3y']:.0%})")
    if np.isfinite(v["pe_vs_sector"]) and v["pe_vs_sector"] > max_vs_sector:
        why.append(f"forward P/E {v['pe_vs_sector']:.1f}x the industry median")
    if np.isfinite(v["rev_30d"]) and v["rev_30d"] < max_cut:
        why.append(f"EPS consensus already cut {v['rev_30d']:.1%} in the 30 days before")
    return (not why), why


def valuation_lines(v):
    """[AG-16] the valuation evidence as prompt lines (numbers only, no names)"""
    if not v:
        return ["- Valuation: not available"]
    f = lambda x, s: "n/a" if not np.isfinite(x) else s.format(x)
    return [f"- Forward P/E (next 12 months): {f(v['fwd_pe'], '{:.1f}')}; percentile of its own last 3 years: "
            f"{f(v['pe_pctile_3y'], '{:.0%}')}; vs industry median: {f(v['pe_vs_sector'], '{:.2f}x')}",
            f"- EPS consensus change over the 30 days before Day 0: {f(v['rev_30d'], '{:+.1%}')}; "
            f"analysts: {f(v['n_est'], '{:.0f}')}; consensus price target vs Day 0 close: {f(v['target_gap'], '{:+.0%}')}"]
