"""
AG: evidence the agent sees for one shock, built only from data known at the shock-day close.

The old LLM filter (LLM-03) saw only the news. The backtest found its "not justified" label did no
better than "has Reuters news", so the agent adds market evidence a human analyst would check:
the stock's trend before the news, whether its industry fell with it, and whether earnings land
inside the trade window. Everything here is point-in-time: slices end at the shock date.

Pure functions on data already on disk (no downloads). Paths follow scripts/run_llm_filter.py.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.news import before_decision, first_versions


def fname(ric):
    """file stem used across data/raw ("AAPL.OQ" -> "AAPL_OQ", "MRO.N^K24" -> "MRO_N~K24")"""
    return ric.replace(".", "_").replace("^", "~")


def load_close(raw, ric):
    """cached: peers are re-read for every shock in their industry (callers must not modify the result)"""
    return _load_close(str(raw), ric)


@lru_cache(maxsize=2048)
def _load_close(raw, ric):
    """
    [AG-01] split-adjusted daily close (TRDPRC_1) and volume from data/raw/prices/daily/<RIC>_adj.parquet
    returns DataFrame (close, volume) with a date index, or None if there is no file
    """
    p = Path(raw) / "prices" / "daily" / f"{fname(ric)}_adj.parquet"
    if not p.exists():
        return None
    px = pd.read_parquet(p)
    if "date" in px.columns:
        px = px.set_index("date")
    px.index = pd.to_datetime(px.index).normalize()
    out = pd.DataFrame({"close": px["TRDPRC_1"].astype(float),
                        "volume": px["ACVOL_UNS"].astype(float) if "ACVOL_UNS" in px else np.nan})
    return out[~out.index.duplicated(keep="last")].sort_index().dropna(subset=["close"])


def price_context(px, date):
    """
    [AG-02] the stock's state up to the shock-day close

    px: AG-01 output; date: shock date. Returns dict of fractions (None if not enough history):
    ret_shock (shock-day return), ret_5d / ret_21d / ret_63d (BEFORE the shock day, so they show the
    trend going in), off_52w_high (pre-shock close / 52-week high - 1), rv_60d (annualized, pre-shock),
    volume_x (shock-day volume / 20-day average), close_vs_200dma (pre-shock)
    """
    date = pd.Timestamp(date).normalize()
    if px is None:
        return None
    h = px.loc[:date]
    if len(h) < 70 or h.index[-1] != date:
        return None
    c, r = h["close"], h["close"].pct_change()
    pre = c.iloc[:-1]                                       # up to the day before the shock

    def back(n):
        return float(pre.iloc[-1] / pre.iloc[-1 - n] - 1) if len(pre) > n else None

    vol = h["volume"]
    return {
        "ret_shock": float(r.iloc[-1]),
        "ret_5d": back(5), "ret_21d": back(21), "ret_63d": back(63),
        "off_52w_high": float(pre.iloc[-1] / pre.iloc[-252:].max() - 1),
        "close_vs_200dma": float(pre.iloc[-1] / pre.iloc[-200:].mean() - 1) if len(pre) >= 200 else None,
        "rv_60d": float(r.iloc[-61:-1].std() * np.sqrt(252)),
        "volume_x": (float(vol.iloc[-1] / vol.iloc[-21:-1].mean())
                     if vol.notna().iloc[-21:].all() and vol.iloc[-21:-1].mean() > 0 else None),
    }


def abnormal_on(raw, ric, date):
    """
    [AG-01b] shock-day abnormal return ret - beta * SPY ret with the EVT-01 rolling beta (same numbers
    as events.csv). For live use, where the shock is not in events.csv yet.
    returns (abnormal, ret, spy_ret) or None if the price files don't reach the date
    """
    from src.events.build_events import rolling_beta
    date = pd.Timestamp(date).normalize()
    px, spy = load_close(raw, ric), load_close(raw, "SPY")
    if px is None or spy is None or date not in px.index or date not in spy.index:
        return None
    r, m = px["close"].pct_change(), spy["close"].pct_change()
    beta = rolling_beta(r, m).get(date)
    if beta is None or not np.isfinite(beta):
        return None
    return float(r[date] - beta * m[date]), float(r[date]), float(m[date])


def members_on(master, date, grace_days=62):
    """RICs in the universe on `date` (security_master span; last_member is a month-end snapshot, so allow grace)"""
    date = pd.Timestamp(date)
    live = (master["first_member"] <= date) & (master["last_member"] >= date - pd.Timedelta(days=grace_days))
    return set(master.loc[live, "ric"])


def peer_context(raw, ric, date, gics, members, min_peers=3, max_peers=40):
    """
    [AG-03] same-day returns of the stock's GICS industry (industry, else industry_group): did the
    whole industry fall (sector repricing, usually justified) or only this stock?

    gics: data/raw/reference/gics.parquet; members: RICs in the universe that day
    returns dict: level ("industry" / "industry_group"), n, median, share_down_3pct, or None
    """
    date = pd.Timestamp(date).normalize()
    g = gics.set_index("ric")
    if ric not in g.index:
        return None
    for level in ("industry", "industry_group"):
        key = g.loc[ric, level]
        if pd.isna(key):
            continue
        peers = [p for p in g.index[g[level] == key] if p != ric and p in members][:max_peers]
        rets = []
        for p in peers:
            px = load_close(raw, p)
            if px is None or date not in px.index:
                continue
            i = px.index.get_loc(date)
            if i > 0:
                rets.append(px["close"].iloc[i] / px["close"].iloc[i - 1] - 1)
        if len(rets) >= min_peers:
            rets = np.array(rets)
            return {"level": level, "n": int(len(rets)), "median": float(np.median(rets)),
                    "share_down_3pct": float((rets <= -0.03).mean())}
    return None


def earnings_context(earnings, ric, date, window_days=35):
    """
    [AG-04] where the shock sits in the earnings cycle

    earnings: data/raw/earnings/earnings_dates.parquet (ric, report_date)
    returns days_since_last (calendar), next_in_window (a report within window_days after the shock,
    i.e. inside a 2-5 week spread). Report dates are announced weeks ahead, so the next one counts as known.
    """
    date = pd.Timestamp(date).normalize()
    d = pd.to_datetime(earnings.loc[earnings["ric"] == ric, "report_date"]).sort_values()
    past, fut = d[d <= date], d[d > date]
    return {"days_since_last": int((date - past.iloc[-1]).days) if len(past) else None,
            "next_in_window": bool(len(fut) and (fut.iloc[0] - date).days <= window_days)}


def market_cap_before(raw, ric, date):
    """[AG-05] last market cap strictly before the shock day (same rule as run_llm_filter.cap_before)"""
    p = Path(raw) / "market_cap" / f"{fname(ric)}.parquet"
    if not p.exists():
        return np.nan
    s = pd.read_parquet(p).iloc[:, 0]
    s.index = pd.to_datetime(s.index).normalize()
    s = s[s.index < pd.Timestamp(date).normalize()].dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def event_news(raw, ric, date):
    """
    [AG-06] Reuters headlines and story texts for one shock, cut to before the shock-day close
    (DAT-08 first versions, DAT-09 before_decision), from the ING-23 / ING-24 files
    """
    tag = f"{fname(ric)}_{pd.Timestamp(date):%Y%m%d}.parquet"
    hp = Path(raw) / "news" / "reuters_events" / tag
    sp = Path(raw) / "news" / "reuters_stories" / tag
    heads = pd.read_parquet(hp) if hp.exists() else pd.DataFrame(columns=["storyId", "event_time", "headline"])
    heads = before_decision(first_versions(_drop_late_ids(heads, date)), date) if len(heads) else heads
    stories = pd.read_parquet(sp) if sp.exists() else pd.DataFrame(columns=["event_time", "text"])
    stories = before_decision(_drop_late_ids(stories, date), date) if len(stories) else stories
    return heads, stories


def _drop_late_ids(news, date):
    """
    second look-ahead guard: drop rows whose storyId carries a date AFTER the shock day
    ("urn:newsml:...:20240626:nNR...") even if event_time says otherwise. Found in the old
    data/raw/news/events files: stories from weeks later stamped "shock day 00:00" and marked before_decision.
    """
    if "storyId" not in news.columns or not len(news):
        return news
    d = news["storyId"].astype(str).str.extract(r":(\d{8}):")[0]
    late = d.notna() & (d > pd.Timestamp(date).strftime("%Y%m%d"))
    return news[~late].reset_index(drop=True)
