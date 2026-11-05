"""
Per-event option panel: for each shock, the puts that could be traded (entry day +1 .. +max_wait,
14-35 days to expiry at entry) and their daily prices until expiry, taken from OPRA day aggregates.

The backtest reads this panel instead of the raw flat files (one ~3 MB file per trading day).
"""
import numpy as np
import pandas as pd

from src.data.security_master import massive_ticker_on
from src.options.chains import bs_delta_vec, implied_vol, occ_root


def event_windows(events, tickers, first_day, last_day, max_wait=10, min_dte=14, max_dte=35):
    """
    [EVT-09] which option root, dates and expiries each event needs

    events: EVT-05 output (ric, date); tickers: Massive ticker history (ING-18) for DAT-03
    first_day / last_day: first and last day with option files
    entry can be day +1 .. +max_wait (STR-3.5b), so expiries run from day +1 + min_dte calendar
    days to day +max_wait + max_dte; data is needed from the shock day until the latest expiry
    returns one row per event: event_id, ric, date, root, min_expiry, max_expiry, window_end,
    complete (False if window_end is after last_day); events before first_day are dropped
    """
    ev = events[["ric", "date"]].copy()
    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
    ev["event_id"] = ev["ric"] + "|" + ev["date"].dt.strftime("%Y-%m-%d")
    ev = ev[ev["date"] >= pd.Timestamp(first_day)].copy()
    ev["root"] = [occ_root(massive_ticker_on(tickers, r, d)) for r, d in zip(ev["ric"], ev["date"])]
    days = ev["date"].to_numpy().astype("datetime64[D]")
    plus = lambda n: pd.to_datetime(np.busday_offset(days, n, roll="forward"))   # n business days later
    ev["min_expiry"] = plus(1) + pd.Timedelta(days=min_dte)
    ev["max_expiry"] = plus(max_wait) + pd.Timedelta(days=max_dte)
    ev["window_end"] = ev["max_expiry"]
    ev["complete"] = ev["window_end"] <= pd.Timestamp(last_day)
    cols = ["event_id", "ric", "date", "root", "min_expiry", "max_expiry", "window_end", "complete"]
    return ev[cols].reset_index(drop=True)


def select_event_puts(day, date, windows):
    """
    [EVT-10] one day's puts for every event whose window covers that day

    day: one flat file with OPT-05 columns added (root, expiry, kind, strike) plus open, high,
    low, close, volume, transactions; date: that file's trading day; windows: EVT-09 output
    returns rows: event_id, ric, date, ticker, root, expiry, strike, open, high, low, close,
    volume, transactions (puts with min_expiry <= expiry <= max_expiry and expiry >= date)
    """
    date = pd.Timestamp(date).normalize()
    w = windows[(windows["date"] <= date) & (windows["window_end"] >= date)]
    if w.empty:
        return pd.DataFrame()
    puts = day[(day["kind"] == "put") & day["root"].isin(set(w["root"]))]
    m = puts.merge(w[["event_id", "ric", "root", "min_expiry", "max_expiry"]], on="root")
    m = m[(m["expiry"] >= m["min_expiry"]) & (m["expiry"] <= m["max_expiry"]) & (m["expiry"] >= date)]
    m = m.assign(date=date)
    cols = ["event_id", "ric", "date", "ticker", "root", "expiry", "strike",
            "open", "high", "low", "close", "volume", "transactions"]
    return m[cols].reset_index(drop=True)


def root_volume(day, date, roots):
    """
    [EVT-11] total option volume per underlying for one day (calls + puts), for STR-1.4a

    returns: date, root, opt_volume, put_volume, contracts_traded
    """
    d = day[day["root"].isin(set(roots))]
    g = d.groupby("root").agg(opt_volume=("volume", "sum"), contracts_traded=("volume", "size"))
    g["put_volume"] = d[d["kind"] == "put"].groupby("root")["volume"].sum()
    g = g.fillna({"put_volume": 0}).reset_index()
    g.insert(0, "date", pd.Timestamp(date).normalize())
    return g[["date", "root", "opt_volume", "put_volume", "contracts_traded"]]


def add_iv_delta(panel, closes, rates, dividends):
    """
    [EVT-12] implied vol and delta for every put in the panel, from its closing price

    closes: unadjusted stock close per (ric, date), columns ric, date, close (strikes are as traded)
    rates: Series of risk-free rates (decimal) by date, e.g. US1MT=RR from ING-22 (forward-filled)
    dividends: ING-21 output (ric, ex_date, amount); cash dividends going ex in (date, expiry]
    are subtracted from S at present value (OPT-02)
    adds: S (unadjusted close), T (years), r, pv_div, iv (OPT-08), delta (OPT-09, negative)
    """
    p = panel.copy()
    c = closes.rename(columns={"close": "S"})[["ric", "date", "S"]]
    p = p.merge(c, on=["ric", "date"], how="left")
    rates = rates.sort_index().ffill()
    p["r"] = rates.reindex(p["date"], method="ffill").to_numpy()
    p["T"] = (p["expiry"] - p["date"]).dt.days.clip(lower=0) / 365
    p["pv_div"] = 0.0
    div = dividends[["ric", "ex_date", "amount"]].dropna()
    for ric, idx in p.groupby("ric").groups.items():
        d = div[div["ric"] == ric]
        if d.empty:
            continue
        rows = p.loc[idx]
        start = rows["date"].to_numpy()[:, None]
        end = rows["expiry"].to_numpy()[:, None]
        ex = d["ex_date"].to_numpy()[None, :]
        inside = (ex > start) & (ex <= end)
        yrs = (ex - start).astype("timedelta64[D]").astype(float) / 365
        disc = np.exp(-rows["r"].fillna(0).to_numpy()[:, None] * yrs)
        p.loc[idx, "pv_div"] = (inside * disc * d["amount"].to_numpy()[None, :]).sum(axis=1)
    s_net = p["S"] - p["pv_div"]
    p["iv"] = implied_vol(p["close"].to_numpy(), s_net.to_numpy(), p["strike"].to_numpy(),
                          p["T"].to_numpy(), p["r"].to_numpy(), kind="put")
    p["delta"] = bs_delta_vec(s_net, p["strike"], p["T"], p["r"], p["iv"], kind="put")
    return p
