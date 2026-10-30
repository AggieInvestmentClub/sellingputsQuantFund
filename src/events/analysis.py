"""
Second look at the event study (EVT-04/05 output): returns from a realistic entry day, the worst
close while holding, and statistics that don't treat same-day shocks as independent.
"""
import numpy as np
import pandas as pd

ENTRY_LAGS = (1, 3)
HOLD_DAYS = (5, 10, 20)


def entry_returns(events, close, spy_ret, lags=ENTRY_LAGS, holds=HOLD_DAYS):
    """
    [EVT-07] returns after entering `lag` trading days after the shock, for one stock's events

    events: rows for one RIC with date and beta (EVT-04 / EVT-05 output)
    close: split-adjusted daily close (date index), spy_ret: SPY daily returns
    adds, per lag L and hold H (entry at the close of day +L, exit at the close of day +L+H):
      car_eL_H  abnormal return (pre-shock beta), sum of days +L+1 .. +L+H
      ret_eL_H  raw return, compounded
      low_eL_H  worst close during the hold vs the entry close (a put spread loses on this)
    NaN when the stock has no data that far (e.g. delisted)
    """
    ret = close.pct_change()
    spy = spy_ret.reindex(close.index)
    days = close.index
    out = events.copy()
    for lag in lags:
        for h in holds:
            for k in ("car", "ret", "low"):
                out[f"{k}_e{lag}_{h}"] = np.nan
    for idx, ev in events.iterrows():
        pos = days.searchsorted(pd.Timestamp(ev["date"]))
        if pos >= len(days) or days[pos] != pd.Timestamp(ev["date"]):
            continue
        for lag in lags:
            entry = pos + lag
            for h in holds:
                if entry + h >= len(days):
                    continue
                r = ret.iloc[entry + 1:entry + h + 1]
                m = spy.iloc[entry + 1:entry + h + 1]
                path = close.iloc[entry + 1:entry + h + 1]
                out.at[idx, f"car_e{lag}_{h}"] = float((r - ev["beta"] * m).sum())
                out.at[idx, f"ret_e{lag}_{h}"] = float(close.iloc[entry + h] / close.iloc[entry] - 1)
                out.at[idx, f"low_e{lag}_{h}"] = float(path.min() / close.iloc[entry] - 1)
    return out


def robust_summary(events, cols, tail=-0.10):
    """
    [EVT-08] mean / median / t-stat per column, plus a t-stat by date and the share of bad outcomes

    t_by_date: averages events on the same day first, then a t-stat across days, so a crash day
    with 40 shocks counts once (EVT-06's t_stat counts it 40 times)
    pct_tail: share of events with the value <= tail (-0.10 = lost 10% or more)
    returns a DataFrame: column, n, n_dates, mean, median, t_stat, t_by_date, pct_positive, pct_tail
    """
    rows = []
    for c in cols:
        sub = events[["date", c]].dropna()
        x = sub[c]
        n = len(x)
        daily = sub.groupby("date")[c].mean()
        nd = len(daily)
        rows.append({
            "column": c, "n": n, "n_dates": nd,
            "mean": x.mean() if n else np.nan,
            "median": x.median() if n else np.nan,
            "t_stat": x.mean() / (x.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
            "t_by_date": daily.mean() / (daily.std(ddof=1) / np.sqrt(nd)) if nd > 1 else np.nan,
            "pct_positive": (x > 0).mean() if n else np.nan,
            "pct_tail": (x <= tail).mean() if n else np.nan,
        })
    return pd.DataFrame(rows)
