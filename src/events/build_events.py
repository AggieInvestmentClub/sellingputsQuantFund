"""
Event study (testing order step 1): find non-earnings shocks for every S&P 500 stock and measure
abnormal returns afterwards. Price-only; news categories come later.

Same rules as STR-2.1 (estimate_beta / abnormal_return / is_shock) and STR-2.2 (+/- 3 day
earnings buffer), vectorized over all days so the whole universe runs in minutes.
"""
import numpy as np
import pandas as pd

HORIZONS = (1, 5, 10, 20)


def rolling_beta(stock_ret, spy_ret, window=250, gap=5, min_obs=200):
    """
    [EVT-01] beta for every day, using only days before it: the `window` days ending `gap` days
    before that day (same as STR-2.1a estimate_beta, for all days at once)

    stock_ret, spy_ret: daily returns, date index. Returns a Series aligned to stock_ret's dates.
    """
    df = pd.concat([stock_ret, spy_ret], axis=1, keys=["s", "m"]).dropna()
    past = df.shift(gap + 1)                    # day t sees returns up to t-(gap+1)
    cov = past["s"].rolling(window, min_periods=min_obs).cov(past["m"])
    var = past["m"].rolling(window, min_periods=min_obs).var()
    return (cov / var).reindex(stock_ret.index)


def member_mask(dates, ric, membership):
    """
    [EVT-02] True on each date the RIC was an S&P 500 member, using the latest month-end snapshot
    strictly before the date (same rule as ING-13 members_before)
    """
    months = set(membership.loc[membership["ric"] == ric, "date"].dt.to_period("M"))
    prev_month = pd.DatetimeIndex(dates).to_period("M") - 1
    return pd.Series([m in months for m in prev_month], index=dates)


def near_earnings(dates, earnings_dates, buffer_days=3):
    """[EVT-03] True for dates within +/- buffer_days weekdays of any earnings date (STR-2.2 shock buffer)"""
    e = np.array(sorted(pd.to_datetime(list(earnings_dates)).normalize().values.astype("datetime64[D]")))
    out = []
    for d in pd.DatetimeIndex(dates).values.astype("datetime64[D]"):
        if e.size == 0:
            out.append(False)
            continue
        i = np.searchsorted(e, d)
        near = [abs(np.busday_count(min(d, x), max(d, x))) for x in e[max(i - 1, 0):i + 1]]
        out.append(min(near) <= buffer_days)
    return pd.Series(out, index=dates)


def find_shocks(ric, close, spy_ret, membership, earnings_dates, threshold=-0.05, max_spy_drop=-0.02):
    """
    [EVT-04] every shock day for one stock, with abnormal returns over the following days

    close: split-adjusted daily close (date index), spy_ret: SPY daily returns
    a shock = member that day, abnormal return <= threshold, SPY > max_spy_drop, not near earnings
    returns one row per shock: ric, date, ret, spy_ret, beta, abnormal, car_1/5/10/20, ret_1/5/10/20,
    earnings_known. car_h = sum of abnormal returns on days +1..+h using the pre-shock beta
    (NaN if the stock has fewer than h days of data after the shock, e.g. delisted)
    """
    ret = close.pct_change()
    spy = spy_ret.reindex(ret.index)
    beta = rolling_beta(ret, spy)
    abn = ret - beta * spy
    shock = (abn <= threshold) & (spy > max_spy_drop) & member_mask(ret.index, ric, membership)
    if earnings_dates is not None and len(earnings_dates):
        shock &= ~near_earnings(ret.index, earnings_dates)
    days = list(ret.index)
    rows = []
    for d in ret.index[shock.fillna(False)]:
        i = days.index(d)
        row = {"ric": ric, "date": d, "ret": ret[d], "spy_ret": spy[d], "beta": beta[d], "abnormal": abn[d],
               "earnings_known": bool(earnings_dates is not None and len(earnings_dates))}
        for h in HORIZONS:
            if i + h < len(days):
                fwd_r, fwd_m = ret.iloc[i + 1:i + h + 1], spy.iloc[i + 1:i + h + 1]
                row[f"car_{h}"] = float((fwd_r - beta[d] * fwd_m).sum())
                row[f"ret_{h}"] = float((1 + fwd_r).prod() - 1)
            else:
                row[f"car_{h}"] = row[f"ret_{h}"] = np.nan
        rows.append(row)
    cols = ["ric", "date", "ret", "spy_ret", "beta", "abnormal", "earnings_known"] + \
        [f"{k}_{h}" for h in HORIZONS for k in ("car", "ret")]
    return pd.DataFrame(rows, columns=cols)


def dedupe_events(events, master, cooldown=20):
    """
    [EVT-05] one event per company: if several RICs of one company shock on the same day (share
    classes like GOOG/GOOGL, or one stock listed under two RICs), keep the RIC with the longest
    membership (ties alphabetical); then drop shocks within `cooldown` trading days after the
    company's previous kept event, so overlapping windows aren't counted twice.
    Different days are kept even if the RIC differs: a company that moved exchange (BKR.N -> BKR.OQ)
    or relisted (CHK -> EXE) keeps its events from both periods.
    """
    if events.empty:
        return events
    span = master.assign(days=(master["last_member"] - master["first_member"]).dt.days)[["ric", "company_id", "days"]]
    ev = events.merge(span, on="ric", how="left")
    ev["company_id"] = ev["company_id"].fillna("RIC:" + ev["ric"])
    ev = (ev.sort_values(["company_id", "date", "days", "ric"], ascending=[True, True, False, True])
            .drop_duplicates(["company_id", "date"]).drop(columns="days"))
    kept = []
    for _, g in ev.sort_values("date").groupby("company_id"):
        last = None
        for idx, r in g.iterrows():
            if last is None or np.busday_count(last.date(), r["date"].date()) > cooldown:
                kept.append(idx)
                last = r["date"]
    return ev.loc[kept].sort_values(["date", "ric"]).reset_index(drop=True)


def summarize(events, thresholds=(-0.05, -0.10, -0.15)):
    """
    [EVT-06] average abnormal return after shocks, per shock size and horizon
    columns: threshold, horizon, n, mean, median, t_stat, pct_positive
    t_stat = mean / (std / sqrt(n)), treats events as independent (EVT-05 removes the worst overlap)
    """
    rows = []
    for t in thresholds:
        sub = events[events["abnormal"] <= t]
        for h in HORIZONS:
            x = sub[f"car_{h}"].dropna()
            n = len(x)
            rows.append({"threshold": t, "horizon": h, "n": n,
                         "mean": x.mean() if n else np.nan, "median": x.median() if n else np.nan,
                         "t_stat": x.mean() / (x.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                         "pct_positive": (x > 0).mean() if n else np.nan})
    return pd.DataFrame(rows)
