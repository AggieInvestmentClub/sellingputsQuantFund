"""
Data checks on downloaded tables (prices, membership). Pure functions: no downloads.
Run them all at once with scripts/validate_data.py after every data refresh.
"""
import numpy as np
import pandas as pd


def price_issues(px, close_col="TRDPRC_1", max_gap_days=7, max_abs_ret=0.5, stale_days=5):
    """
    [DAT-05] problems in one stock's daily price table (ING-19 output: date index, TRDPRC_1 close)

    returns a dict: rows, first, last, nan_close, nonpositive, duplicate_dates, unsorted,
    max_gap_days (weekdays missing between two rows), big_moves (|daily return| > max_abs_ret),
    big_move_dates (first 5), longest_stale (most days in a row with the same close),
    problems (text, "" if clean)
    """
    out = {"rows": len(px), "first": None, "last": None, "nan_close": 0, "nonpositive": 0,
           "duplicate_dates": 0, "unsorted": False, "max_gap_days": 0, "big_moves": 0,
           "big_move_dates": "", "longest_stale": 0, "problems": ""}
    if px is None or px.empty:
        out["rows"] = 0
        out["problems"] = "empty"
        return out
    if close_col not in px:
        out["problems"] = f"no {close_col} column"
        return out

    idx = pd.DatetimeIndex(px.index)
    close = pd.Series(pd.to_numeric(px[close_col], errors="coerce").to_numpy(), index=idx)
    out["first"], out["last"] = idx.min().date(), idx.max().date()
    out["nan_close"] = int(close.isna().sum())
    out["nonpositive"] = int((close <= 0).sum())
    out["duplicate_dates"] = int(idx.duplicated().sum())
    out["unsorted"] = not idx.is_monotonic_increasing

    c = close[~idx.duplicated(keep="last")].sort_index().dropna()
    c = c[c > 0]
    if len(c) > 1:
        days = c.index.values.astype("datetime64[D]")
        gaps = np.busday_count(days[:-1], days[1:]) - 1
        out["max_gap_days"] = int(gaps.max())
        ret = c.pct_change().dropna()
        big = ret[ret.abs() > max_abs_ret]
        out["big_moves"] = len(big)
        out["big_move_dates"] = ",".join(str(d.date()) for d in big.index[:5])
        run_id = (c != c.shift()).cumsum()
        out["longest_stale"] = int(run_id.value_counts().max())

    problems = []
    if out["nan_close"]:
        problems.append(f"{out['nan_close']} missing closes")
    if out["nonpositive"]:
        problems.append(f"{out['nonpositive']} closes <= 0")
    if out["duplicate_dates"]:
        problems.append(f"{out['duplicate_dates']} duplicate dates")
    if out["unsorted"]:
        problems.append("dates not sorted")
    if out["max_gap_days"] > max_gap_days:
        problems.append(f"gap of {out['max_gap_days']} weekdays")
    if out["big_moves"]:
        problems.append(f"{out['big_moves']} daily moves over {max_abs_ret:.0%}")
    if out["longest_stale"] >= stale_days:
        problems.append(f"same close {out['longest_stale']} days in a row")
    out["problems"] = "; ".join(problems)
    return out


def membership_issues(membership, changes=None, min_size=450, max_size=510):
    """
    [DAT-06] problems in the month-end S&P 500 membership table (ING-12), optionally checked
    against the join / leave events (ING-14)

    membership: date, ric. changes: ric, change ("Joiner" / "Leaver"), date
    returns a DataFrame (date, kind, ric, detail), one row per problem; empty if clean.
    kinds: blank_ric, duplicate_ric, missing_month, size, and with changes:
    removed_without_leaver, leaver_still_member, leaver_not_member
    (and added_without_joiner, joiner_not_member if the changes include joiners)
    """
    rows = []
    m = membership.copy()
    m["date"] = pd.to_datetime(m["date"])
    m["ric"] = m["ric"].astype("string")

    blank = m["ric"].isna() | (m["ric"].fillna("").str.strip() == "")
    rows += [(d, "blank_ric", "", "empty RIC in snapshot") for d in m.loc[blank, "date"]]
    m = m[~blank]
    dup = m.duplicated(["date", "ric"])
    rows += [(r.date, "duplicate_ric", r.ric, "listed twice") for r in m[dup].itertuples()]
    m = m[~dup]

    dates = sorted(m["date"].unique())
    if not dates:
        return pd.DataFrame(rows, columns=["date", "kind", "ric", "detail"])
    expected = pd.date_range(dates[0], dates[-1], freq="ME")
    rows += [(d, "missing_month", "", "no snapshot") for d in expected.difference(pd.DatetimeIndex(dates))]
    sizes = m.groupby("date").size()
    rows += [(d, "size", "", f"{n} members") for d, n in sizes.items() if n < min_size or n > max_size]

    if changes is not None and len(changes):
        ch = changes.copy()
        ch["date"] = pd.to_datetime(ch["date"])
        kind = ch["change"].astype(str).str.lower()
        sets = {d: set(g) for d, g in m.groupby("date")["ric"]}
        for name, events, side in (("leaver", ch[kind.str.startswith("leav")], "removed"),
                                   ("joiner", ch[kind.str.startswith("join")], "added")):
            if events.empty:
                continue
            # 1) every RIC that drops out (or comes in) between two snapshots should have an event
            for a, b in zip(dates, dates[1:]):
                moved = sets[a] - sets[b] if side == "removed" else sets[b] - sets[a]
                window = set(events.loc[(events["date"] > a) & (events["date"] <= b), "ric"])
                rows += [(b, f"{side}_without_{name}", r, f"between {a.date()} and {b.date()}")
                         for r in sorted(moved - window)]
            # 2) every event inside the snapshot range should match the snapshots around it
            inside = events[(events["date"] > dates[0]) & (events["date"] < dates[-1])]
            for e in inside.itertuples():
                before = max(d for d in dates if d < e.date)
                after = min(d for d in dates if d > e.date)
                was, now = e.ric in sets[before], e.ric in sets[after]
                if name == "leaver" and now:
                    rows.append((e.date, "leaver_still_member", e.ric, f"still listed on {after.date()}"))
                elif name == "leaver" and not was:
                    rows.append((e.date, "leaver_not_member", e.ric, f"not listed on {before.date()}"))
                elif name == "joiner" and was:
                    rows.append((e.date, "joiner_already_member", e.ric, f"already listed on {before.date()}"))
                elif name == "joiner" and not now:
                    rows.append((e.date, "joiner_not_member", e.ric, f"not listed on {after.date()}"))

    out = pd.DataFrame(rows, columns=["date", "kind", "ric", "detail"])
    return out.sort_values(["date", "kind", "ric"]).reset_index(drop=True)


def split_factors(adj_close, unadj_close, tol=0.005):
    """
    [DAT-07] price adjustments (splits, spin-offs) found by comparing adjusted and unadjusted closes

    LSEG adjusts for splits and capital changes such as spin-offs, not regular dividends, so
    unadjusted / adjusted is constant between those events.
    returns a DataFrame (date, ratio_before, ratio_after, split): one row per day the ratio
    changes by more than tol; split = ratio_before / ratio_after (4.0 = 4-for-1, 0.1 = 1-for-10,
    a non-round value like 1.40 is usually a spin-off: A.N 2014-11-03, Keysight)
    """
    both = pd.concat([pd.to_numeric(adj_close, errors="coerce"),
                      pd.to_numeric(unadj_close, errors="coerce")], axis=1, keys=["adj", "raw"])
    both = both[~both.index.duplicated(keep="last")].sort_index().dropna()
    both = both[(both["adj"] > 0) & (both["raw"] > 0)]
    ratio = both["raw"] / both["adj"]
    prev = ratio.shift()
    jump = (ratio / prev - 1).abs() > tol
    out = pd.DataFrame({"date": ratio.index[jump], "ratio_before": prev[jump].to_numpy(),
                        "ratio_after": ratio[jump].to_numpy()})
    out["split"] = out["ratio_before"] / out["ratio_after"]
    return out.reset_index(drop=True)
