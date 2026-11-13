"""
Parameter combinations and event filters for the backtest.

Trade parameters change the simulated trade, so each combination is simulated once over all events.
Event filters only choose which events count, so they are applied to finished trades for free:
every (trade combination x event filter) pair can be scored without re-simulating.
"""
import itertools
import json

import numpy as np
import pandas as pd


def combo_id(params):
    return json.dumps(params, sort_keys=True, default=list)


def expand_grid(space, baseline, dims=None):
    """
    [BT-05] every combination of the chosen dimensions; all other parameters stay at baseline

    space: {name: [values]}, baseline: {name: value}, dims: names to cross (None = all of space)
    returns list of dicts (only the varied names; merge with baseline when simulating)
    ex: dims ["short_delta", "width_pct"] with 3 and 4 values -> 12 dicts
    """
    dims = list(space) if dims is None else list(dims)
    out = []
    for values in itertools.product(*(space[d] for d in dims)):
        out.append(dict(zip(dims, values)))
    return out


def one_at_a_time(space, baseline):
    """
    [BT-06] the baseline plus one variant per non-baseline value of each parameter (spec section 7.5)

    returns list of (label, params) with label "baseline" or "<name>=<value>"
    """
    out = [("baseline", {})]
    for name, values in space.items():
        for v in values:
            if v != baseline.get(name):
                out.append((f"{name}={v}", {name: v}))
    return out


EVENT_SETS = {
    "all": lambda e: pd.Series(True, index=e.index),
    "has_news": lambda e: e["n_headlines"] > 0,
    "nj_both": lambda e: e["verdict_gpt4o"].eq("not justified") & e["verdict_opus"].eq("not justified"),
    "nj_both_samecat": lambda e: (e["verdict_gpt4o"].eq("not justified") & e["verdict_opus"].eq("not justified")
                                  & e["category_gpt4o"].eq(e["category_opus"])),
    "nj_opus": lambda e: e["verdict_opus"].eq("not justified"),
    "nj_gpt4o": lambda e: e["verdict_gpt4o"].eq("not justified"),
    "not_j_both": lambda e: ~(e["verdict_gpt4o"].eq("justified") & e["verdict_opus"].eq("justified")),
    "j_both": lambda e: e["verdict_gpt4o"].eq("justified") & e["verdict_opus"].eq("justified"),
}


def event_mask(events, event_set="all", category="any", min_ratio=None, max_shock=-0.05, mcap_floor=5e9):
    """
    [BT-07] which events one event filter keeps

    events: event table from scripts/run_backtest.py (labels merged in, one row per event)
    event_set: key of EVENT_SETS; category: Opus category or "any"; min_ratio: market value lost /
    Opus mid impact (None = off); max_shock: abnormal return must be <= this; mcap_floor: $ before the shock
    returns boolean Series aligned with events
    """
    m = EVENT_SETS[event_set](events).fillna(False).astype(bool)
    if category != "any":
        m &= events["category_opus"].eq(category)
    if min_ratio is not None:
        m &= events["ratio_opus"].ge(min_ratio)
    m &= events["abnormal"].le(max_shock)
    m &= events["market_cap"].fillna(0).ge(mcap_floor)
    return m


def event_filters(space):
    """
    [BT-08] every event filter from a {name: [values]} space with keys event_set, category,
    min_ratio, max_shock, mcap_floor (missing keys use the BT-07 defaults)

    returns list of dicts
    """
    keys = [k for k in ["event_set", "category", "min_ratio", "max_shock", "mcap_floor"] if k in space]
    return [dict(zip(keys, v)) for v in itertools.product(*(space[k] for k in keys))]


def trade_metrics(trades):
    """
    [BT-09] per-trade results for one set of trades (spec section 8)

    trades: rows with ror (P&L / max loss), pnl, entry_date, days_held
    returns Series: n, n_dates, win_rate, mean_ror, median_ror, sum_ror, worst5_ror (5th percentile),
    max_loss_share (ror <= -0.9), t_by_date (t-stat of the mean ror per entry date: shocks on the same day
    count once), avg_days
    """
    t = trades.dropna(subset=["ror"])
    n = len(t)
    if n == 0:
        return pd.Series({"n": 0, "n_dates": 0, "win_rate": np.nan, "mean_ror": np.nan, "median_ror": np.nan,
                          "sum_ror": 0.0, "worst5_ror": np.nan, "max_loss_share": np.nan,
                          "t_by_date": np.nan, "avg_days": np.nan})
    by_date = t.groupby("entry_date")["ror"].mean()
    sd = by_date.std()
    tstat = by_date.mean() / (sd / np.sqrt(len(by_date))) if len(by_date) > 1 and sd > 0 else np.nan
    return pd.Series({"n": n, "n_dates": len(by_date), "win_rate": (t["pnl"] > 0).mean(),
                      "mean_ror": t["ror"].mean(), "median_ror": t["ror"].median(), "sum_ror": t["ror"].sum(),
                      "worst5_ror": t["ror"].quantile(0.05), "max_loss_share": (t["ror"] <= -0.9).mean(),
                      "t_by_date": tstat, "avg_days": t["days_held"].mean()})


def portfolio_curve(trades, sectors, capital=100_000, risk_pct=0.02, max_sector_pct=0.30):
    """
    [BT-10] trades run as one book: each trade risks risk_pct of starting capital (STR-3.8a, no
    compounding), one position per ticker and the sector cap (STR-5b), P&L booked on the exit date

    trades: ric, entry_date, exit_date, ror. sectors: {ric: GICS sector}
    returns (taken trades with dollar_pnl, daily equity Series, stats dict: total_return, max_drawdown,
    sharpe (daily, annualized), n_taken, n_blocked)
    """
    from strategy.strategy_logic import portfolio_ok

    t = trades.dropna(subset=["ror"]).sort_values("entry_date")
    risk = capital * risk_pct
    open_pos, taken, blocked = [], [], 0
    for r in t.itertuples():
        open_pos = [p for p in open_pos if p["exit"] > r.entry_date]
        ok, _ = portfolio_ok(open_pos, r.ric, sectors.get(r.ric, "unknown"), risk, max_sector_pct)
        if not ok:
            blocked += 1
            continue
        open_pos.append({"ticker": r.ric, "sector": sectors.get(r.ric, "unknown"), "max_loss": risk,
                         "exit": r.exit_date})
        taken.append({**r._asdict(), "dollar_pnl": r.ror * risk})
    taken = pd.DataFrame(taken)
    if taken.empty:
        return taken, pd.Series(dtype=float), {"total_return": 0.0, "max_drawdown": 0.0, "sharpe": np.nan,
                                               "n_taken": 0, "n_blocked": blocked}
    days = pd.bdate_range(taken["entry_date"].min(), taken["exit_date"].max()).union(
        pd.DatetimeIndex(taken["exit_date"].unique()))
    daily = taken.groupby("exit_date")["dollar_pnl"].sum().reindex(days, fill_value=0.0)
    equity = capital + daily.cumsum()
    dd = (equity / equity.cummax() - 1).min()
    rets = daily / capital
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else np.nan
    return taken, equity, {"total_return": equity.iloc[-1] / capital - 1, "max_drawdown": dd,
                           "sharpe": sharpe, "n_taken": len(taken), "n_blocked": blocked}
