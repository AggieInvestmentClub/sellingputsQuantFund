"""
Live scan (AG-11): find today's shocks with the event-study rule, pull their Reuters news, and run the
agent on each. Run AFTER the US close (16:00 New York = 15:00 Matamoros/CST) with LSEG Workspace open.

    python -u scripts/live_scan.py                          # today
    python -u scripts/live_scan.py --days 3                 # today and the 2 trading days before
                                                            # (entry is allowed up to day +3)
    python -u scripts/live_scan.py --date 2026-11-18 --skip-update   # re-run without downloading prices

Steps (each one is the repo's own code):
  1. prices: append the missing days to data/raw/prices/daily/<RIC>_adj.parquet for the current
     S&P 500 (universe/sp500_current.csv) and SPY (ING-19 fields via lseg_bulk.fetch_series)
  2. shocks: abnormal = ret - beta * SPY ret (EVT-01 beta) <= -5%, SPY > -2%, not within 3 weekdays
     of earnings (EVT-03), market cap >= $5B (ING-20, appended to data/raw/market_cap/)
  3. news: Reuters headlines (ING-23) + 2 story texts (ING-24) -> news/reuters_events, reuters_stories
  4. agent: scripts/run_agent.Agent.screen on each shock -> data/results/agent/live_<date>.csv

The agent only says WHICH shocks to trade. Strikes / size / exits stay with the backtested rules
(short delta 0.15 below the post-news low, enter by day +3, no stop, 2% of capital at risk) and a
person on the team approves every order in the broker.
"""
import argparse
import sys
import time
import warnings

# lseg.data.get_history raises a NumPy DeprecationWarning per call (inside the library, harmless); hide it
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from scripts.run_agent import Agent
from src.agent.evidence import fname, load_close, market_cap_before
from src.data.news import before_decision, first_versions, on_shock_day
from src.events.build_events import near_earnings, rolling_beta

RAW = Path("data/raw")
PX = RAW / "prices" / "daily"
STORY_COLS = ["ric", "shock_date", "storyId", "event_time", "headline", "text"]
FIELDS = ["TRDPRC_1", "HIGH_1", "LOW_1", "OPEN_PRC", "ACVOL_UNS"]


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def update_prices(rics, end):
    """append missing days (ING-19 adjusted fields) to each RIC's daily file; new RICs get 2 years"""
    from src.ingest.lseg_bulk import fetch_series
    PX.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(end).normalize()
    failed, t0 = [], time.time()
    for i, ric in enumerate(rics, 1):
        p = PX / f"{fname(ric)}_adj.parquet"
        try:
            old = pd.read_parquet(p) if p.exists() else None
        except Exception as e:          # half-written / OneDrive-locked file: set it aside, download fresh
            log(f"  {ric}: unreadable price file ({type(e).__name__}), moved to {p.name}.bad and re-downloading")
            p.replace(p.with_name(p.name + ".bad"))
            old = None
        if old is not None and "date" in old.columns:
            old = old.set_index("date")
        if old is not None:
            old.index = pd.to_datetime(old.index).normalize()
        last = old.index.max() if old is not None and len(old) else end - pd.Timedelta(days=730)
        if last >= end:
            continue
        try:
            new = fetch_series(ric, (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                               end.strftime("%Y-%m-%d"), FIELDS, True)
        except Exception as e:
            failed.append((ric, type(e).__name__))
            continue
        if new.empty:
            continue
        new = new.drop(columns="ric", errors="ignore").set_index("date")
        out = pd.concat([old, new]) if old is not None else new
        out = out[~out.index.duplicated(keep="last")].sort_index()
        out.index.name = "date"
        tmp = p.with_name(p.name + ".tmp")     # write then swap, so an interrupted run can't corrupt the file
        out.to_parquet(tmp)
        tmp.replace(p)
        if i % 50 == 0:
            log(f"  prices {i}/{len(rics)} ({time.time() - t0:.0f}s)")
    if failed:
        log(f"  price update failed for {len(failed)} RICs, e.g. {failed[:5]}")


def update_market_cap(ric, end):
    """append missing days of TR.CompanyMarketCap (ING-20) to data/raw/market_cap/<RIC>.parquet"""
    from src.ingest.lseg_reference import fetch_market_cap
    p = RAW / "market_cap" / f"{fname(ric)}.parquet"
    old = pd.read_parquet(p) if p.exists() else None
    start = (old.index.max() + pd.Timedelta(days=1)) if old is not None and len(old) else pd.Timestamp(end) - pd.Timedelta(days=30)
    s = fetch_market_cap(ric, start=start.strftime("%Y-%m-%d"), end=pd.Timestamp(end).strftime("%Y-%m-%d"))
    if len(s):
        out = pd.concat([old, s.to_frame()]) if old is not None else s.to_frame()
        out = out[~out.index.duplicated(keep="last")].sort_index()
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(p)


def find_today_shocks(rics, date, earnings, threshold=-0.05, max_spy_drop=-0.02):
    """EVT-04's rule on one day for every RIC (membership = today's index list)"""
    date = pd.Timestamp(date).normalize()
    spy = load_close(RAW, "SPY")
    if spy is None or date not in spy.index:
        raise SystemExit(f"SPY has no bar for {date.date()}: run without --skip-update, after the close, on a trading day")
    m = spy["close"].pct_change()
    if m[date] <= max_spy_drop:
        log(f"SPY fell {m[date]:.1%} on {date.date()}: market-wide day, no shocks by rule (STR-2.1)")
        return pd.DataFrame()
    rows = []
    for ric in rics:
        px = load_close(RAW, ric)
        if px is None or date not in px.index:
            continue
        r = px["close"].pct_change()
        beta = rolling_beta(r, m).get(date)
        if beta is None or not np.isfinite(beta):
            continue
        abn = r[date] - beta * m[date]
        if abn > threshold or px["close"].loc[date] < 20:     # spec 1.2: no stocks under $20
            continue
        ed = earnings.loc[earnings["ric"] == ric, "report_date"]
        if len(ed) and near_earnings(pd.DatetimeIndex([date]), ed).iloc[0]:
            continue
        rows.append({"ric": ric, "date": date, "ret": r[date], "spy_ret": m[date], "beta": beta, "abnormal": abn})
    return pd.DataFrame(rows)


def pull_news(ric, date, n_stories=2):
    """ING-23 headlines + ING-24 texts for one shock, saved like scripts/download_event_news.py"""
    from src.ingest.lseg_news import fetch_event_headlines, fetch_story_text
    tag = f"{fname(ric)}_{pd.Timestamp(date):%Y%m%d}.parquet"
    hd, sd = RAW / "news" / "reuters_events", RAW / "news" / "reuters_stories"
    hd.mkdir(parents=True, exist_ok=True)
    sd.mkdir(parents=True, exist_ok=True)
    heads = fetch_event_headlines(ric, date, days_after=0)
    heads.to_parquet(hd / tag, index=False)
    st = before_decision(first_versions(heads), date) if len(heads) else heads
    day = on_shock_day(st, date) if len(st) else st
    pick = day.head(n_stories) if len(day) else st.tail(n_stories)
    rows = [{"ric": ric, "shock_date": pd.Timestamp(date), "storyId": s.storyId, "event_time": s.event_time,
             "headline": s.headline, "text": fetch_story_text(s.storyId)} for s in pick.itertuples()]
    pd.DataFrame(rows, columns=STORY_COLS).to_parquet(sd / tag, index=False)
    return len(heads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="last day to scan (default: today)")
    ap.add_argument("--days", type=int, default=1, help="trading days to scan back from --date (entry allowed to day +3)")
    ap.add_argument("--skip-update", action="store_true", help="don't download prices (use what's on disk)")
    ap.add_argument("--provider", choices=["openrouter"])
    ap.add_argument("--model")
    ap.add_argument("--mcap-floor", type=float, default=5e9)
    args = ap.parse_args()

    end = pd.Timestamp(args.date or pd.Timestamp.today()).normalize()
    now_ny = pd.Timestamp.now(tz="America/New_York")
    if end.date() == now_ny.date() and now_ny.hour < 16:
        log("WARNING: the US market is still open: today's bar is not final. Run after 16:00 New York (15:00 CST).")

    rics = pd.read_csv(RAW / "universe" / "sp500_current.csv")["Instrument"].dropna().tolist()
    earnings = pd.read_parquet(RAW / "earnings" / "earnings_dates.parquet")
    agent = Agent("data", args.provider, args.model)

    from src.ingest.lseg_client import lseg_session
    with lseg_session():
        if not args.skip_update:
            log(f"updating prices for {len(rics)} stocks + SPY through {end.date()} ...")
            update_prices(["SPY", ".VIX"] + rics, end)   # .VIX feeds the index plan (trade_desk index)
        days = load_close(RAW, "SPY").loc[:end].index[-args.days:]
        shocks = pd.concat([find_today_shocks(rics, d, earnings) for d in days], ignore_index=True)
        log(f"{len(shocks)} shocks before the market-cap floor on {[str(d.date()) for d in days]}")
        rows = []
        for s in shocks.itertuples():
            try:
                update_market_cap(s.ric, s.date)
            except Exception as e:
                log(f"  {s.ric}: market cap update failed ({type(e).__name__}), using last on disk")
            cap = market_cap_before(RAW, s.ric, s.date)
            if not (np.isfinite(cap) and cap >= args.mcap_floor):
                log(f"  {s.ric} {s.date.date()}: market cap {cap / 1e9 if np.isfinite(cap) else float('nan'):.1f}B < floor, skipped")
                continue
            n = pull_news(s.ric, s.date)
            r = agent.screen(s.ric, s.date, s.abnormal)
            px = pd.read_parquet(PX / f"{fname(s.ric)}_adj.parquet")
            px.index = pd.to_datetime(px.index).normalize()
            r.update({"headlines": n, "post_news_low": float(px.loc[s.date, "LOW_1"]),
                      "close": float(px.loc[s.date, "TRDPRC_1"]),
                      "days_since_shock": int(len(load_close(RAW, "SPY").loc[s.date:end]) - 1)})
            rows.append(r)
            log(f"  {s.ric} {s.date.date()} abn {s.abnormal:.1%} -> {'TAKE' if r.get('agent_take') else 'skip'}"
                f" {('(' + r['why_not'] + ')') if r.get('why_not') else ''}")

    out = pd.DataFrame(rows)
    path = agent.out / f"live_{end:%Y%m%d}.csv"
    agent.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    if out.empty:
        log(f"no shocks passed the rules. saved {path}")
        return
    cols = ["ric", "date", "days_since_shock", "abnormal", "close", "post_news_low", "agent_take",
            "confidence", "category", "why_not", "reason", "key_risk"]
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print(out[[c for c in cols if c in out.columns]].to_string(index=False))
    take = out[out.get("agent_take", False) == True]
    log(f"saved {path}: {len(out)} shocks, {len(take)} TAKE. Next: for each TAKE, check the option chain in the "
        f"broker (short put ~0.15 delta below post_news_low, ~3-5 weeks, enter by day +3), size at 2% of capital, "
        f"and have a teammate approve the order.")


if __name__ == "__main__":
    main()
