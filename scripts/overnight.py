"""
Overnight run: download prices, market cap, earnings, dividends and Treasury yields for every
stock ever in the S&P 500 since 2015, then run the event study (testing order step 1).

Needs: LSEG Workspace open + signed in, data/raw/universe/security_master.parquet
(run scripts/build_security_master.py first).

Run from the repo root:  python -u scripts/overnight.py [--budget 7000]
Safe to re-run: files already on disk are skipped, so it resumes where it stopped.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import lseg.data as ld
import pandas as pd

from src.events.build_events import dedupe_events, find_shocks, summarize
from src.ingest.lseg_client import lseg_session
from src.ingest.lseg_earnings import fetch_earnings_dates
from src.ingest.lseg_prices import fetch_daily_history
from src.ingest.lseg_reference import fetch_dividends, fetch_market_cap, fetch_treasury_yields

START = "2014-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")
RAW = Path("data/raw")
PRICES, CAPS = RAW / "prices" / "daily", RAW / "market_cap"
RESULTS = Path("data/results/event_study")

parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int, default=7000, help="max LSEG requests this run (limit is 10,000/day)")
args = parser.parse_args()


class Budget(BaseException):
    # BaseException so the ingest functions' "except Exception" (skip a failed batch) can't swallow it
    pass


calls = {"n": 0}


def counted(fn):
    def wrapper(*a, **k):
        if calls["n"] >= args.budget:
            raise Budget()
        calls["n"] += 1
        return fn(*a, **k)
    return wrapper


ld.get_history, ld.get_data = counted(ld.get_history), counted(ld.get_data)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} [{calls['n']:5d} req] {msg}", flush=True)


def fname(ric):
    return ric.replace(".", "_").replace("^", "~")


def retry(fn, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Budget:
            raise
        except Exception as e:
            if i == tries - 1:
                raise
            log(f"  retry after {type(e).__name__}")
            time.sleep(5 * 2 ** i)


master = pd.read_parquet(RAW / "universe" / "security_master.parquet")
membership = pd.read_parquet(RAW / "universe" / "sp500_membership.parquet")
rics = sorted(master["ric"])
for p in (PRICES, CAPS, RAW / "earnings", RAW / "corporate_actions", RAW / "rates", RESULTS):
    p.mkdir(parents=True, exist_ok=True)

# ---------------- downloads ----------------
failed = []
with lseg_session():
    try:
        log(f"start: {len(rics)} RICs, budget {args.budget} requests")
        if not (PRICES / "SPY_adj.parquet").exists():
            retry(lambda: fetch_daily_history("SPY", START, END)).to_parquet(PRICES / "SPY_adj.parquet")
            log("SPY done")
        if not (RAW / "rates" / "treasury_yields.parquet").exists():
            retry(lambda: fetch_treasury_yields(start=START)).to_parquet(RAW / "rates" / "treasury_yields.parquet")
            log("Treasury yields done")
        if not (RAW / "earnings" / "earnings_dates.parquet").exists():
            fetch_earnings_dates(rics, start=START).to_parquet(RAW / "earnings" / "earnings_dates.parquet")
            log("earnings dates done")
        if not (RAW / "corporate_actions" / "dividends.parquet").exists():
            fetch_dividends(rics, start=START).to_parquet(RAW / "corporate_actions" / "dividends.parquet")
            log("dividends done")

        for i, ric in enumerate(rics, 1):
            jobs = [(PRICES / f"{fname(ric)}_adj.parquet", lambda r=ric: fetch_daily_history(r, START, END)),
                    (PRICES / f"{fname(ric)}_unadj.parquet", lambda r=ric: fetch_daily_history(r, START, END, "unadjusted")),
                    (CAPS / f"{fname(ric)}.parquet", lambda r=ric: fetch_market_cap(r, START, END).to_frame())]
            for path, job in jobs:
                if path.exists():
                    continue
                try:
                    retry(job).to_parquet(path)       # empty results are saved too, so they aren't re-requested
                except Budget:
                    raise
                except Exception as e:
                    failed.append((ric, path.name, f"{type(e).__name__}: {str(e)[:80]}"))
            if i % 50 == 0:
                log(f"stocks {i}/{len(rics)}")
        log("all downloads done")
    except Budget:
        log(f"STOPPED: request budget of {args.budget} reached. Re-run tomorrow to continue.")

if failed:
    pd.DataFrame(failed, columns=["ric", "file", "error"]).to_csv(RESULTS / "download_failures.csv", index=False)
    log(f"{len(failed)} downloads failed, see {RESULTS / 'download_failures.csv'}")

# ---------------- event study ----------------
for needed in (PRICES / "SPY_adj.parquet", RAW / "earnings" / "earnings_dates.parquet"):
    if not needed.exists():
        log(f"event study SKIPPED: {needed} not downloaded yet (needed for returns / earnings filter). Re-run to finish.")
        sys.exit(0)
log("event study: finding shocks ...")
spy_close = pd.read_parquet(PRICES / "SPY_adj.parquet")["TRDPRC_1"].astype(float).dropna()
spy_ret = spy_close.pct_change()
earn = pd.read_parquet(RAW / "earnings" / "earnings_dates.parquet")
earn_by_ric = earn.groupby("ric")["report_date"].apply(list).to_dict()

all_events, missing = [], []
for i, ric in enumerate(rics, 1):
    path = PRICES / f"{fname(ric)}_adj.parquet"
    if not path.exists():
        missing.append(ric)
        continue
    px = pd.read_parquet(path)
    if px.empty or "TRDPRC_1" not in px:
        missing.append(ric)
        continue
    close = px["TRDPRC_1"].astype(float).dropna()
    ev = find_shocks(ric, close, spy_ret, membership, earn_by_ric.get(ric, []))
    if len(ev):
        all_events.append(ev)
    if i % 100 == 0:
        log(f"  scanned {i}/{len(rics)}")

events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
events.to_csv(RESULTS / "events_raw.csv", index=False)
clean = dedupe_events(events, master)
clean.to_csv(RESULTS / "events.csv", index=False)

lines = [f"Event study, run {END}",
         f"stocks scanned: {len(rics) - len(missing)} of {len(rics)} (no price data: {len(missing)})",
         f"raw shocks (abnormal <= -5%, SPY > -2%, member, not within 3 days of earnings): {len(events)}",
         f"after one share class per company + 20-day cooldown: {len(clean)}",
         f"  of which earnings dates unknown for the stock: {(~clean['earnings_known']).sum() if len(clean) else 0}",
         "", "Average abnormal return AFTER the shock day (car_h = days +1..+h), by shock size:"]
summary = summarize(clean)
lines.append(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
if len(clean):
    by_year = clean.assign(year=clean["date"].dt.year).groupby("year")["car_10"].agg(n="count", mean="mean", median="median")
    lines += ["", "10-day abnormal return after -5% shocks, by year:", by_year.to_string(float_format=lambda x: f"{x:.4f}")]
text = "\n".join(lines)
(RESULTS / "summary.txt").write_text(text, encoding="utf-8")
summary.to_csv(RESULTS / "summary.csv", index=False)
print("\n" + text, flush=True)
log("done")
