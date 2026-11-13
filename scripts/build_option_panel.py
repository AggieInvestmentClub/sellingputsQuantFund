"""
Build the per-event option panel the backtest reads, from the OPRA day-aggregate flat files.
No LSEG or Massive requests: everything comes from files already on disk.

Needs:
  data/raw/options/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz   (copied from darkpool)
  data/results/event_study/events.csv                       (scripts/overnight.py)
  data/raw/universe/massive_tickers.parquet                 (scripts/build_security_master.py)
  data/raw/prices/daily/<RIC>_unadj.parquet, data/raw/rates/treasury_yields.parquet,
  data/raw/corporate_actions/dividends.parquet              (scripts/overnight.py)

Run from the repo root:  py -3.12 -u scripts/build_option_panel.py
    random-entry control:  ... --events data/results/event_study/random_events.csv --out data/processed/options_random
Writes data/processed/options/: event_windows.parquet, event_puts.parquet, root_volume.parquet,
coverage.txt. Resumable: each finished month is cached in data/processed/options/_months/.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.events.option_panel import add_iv_delta, event_windows, root_volume, select_event_puts
from src.options.chains import parse_occ

ap = argparse.ArgumentParser()
ap.add_argument("--events", default="data/results/event_study/events.csv")
ap.add_argument("--out", default="data/processed/options")
args = ap.parse_args()

FLAT = Path("data/raw/options/day_aggs_v1")
RAW = Path("data/raw")
OUT = Path(args.out)
MONTHS = OUT / "_months"
MONTHS.mkdir(parents=True, exist_ok=True)


def fname(ric):
    return ric.replace(".", "_").replace("^", "~")      # same as scripts/overnight.py


files = sorted(FLAT.glob("*/*/*.csv.gz"))
if not files:
    raise SystemExit(f"no flat files under {FLAT}")
day_of = {f: pd.Timestamp(f.name[:10]) for f in files}
first, last = min(day_of.values()), max(day_of.values())
print(f"flat files: {len(files)} days, {first.date()} .. {last.date()}")

events = pd.read_csv(args.events, parse_dates=["date"])
tickers = pd.read_parquet(RAW / "universe" / "massive_tickers.parquet")
windows = event_windows(events, tickers, first, last)
windows.to_parquet(OUT / "event_windows.parquet", index=False)
roots = set(windows["root"].dropna())
print(f"events in the options period: {len(windows)} ({windows['complete'].sum()} with the full window), "
      f"{len(roots)} option roots")

# ---- one pass over the daily files, cached by month ----
t0 = time.time()
by_month = {}
for f in files:
    by_month.setdefault(f.parent.parent.name + "-" + f.parent.name, []).append(f)
for i, (month, month_files) in enumerate(sorted(by_month.items()), 1):
    puts_path, vol_path = MONTHS / f"puts_{month}.parquet", MONTHS / f"vol_{month}.parquet"
    if puts_path.exists() and vol_path.exists():
        continue
    puts, vols = [], []
    for f in month_files:
        day = pd.read_csv(f, usecols=["ticker", "volume", "open", "close", "high", "low", "transactions"])
        day = day[day["ticker"].str.slice(2).str.extract(r"^([A-Z0-9.]+?)\d{6}[CP]", expand=False).isin(roots)]
        day = pd.concat([day.reset_index(drop=True), parse_occ(day["ticker"]).reset_index(drop=True)], axis=1)
        puts.append(select_event_puts(day, day_of[f], windows))
        vols.append(root_volume(day, day_of[f], roots))
    pd.concat(puts, ignore_index=True).to_parquet(puts_path, index=False)
    pd.concat(vols, ignore_index=True).to_parquet(vol_path, index=False)
    print(f"  {month} done ({i}/{len(by_month)}, {time.time() - t0:.0f}s)")

panel = pd.concat([pd.read_parquet(p) for p in sorted(MONTHS.glob("puts_*.parquet"))], ignore_index=True)
vols = pd.concat([pd.read_parquet(p) for p in sorted(MONTHS.glob("vol_*.parquet"))], ignore_index=True)
vols.to_parquet(OUT / "root_volume.parquet", index=False)
print(f"put rows: {len(panel):,} for {panel['event_id'].nunique()} events")

# ---- implied vol and delta from each put's close ----
closes = []
for ric in panel["ric"].unique():
    p = RAW / "prices" / "daily" / f"{fname(ric)}_unadj.parquet"
    if p.exists():
        px = pd.read_parquet(p)
        if "TRDPRC_1" in px:
            c = px["TRDPRC_1"].astype(float).rename("close").rename_axis("date").reset_index()
            c["date"] = pd.to_datetime(c["date"]).dt.normalize()
            closes.append(c.assign(ric=ric))
closes = pd.concat(closes, ignore_index=True) if closes else pd.DataFrame(columns=["ric", "date", "close"])
rates = pd.read_parquet(RAW / "rates" / "treasury_yields.parquet")["US1MT=RR"].dropna()
dividends = pd.read_parquet(RAW / "corporate_actions" / "dividends.parquet")
panel = add_iv_delta(panel, closes, rates, dividends)
panel.to_parquet(OUT / "event_puts.parquet", index=False)

# ---- coverage report ----
first_days = panel.groupby("event_id")["date"].min()
ev = windows.set_index("event_id")
day1 = panel.merge(ev[["date"]].rename(columns={"date": "shock"}), left_on="event_id", right_index=True)
day1 = day1[day1["date"] > day1["shock"]]
day1 = day1[day1["date"] == day1.groupby("event_id")["date"].transform("min")]
dte = (day1["expiry"] - day1["date"]).dt.days
tradable = day1[(dte >= 14) & (dte <= 35) & day1["delta"].between(-0.30, -0.10) & (day1["volume"] > 0)]
ok = set(tradable["event_id"])
w = windows.assign(year=windows["date"].dt.year, has_puts=windows["event_id"].isin(set(panel["event_id"])),
                   tradable_day1=windows["event_id"].isin(ok))
L = [f"Option panel, built {pd.Timestamp.today():%Y-%m-%d} from {len(files)} flat files ({first.date()}..{last.date()})",
     "",
     f"events in the options period:        {len(w)}",
     f"  full window inside the data:        {w['complete'].sum()}",
     f"  root not found (no Massive ticker): {w['root'].isna().sum()}",
     f"  with any put rows:                  {w['has_puts'].sum()}",
     f"  with a traded put on day +1, 14-35 DTE, delta -0.10..-0.30: {w['tradable_day1'].sum()}",
     "",
     "by year:",
     w.groupby("year")[["complete", "has_puts", "tradable_day1"]].sum().assign(events=w.groupby("year").size())
      .to_string(),
     "",
     f"put rows: {len(panel):,}  | rows with an implied vol: {panel['iv'].notna().mean():.0%}",
     f"implied vol percentiles (decimal): {panel['iv'].quantile([.05, .25, .5, .75, .95]).round(3).tolist()}",
     f"events with no stock close for the IV: {panel.loc[panel['S'].isna(), 'event_id'].nunique()}",
     "",
     "events with puts but none tradable on day +1 (first 20):",
     w.loc[w["has_puts"] & ~w["tradable_day1"], ["event_id", "root"]].head(20).to_string(index=False)]
text = "\n".join(L)
(OUT / "coverage.txt").write_text(text, encoding="utf-8")
print("\n" + text)
print(f"\nsaved {OUT}/event_puts.parquet, root_volume.parquet, event_windows.parquet, coverage.txt "
      f"({time.time() - t0:.0f}s)")
