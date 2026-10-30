"""
Second look at the event study (after scripts/overnight.py): realistic entry days, the worst close
while holding, same-day clustering, and the share of bad outcomes. No LSEG needed.

Run from the repo root:  python scripts/analyze_events.py
Reads data/results/event_study/events.csv and data/raw/prices/daily/*_adj.parquet.
Writes data/results/event_study/events_entry.csv and analysis.txt.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import pandas as pd

from src.events.analysis import ENTRY_LAGS, HOLD_DAYS, entry_returns, robust_summary

PRICES = Path("data/raw/prices/daily")
RESULTS = Path("data/results/event_study")
CROWDED = 5          # a day with this many shocks or more counts as a market-wide day


def fname(ric):
    return ric.replace(".", "_").replace("^", "~")      # same as scripts/overnight.py


def close_of(ric):
    px = pd.read_parquet(PRICES / f"{fname(ric)}_adj.parquet")
    return px["TRDPRC_1"].astype(float).dropna()


events = pd.read_csv(RESULTS / "events.csv", parse_dates=["date"])
spy_ret = close_of("SPY").pct_change()
parts = [entry_returns(g, close_of(ric), spy_ret) for ric, g in events.groupby("ric")]
ev = pd.concat(parts).sort_values(["date", "ric"]).reset_index(drop=True)
ev.to_csv(RESULTS / "events_entry.csv", index=False)

fmt = {"float_format": lambda x: f"{x:.4f}", "index": False}
per_day = ev.groupby("date").size()
crowded_days = per_day[per_day >= CROWDED].index
crowded = ev["date"].isin(crowded_days)
L = [f"Event study, second look ({len(ev)} events, {ev['date'].nunique()} distinct days, "
     f"{ev['date'].min().date()} .. {ev['date'].max().date()})", ""]

L += ["1) Abnormal return by entry day. car_H = enter at the shock-day close (EVT-04);",
      "   car_eL_H = enter at the close of day +L, hold H days (EVT-07).",
      "   t_by_date counts each day once; compare it with t_stat."]
cols = [f"car_{h}" for h in HOLD_DAYS] + [f"car_e{lag}_{h}" for lag in ENTRY_LAGS for h in HOLD_DAYS]
L += [robust_summary(ev, cols).to_string(**fmt), ""]

L += ["2) Raw return and worst close while holding, entry day +1 (what a put spread sees).",
      "   pct_tail = share of events at -10% or worse."]
cols = [f"{k}_e1_{h}" for h in HOLD_DAYS for k in ("ret", "low")]
L += [robust_summary(ev, cols).to_string(**fmt)]
q = ev[[f"low_e1_{h}" for h in HOLD_DAYS]].quantile([0.01, 0.05, 0.10, 0.25, 0.50])
q.index = [f"{p:.0%} worst" if p < 0.5 else "median" for p in q.index]
L += ["", "   worst close during the hold, percentiles:", q.to_string(float_format=lambda x: f"{x:.4f}"), ""]

L += [f"3) Clustering: {len(crowded_days)} days with {CROWDED}+ shocks hold {crowded.sum()} events "
      f"({crowded.mean():.0%}). Busiest days:"]
L += [per_day.sort_values(ascending=False).head(10).rename("shocks").to_string(), ""]
L += ["   car_e1_10 with and without those days:"]
both = pd.concat([robust_summary(ev, ["car_e1_10"]).assign(sample="all"),
                  robust_summary(ev[~crowded], ["car_e1_10"]).assign(sample=f"days with <{CROWDED} shocks"),
                  robust_summary(ev[crowded], ["car_e1_10"]).assign(sample=f"days with {CROWDED}+ shocks")])
L += [both[["sample", "n", "n_dates", "mean", "median", "t_by_date", "pct_positive"]].to_string(**fmt), ""]

L += ["4) By shock size (abnormal return on the shock day), entry +1, 10-day hold:"]
rows = []
for lo, hi in ((-0.10, -0.05), (-0.15, -0.10), (-1.0, -0.15)):
    sub = ev[(ev["abnormal"] > lo) & (ev["abnormal"] <= hi)]
    rows.append(robust_summary(sub, ["car_e1_10", "low_e1_10"]).assign(shock=f"{lo:.0%} < abn <= {hi:.0%}"))
L += [pd.concat(rows)[["shock", "column", "n", "mean", "median", "t_by_date", "pct_tail"]].to_string(**fmt), ""]

if "earnings_known" in ev:
    known = ev["earnings_known"].astype(bool)
    L += [f"5) Earnings dates unknown for {(~known).sum()} events (their shocks may be earnings). "
          "car_e1_10 with earnings known only:"]
    L += [robust_summary(ev[known], ["car_e1_10"]).to_string(**fmt), ""]

text = "\n".join(L)
(RESULTS / "analysis.txt").write_text(text, encoding="utf-8")
print(text)
print(f"\nsaved {RESULTS / 'analysis.txt'} and {RESULTS / 'events_entry.csv'}")
