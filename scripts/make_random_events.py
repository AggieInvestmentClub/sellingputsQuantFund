"""
Random-entry control: for each real shock in the options period, random "event" dates in the same
stock with no shock nearby. Run through the same backtest, they show what the spread rules earn
without a shock: if the shock trades do no better, the edge is selling put premium, not the shock.

Run from the repo root:  python scripts/make_random_events.py [--per-event 2] [--seed 42]
Writes data/results/event_study/random_events.csv (same columns the backtest reads: ric, date,
abnormal = that day's stock return), then:
    python -u scripts/build_option_panel.py --events data/results/event_study/random_events.csv \
        --out data/processed/options_random
    python -u scripts/run_backtest.py --events data/results/event_study/random_events.csv \
        --panel data/processed/options_random/event_puts.parquet --config strategy/backtest_random.yaml ...
"""
import argparse
import sys

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.backtest.market import load_stock

ap = argparse.ArgumentParser()
ap.add_argument("--per-event", type=int, default=2)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--start", default="2024-04-02")
ap.add_argument("--end", default="2026-09-30", help="last random date (leaves room for the option window)")
ap.add_argument("--gap", type=int, default=20, help="trading days to keep away from any shock in the stock")
ap.add_argument("--out", default="data/results/event_study/random_events.csv")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
ev = pd.read_csv("data/results/event_study/events.csv", parse_dates=["date"])
real = ev[(ev.date >= args.start) & (ev.date <= "2026-12-31")]
rows = []
for ric, g in real.groupby("ric"):
    st = load_stock(ric)
    if st is None:
        continue
    days = st.index[(st.index >= args.start) & (st.index <= args.end)]
    pos = np.arange(len(st.index))
    near = np.zeros(len(st.index), bool)
    for d in ev.loc[ev.ric == ric, "date"]:          # every shock in this stock, any year
        k = st.index.searchsorted(d)
        near[max(0, k - args.gap):k + args.gap + 1] = True
    ok = [d for d, p in zip(st.index, pos) if not near[p] and d in days]
    n = min(len(ok), args.per_event * len(g))
    if n == 0:
        continue
    for d in sorted(rng.choice(ok, size=n, replace=False)):
        d = pd.Timestamp(d)
        rows.append({"ric": ric, "date": d.strftime("%Y-%m-%d"), "abnormal": float(st.at[d, "ret"])})
out = pd.DataFrame(rows).sort_values(["date", "ric"])
out.to_csv(args.out, index=False)
print(f"{len(out)} random dates in {out.ric.nunique()} stocks (real shocks: {len(real)}), "
      f"{out.date.min()} .. {out.date.max()} -> {args.out}")
