"""
Daily signal for the index branch (AG-29): is today an entry day for the walk-forward pick?
Rule in force for 2026 (strategy/preregistered_index_branch.md, walk-forward pick from 2018 on):
    shock+guard | short ~0.10 delta / long ~0.03 delta | dynamic take-profit
Run after 15:00 CST, after scripts/live_scan.py has updated data/raw/prices/daily/SPY_adj.parquet:

    py -3.12 -u scripts/index_signal.py --capital 5000

--capital sets the spread width by the capital rule v2 (AG-28f, strategy/preregistered_capital_fix.md):
W = floor(capital x 10% / 100) dollars; the long leg is the NARROWER of $W below the short and the 0.03-delta put,
and the max loss must fit 10% of capital (v1 used the 0.03-delta leg unchecked at W >= 15: 16-22% risk in 2024-26). Capital test 2010-2026 (data/results/index_branch/capital_run.log):
$1,000 FAIL, $2,000 FAIL, $2,500 PASS (borderline), $5,000 / $10,000 / $25,000 PASS.

Entry = SPY fell >= 1.5% in a day (or >= 2.5% over 2 days), then on day +1..+3 closes ABOVE the post-shock low,
while SPY is above its 200-day average. Part of the guard needs the option chain: the ~9-day ATM put IV must not be
above the ~30-day ATM put IV (check it in the broker). Nothing here sends orders. PAPER-TRADE until the account can
carry one spread: these spreads are ~$10-25 wide on XSP, i.e. ~$1,000-2,500 of risk each.
"""
import math
import sys
import argparse
from pathlib import Path

import pandas as pd

P = Path("data/raw/prices/daily/SPY_adj.parquet")      # updated daily by scripts/live_scan.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, help="account size in $; sets the spread width (capital rule)")
    a = ap.parse_args()
    px = pd.read_parquet(P)
    px = px.set_index("date") if "date" in px.columns else px
    px.index = pd.to_datetime(px.index).normalize()
    c = px["TRDPRC_1"].astype(float).sort_index().dropna()
    r1, r2 = c.pct_change(), c.pct_change(2)
    ma200 = c.rolling(200).mean()
    today = c.index[-1]
    shocks = c.index[(r1 <= -0.015) | (r2 <= -0.025)]
    recent = [s for s in shocks if 1 <= len(c.loc[s:today]) - 1 <= 3]
    print(f"SPY {today.date()}: close {c.iloc[-1]:.2f}, 1-day {r1.iloc[-1]:+.2%}, 200-day avg {ma200.iloc[-1]:.2f}")
    if not recent:
        print("NO SIGNAL: no SPY shock in the last 3 trading days.")
        return
    s = recent[0]
    low = c.loc[s:today].iloc[:-1].min()
    trend = c.iloc[-1] > ma200.iloc[-1]
    held = c.iloc[-1] > low
    day = len(c.loc[s:today]) - 1
    print(f"shock on {s.date()} (1-day {r1.loc[s]:+.2%}); today is day +{day}; post-shock low {low:.2f}")
    if not trend:
        print("NO ENTRY: SPY is below its 200-day average (crash guard).")
    elif not held:
        print("WAIT: SPY made a new low today; selling not done." + ("" if day < 3 else " Last allowed day: drop it."))
    else:
        rv = c.pct_change().iloc[-20:].std() * math.sqrt(252)
        print(f"ENTRY SIGNAL (paper unless the account can carry it). Realized vol 20d {rv:.0%}.")
        print("  In the broker, XSP puts (European, cash-settled: no early assignment; do NOT use SPY options),")
        print("  the Friday expiry closest to 21 days (14-35):")
        print("  1. Guard: ATM put IV at ~9 days must NOT be above ATM put IV at ~30 days. If it is: no entry.")
        W = int(a.capital * 0.10 // 100) if a.capital else 15
        if W < 1 or (a.capital and a.capital < 2500):
            print(f"  ACCOUNT ${a.capital:,.0f}: FAILED the capital test (needs >= $2,500 at 10% risk). Paper-trade only.")
        Wd = max(W, 1)
        print(f"  2. Sell the XSP put closest to delta -0.10. Buy the put ${Wd} lower OR the put closest to delta -0.03,")
        print(f"     whichever strike is HIGHER (narrower spread), same expiry. Max loss must be <= ${Wd * 100 * 1.15:,.0f}"
              f" (10% of capital, +15% slack); if not, skip.")
        print("  3. Only if credit (bid of short - ask of long) > 0 after fees; max loss = width - credit per share.")
        print("  4. Exit: dynamic take-profit (trade_desk.py manage), else hold to expiry; no stop.")


if __name__ == "__main__":
    sys.exit(main())
