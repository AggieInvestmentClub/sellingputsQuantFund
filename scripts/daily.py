"""
One daily run of the fund's agent (AG-18). Run after 15:00 Matamoros (16:00 New York) with LSEG Workspace open.

    py -3.12 -u scripts/daily.py --capital 1000 --risk-usd 100

Steps, each the existing script:
  1. live_scan.py --days 3        prices (S&P 500, SPY, VIX), today's shocks, news, agent verdicts
  2. index_signal.py              CORE: the tested index branch (SPY shock + crash guard), entry days only
  3. trade_desk.py plan           SATELLITE, PAPER ONLY: entry plans for agent TAKEs, logged to paper_trades.csv
  4. trade_desk.py manage         HOLD / CLOSE for every open spread in positions.csv
  +  order_desk.py tp             resting take-profit (buy-to-close) prices; order_desk.py open/fill/report on entry days
Shock trades stay on paper until score_signals.py shows the agent's TAKEs beating the rest out of sample
(target: 100+ post-cutoff TAKEs). Nothing here sends orders; a teammate approves each one in the broker.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

OUT = Path("data/results/agent")


def run(args):
    print(f"\n=== {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, "-u", *args])
    if r.returncode:
        print(f"(step failed with code {r.returncode}; continuing)")
    return r.returncode


def log_paper():
    """append today's ENTER TOMORROW shock plans to paper_trades.csv (one row per shock, first plan wins)"""
    scans = sorted(OUT.glob("live_*.csv"))
    if not scans:
        return
    day = scans[-1].stem.split("_")[1]                   # only the plan made from the latest scan
    f_plan = OUT / f"plan_{day}.csv"
    if not f_plan.exists() or f_plan.stat().st_mtime < scans[-1].stat().st_mtime:
        return
    p = pd.read_csv(f_plan)
    p = p[p["action"] == "ENTER TOMORROW"]
    if p.empty:
        return
    f = OUT / "paper_trades.csv"
    old = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["ric", "shock_date"])
    new = p[~p.set_index(["ric", "shock_date"]).index.isin(old.set_index(["ric", "shock_date"]).index)]
    pd.concat([old, new], ignore_index=True).to_csv(f, index=False)
    print(f"paper trades logged: {len(new)} new, {len(old) + len(new)} total in {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, required=True)
    ap.add_argument("--risk-usd", type=float, required=True, help="max loss on one spread, in $")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--skip-scan", action="store_true", help="reuse today's scan (no LSEG calls)")
    a = ap.parse_args()
    if not a.skip_scan:
        run(["scripts/live_scan.py", "--days", "3"])
    # The weekly XSP spread (trade_desk index) FAILED its tests (strategy/preregistered_spy.md, 2010-2023),
    # so the index step is now the tested index-branch signal (strategy/preregistered_index_branch.md).
    run(["scripts/index_signal.py", "--capital", str(a.capital)])
    try:   # [AG-54] macro regime, information only (prototype; not a trading rule)
        from src.agent import macro as M
        px = M.load_closes()
        d = px.index[-1]
        c = M.classify(M.macro_features(px, d))
        print(f"\nMacro regime {d:%Y-%m-%d} (AG-54 prototype, information only): {c['label']} - {'; '.join(c['reasons'])}")
    except Exception as e:  # noqa: BLE001
        print(f"\n(macro regime unavailable: {e})")
    run(["scripts/order_desk.py", "tp"])   # AG-38: resting buy-to-close limits for open spreads (re-place each morning)
    print("On an ENTRY SIGNAL: scripts/order_desk.py open --underlying XSP --expiry YYYY-MM-DD --short K --long K "
          f"--sb .. --sa .. --lb .. --la .. --capital {a.capital:.0f}   (limit-ladder ticket; then log the fill with "
          "order_desk.py fill)")
    if any(OUT.glob("live_*.csv")):
        run(["scripts/trade_desk.py", "plan", "--capital", str(a.capital), "--width", str(a.width),
             "--risk-usd", str(a.risk_usd)])
        log_paper()
    run(["scripts/trade_desk.py", "manage"])
    print("\nIndex: act only on an ENTRY SIGNAL above, sized per scripts/index_branch.py --stage capital. "
          "Shock plans: paper only. Nothing here sends orders; a teammate approves each one.")


if __name__ == "__main__":
    main()
