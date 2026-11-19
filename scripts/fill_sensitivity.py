"""
[AG-28e] Fill sensitivity for the index branch: how much of the return do bid/ask fills cost?

Every test so far sold at the bid and bought at the ask (FILL = 1). An execution agent that works limit orders could
get better prices. This reruns the capital rule (AG-28d) with FILL = 1, 0.5, 0.25 and 0 (mid) for several account sizes.
FILL = 0 is an upper bound, not a promise: nobody fills every order at mid. This is a sensitivity, not a pass/fail test,
so it changes nothing in the pre-registered results.

    py -3.12 -u scripts/fill_sensitivity.py
    py -3.12 -u scripts/fill_sensitivity.py --capital 2500 5000 --fills 1 0.5 0
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.index_branch as IB  # noqa: E402

END = pd.Timestamp("2026-11-18")


def run(f, by, ix, spot, capital, fill, risk=0.10):
    IB.FILL = fill
    budget = capital * risk
    W = int(budget // 100)
    struct = (0.10, ("delta", 0.03)) if W >= 15 else (0.10, ("abs", float(W)))
    rows = [IB.trade(d, by, ix, spot, struct, "tp") for d in IB.entry_dates(f, "shock+guard")]
    t = pd.DataFrame([r for r in rows if r])
    if W < 15:
        t = t[(t["width"] - t["credit"]) * 100 <= budget * 1.15]
    raw, acct = IB.stats(t, risk), IB.stats(IB.one_open(t), risk)
    o = IB.one_open(t)
    yrs = (END - pd.Timestamp("2010-01-01")).days / 365.25
    recent = o[o["date"] >= "2024-01-01"]
    return {"capital": capital, "width": "0.03d" if W >= 15 else f"${W}", "fill": fill, "n": raw["n"],
            "mean": raw["mean"], "ci_lo": raw["ci_lo"], "ci_hi": raw["ci_hi"], "win": raw["win"],
            "acct_total": acct["total"], "per_year_$": acct["total"] * capital / yrs,
            "per_year_%": acct["total"] / yrs, "since_2024_$": float((recent["ror_net"] * budget).sum()),
            "credit_$": float(t["credit"].mean() * 100), "passes_check1": raw["ci_lo"] > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, nargs="+", default=[2500, 5000, 10000, 25000])
    ap.add_argument("--fills", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.0])
    a = ap.parse_args()
    f, by, ix, spot = IB.build("capital")
    res = []
    for c in a.capital:
        for fl in a.fills:
            r = run(f, by, ix, spot, c, fl)
            res.append(r)
            print(f"${c:>7,.0f} {r['width']:>6s} fill={fl:<4} n={r['n']:3d} mean={r['mean']:+.2%} "
                  f"[{r['ci_lo']:+.2%}, {r['ci_hi']:+.2%}] win={r['win']:.0%} account {r['acct_total']:+.1%} "
                  f"= ${r['per_year_$']:,.0f}/yr ({r['per_year_%']:+.2%}/yr) | since 2024 ${r['since_2024_$']:+,.0f} "
                  f"| avg credit ${r['credit_$']:.0f}", flush=True)
    IB.FILL = 1.0
    pd.DataFrame(res).to_csv(IB.OUT / "fill_sensitivity.csv", index=False)
    print(f"\nsaved {IB.OUT / 'fill_sensitivity.csv'}")


if __name__ == "__main__":
    main()
