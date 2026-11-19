# Pre-registered test: index branch, walk-forward (frozen 2026-11-18 ~22:25 CST, before the run)

History of this test (kept for honesty):
- 2026-11-18 22:15: design stage on 2010-2016 picked `shock | S2 short 0.15d / long 0.05d | tp+7dte` from 48 fixed
  candidates (t = 4.99, deflated-Sharpe probability 0.98).
- 22:20: validation on 2017-2020 FAILED (-3.15% per trade, n = 41). That single-rule test is closed.
- This file replaces it with a walk-forward PROCEDURE, written before running it. No rule below is chosen by
  looking at 2017-2026 results; the only 2017-2020 result seen is the one failed candidate above.

## Candidates (unchanged, fixed in scripts/index_branch.py before any real run)
8 entry rules x 3 structures x 2 exits = 48. Entry rules: weekly, weekly+guard, vrp, vrp+guard, shock, shock+guard,
shock+vrp+guard, shock+guard+cap. Structures: short 0.15d / long 5% lower; short 0.15d / long 0.05d;
short 0.10d / long 0.03d. Exits: dynamic take-profit; take-profit or close at 7 days left.
Fills: Kaggle 2010-2023 at real bid/ask; OPRA 2024-2026 closes -/+ max($0.01, 5%); $0.65 per contract per leg.

## Procedure
For each calendar year Y = 2014 .. 2026:
1. Using ONLY trades that exited before Jan 1 of Y (expanding window from 2010), compute each candidate's
   per-trade t-statistic of return on risk. Candidates with fewer than 20 trades are not eligible.
2. Trade the candidate with the highest t in year Y (entries dated in Y).
3. Stitch the year-Y trades together. One position at a time, 10% of capital at risk per trade.

## Pass / fail (decided now), on the stitched 2014-2026 out-of-sample trades, net of costs
1. Mean return on risk per trade > 0, with the 95% bootstrap interval above 0.
2. Positive in at least 70% of calendar years.
3. Worst calendar month better than -25% of the account.
4. Maximum drawdown better than -30% of the account.
Fail any -> not proven. No re-runs with a changed procedure.

```json
{"procedure": "walk_forward", "first_year": 2014, "last_year": 2026, "min_trades": 20, "score": "t", "risk": 0.10}
```
