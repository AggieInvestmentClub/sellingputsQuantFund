# Pre-registered test: return branch (frozen 2026-11-19 ~01:20 CST, before any run of this family)

Question: can the SAME elements (SPY shock entry, day +1..+3 "no new low", crash guard, bull put spread, dynamic
take-profit) earn materially more on the ACCOUNT than the index branch (~0.5-1% a year at 10% risk per trade),
out of sample, without an unacceptable drawdown? Code: `scripts/return_branch.py` (AG-41..45).

## What was already seen before writing this (kept for honesty)
- The AG-28 family on 2010-2026 (`data/results/index_branch/all_candidate_trades.csv`): shock and shock+guard at
  -1.5% / -2.5%, short 0.10/0.15 delta, ~21 DTE, tp and tp+7dte, one position at a time, 10% risk.
- The walk-forward result of strategy/preregistered_index_branch.md, the capital test and the fill sensitivity.
- NOT seen: the -1.0% threshold, short 0.20 delta, 7-14 and 30-45 DTE, overlapping positions, 20% risk.

## Candidate family (fixed now): 4 entries x 3 structures x 3 tenors x 4 sizings = 144
- Entry (4): shock threshold T15 = 1-day <= -1.5% or 2-day <= -2.5%; T10 = 1-day <= -1.0% or 2-day <= -2.0%;
  each with the crash guard (SPY above its 200-day average and ~9-day ATM put IV not above ~30-day) or without it
  (kept so the tail is visible). Entry day +1..+3: the first day SPY closes above the post-shock low (AG-28a).
- Structure (3): short 0.10 delta / long 0.03 delta; short 0.15 / long 0.05; short 0.20 / long 0.05.
- Tenor (3): 7-14 DTE (target 10), 14-35 DTE (target 21), 30-45 DTE (target 38). Exit: dynamic take-profit
  (STR-3.6a: >= 50% kept and ahead of time by 20 pts, or >= 75%), else hold to expiry and settle. No stop.
- Sizing (4): A = 1 open, 10% of the account at risk; B = up to 3 open, 10% each, total open risk <= 30%;
  C = 1 open, 20% at risk; D = up to 2 open, 20% each, total open risk <= 40%.
- Fills: sell at the bid, buy at the ask (FILL = 1), $0.65 per contract per leg. Kaggle real bid/ask 2010-2023;
  OPRA closes -/+ max($0.01, 5%) 2024-2026 (modeled, not real quotes).

## Account accounting (same for every candidate and for the comparison)
Risk of a new position = sizing fraction x account equity at entry (realized P&L only). P&L = return on risk x that
risk, booked on the exit day. A new entry is skipped if the open-position limit or the open-risk cap would be
exceeded. Equity, drawdown and months are on booked P&L (no daily mark-to-market: this understates intra-trade
drawdown, so the maximum open risk is reported next to it).

## Walk-forward procedure
For each year Y = 2014 .. 2026:
1. Simulate every candidate on entries from 2010-01-01 whose exit is before Jan 1 of Y (expanding window).
2. Eligible: >= 20 trades taken, max drawdown better than -20%, worst month better than -10%.
3. Pick the eligible candidate with the highest CAGR over 2010-01-01 .. Y-01-01 (ties: the smaller sizing A<B<C<D).
   None eligible -> no new entries in Y.
4. In Y, take the pick's entries dated in Y with the pick's sizing. Positions carry across year ends.
One stitched account from 2014-01-01 (equity 1.0) to the last exit (data end 2026-11-18).

Comparison ("current rule"): the stitched out-of-sample trades of strategy/preregistered_index_branch.md
(`data/results/index_branch/walkforward_oos_trades.csv`), run through the same accounting with sizing A.

## Pass / fail (decided now), on the stitched 2014-2026 out-of-sample account
1. CAGR above the current rule's CAGR, AND the 95% bootstrap interval (5,000 resamples of the paired calendar-year
   returns, 2014-2026) of the mean yearly difference (new - current) above 0.
2. Positive in at least 70% of the calendar years with trades.
3. Maximum drawdown better than -25%.
All three -> PASS. Any fail -> not proven. No re-run with a changed procedure.

Reported, not scored: the best full-period (2010-2026) candidate under the same eligibility, next to the
out-of-sample result (the gap = what overfitting would have promised); deflated Sharpe and PBO (CSCV, 8 blocks) over
the 144 candidates' monthly account returns 2010-2026; return per unit of max drawdown, worst trade, number of
trades losing >= 90% of risk, max open risk; FILL = 0.25 as a sensitivity; the 2026 pick translated to $2,500 /
$5,000 / $10,000 accounts with the capital width rule (W = floor(capital x risk / $100); W >= 15 -> the delta long
leg), in $ per year over 2014-2026 (that translation applies the latest pick to every year, so it is not walk-forward).

```json
{"family": {"entry": ["T15+guard", "T15", "T10+guard", "T10"], "structure": [[0.10, 0.03], [0.15, 0.05], [0.20, 0.05]],
 "tenor": [[7, 10, 14], [14, 21, 35], [30, 38, 45]], "sizing": {"A": [1, 0.10, 0.10], "B": [3, 0.10, 0.30],
 "C": [1, 0.20, 0.20], "D": [2, 0.20, 0.40]}, "exit": "tp", "fill": 1.0},
 "walk_forward": {"first_year": 2014, "last_year": 2026, "min_trades": 20, "max_dd": -0.20, "worst_month": -0.10,
 "score": "cagr"}, "checks": {"boot": 5000, "years_pos": 0.70, "max_dd": -0.25}}
```
