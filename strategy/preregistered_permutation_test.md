# Pre-registered test: entry-timing permutation test of the index branch (written 2026-11-19 ~17:20 CST, before any run)

Question: does the index branch's ENTRY SIGNAL (SPY shock + crash guard + "no new low") earn more than the same
spreads sold on randomly chosen days? This is the same idea as a score permutation test: keep everything about the
strategy (structure, exits, sizing, account engine, number of entries) and destroy only the timing.

## Strategy under test (frozen, nothing re-tuned)
Capital rule v2 (strategy/preregistered_capital_fix.md, sha 0843ea5b32fc279a): short ~0.10 delta, long = narrower of
$W and the 0.03-delta put, 14-35 DTE nearest 21, dynamic take-profit, no stop, bid/ask fills, budget check, one spread
open at a time. Data 2010-01 .. 2026-10-28 (Kaggle SPY chains + rebuilt OPRA). Primary account $5,000; also $2,500
and $10,000 reported.

## Statistic
CAGR of the spread-only account: capital + cumulative spread P&L in $, from the first data day to the last exit
(no T-bill interest, so the test is about the spreads alone). Secondary: mean return on risk per trade.

## Null distributions (10,000 permutations each, seed 20261119)
- Null A (primary): draw the same number of entry dates as the signal produced (before the one-open rule) uniformly
  from ALL trading days on which the same structure can be traded; apply the one-open rule; compute the statistic.
- Null B (secondary): the same, drawing only from days where the crash guard is on (SPY above its 200-day average
  and 9-day IV not above 30-day IV). Tests whether the shock timing adds anything beyond the guard.

## Pass / fail
p = (1 + number of permutations with CAGR >= the real one) / (1 + 10,000).
PASS if p < 0.05 under Null A at the primary account size. Null B and the other sizes are reported, not scored.
Note: a large share of permutations will make money too (selling far out-of-the-money puts usually does); the test is
whether the signal beats random timing, not whether it is profitable.

```json
{"test": "entry_permutation", "strategy": "capital_v2", "capitals": [2500, 5000, 10000], "primary": 5000,
 "n_perm": 10000, "seed": 20261119, "stat": "spread_only_cagr", "nulls": ["all_days", "guard_days"], "alpha": 0.05}
```
