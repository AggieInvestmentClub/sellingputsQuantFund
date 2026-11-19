# Pre-registered test: index branch, account-size version (frozen 2026-11-19 ~00:20 CST, before the run)

Question: does the walk-forward rule still work when the spread is narrow enough for a ~$1,000 account?
Only the WIDTH changes. Entry, short strike and exit are the rule the walk-forward chose from 2018 on
(strategy/preregistered_index_branch.md):

- Entry: SPY 1-day return <= -1.5% or 2-day <= -2.5%; enter day +1..+3 on the first close above the post-shock low;
  only if SPY > its 200-day average and ATM put IV at ~9 days <= ATM put IV at ~30 days.
- Short put: delta closest to -0.10, expiry 14-35 days closest to 21.
- Long put: the listed strike closest to (short strike - W), W = $1 (primary), $2 and $3 reported as sensitivity.
- Fills: Kaggle 2010-2023 real bid/ask; OPRA 2024-2026 closes -/+ max($0.01, 5%); $0.65 per contract per leg.
- Exit: dynamic take-profit (STR-3.6a), else hold to expiry; no stop.
- Account: one position at a time; one spread per trade (risk = width - credit, ~$90-100 at W = $1 on $1,000 = ~10%).

Period: 2010-01 .. 2026-10 (all data; the rule's entry was chosen on these years, so this is a robustness test of
the width, not a fresh out-of-sample test of the entry rule).

## Pass / fail for W = $1 (decided now)
1. Mean return on risk per trade > 0, 95% bootstrap interval above 0.
2. Positive in >= 70% of calendar years with trades.
3. Worst calendar month better than -25% of the account (at 10% risk per trade).
4. Max drawdown better than -30% of the account.
W = $2 and $3 are reported, not scored.

```json
{"entry": "shock+guard", "short_delta": 0.10, "widths": [1.0, 2.0, 3.0], "primary": 1.0, "exit": "tp", "risk": 0.10}
```

## Capital rule (added 2026-11-19 ~00:30 CST, before any run with W >= 4)
Because the fund's capital is not fixed, the width is set by the account, not chosen by results:
W = largest whole dollar with W x 100 <= capital x 10%. W >= 15 uses the tested delta-based long leg (0.03 delta).
Same entry, short strike, exit and the same 4 pass/fail checks. Run: `scripts/index_branch.py --stage capital --capital N`.
Already seen before this rule: W = 1 (FAIL), 2 and 3 (positive, reported). W >= 4 is untested at writing time.
