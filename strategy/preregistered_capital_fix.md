# Pre-registered correction: capital rule v2 (written 2026-11-19 ~02:00 CST, before any run of v2)

Why: the capital rule in strategy/preregistered_index_branch_small.md ("Capital rule", AG-28d) switches to the
0.03-delta long leg when W >= 15 and then applies NO budget check. In 2024-2026, with SPY near $550-730, that leg sat
$36-55 below the short strike, so one spread risked 16-22% of a $25,000 account instead of the intended 10%.
The intent of the rule (max loss per trade <= 10% of capital) was not met. That file is frozen and is NOT edited;
this file states the corrected rule. Results of v1 stay in data/results/index_branch/capital_*_trades.csv.

## Rule v2 (AG-28f)
- Budget B = capital x 10%; W = floor(B / $100) dollars (unchanged).
- Long leg = the NARROWER of (a) the strike nearest to (short strike - W) and (b) the put nearest to -0.03 delta
  (same expiry). This is the rule the ETF branch already uses (strategy/preregistered_etf_branch.md).
- A trade is taken only if its max loss (width - credit) x 100 <= 1.15 x B, for EVERY account size (v1 skipped this
  check when W >= 15). One contract per trade (more contracts only if they fit inside B).
- Everything else is unchanged: shock+guard entry, short ~0.10 delta, 14-35 DTE nearest 21, dynamic take-profit,
  bid/ask fills (FILL = 1), $0.65 per contract per leg, 2010-2026 on Kaggle (2010-2023) + rebuilt OPRA (2024-2026,
  complete through 2026-10-28; the old file was truncated at 2026-08-17).
- Same 4 checks as AG-28d: mean return on risk > 0 with 95% bootstrap interval above 0; positive in >= 70% of years
  with trades; worst month better than -25%; max drawdown better than -30%.
- Account sizes run: $2,500, $5,000, $10,000, $25,000 (each reported; no size is chosen from the results).

## What was already seen
v1 results for all four sizes (truncated data). v2 changes the result only where the 0.03-delta leg is narrower
than W (expected rare below $10,000) and at $25,000 (where the delta leg was wider than the budget allowed).

```json
{"rule": "capital_v2", "risk": 0.10, "long_leg": "narrower_of_W_and_0.03d", "budget_filter": 1.15,
 "capitals": [2500, 5000, 10000, 25000], "entry": "shock+guard", "short_delta": 0.10, "exit": "tp", "fill": 1.0}
```
