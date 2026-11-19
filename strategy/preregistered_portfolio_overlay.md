# Pre-registered test: portfolio overlay and tail stress (written 2026-11-19 ~02:10 CST, before any code or run)

Question: the index branch earns well under 1% a year on the account because ~90% of the money sits idle. When the
idle cash earns the 3-month T-bill rate and orders are worked from the mid-price, (a) what is the TOTAL account return,
and (b) does the spread sleeve add anything ON TOP of simply holding T-bills? Separately: what is the tail risk of
each candidate, given that the backtest has never seen a full loss? Code: `scripts/portfolio_overlay.py` (AG-56..60).

## Inputs (all frozen before this file)
- Trades: the capital rule v2 (strategy/preregistered_capital_fix.md, sha 0843ea5b32fc279a): shock+guard, short
  ~0.10 delta, long = narrower of $W and the 0.03-delta put, 14-35 DTE, dynamic take-profit, budget check.
  Account sizes $2,500, $5,000, $10,000, $25,000. Entries 2014-01-01 .. data end (T-bill data starts 2014).
- Rates: 3-month T-bill yield (LSEG US3MT=RR, decimal, annualized), previous business day's value, forward-filled,
  accrued daily as yield x days/360 on idle cash.
- Benchmarks: T-bills only on the whole account; SPY (LSEG adjusted close) buy and hold.

## Account model
Equity starts at the account size on 2014-01-02. A spread's collateral = its max loss (width - credit) x 100 x
contracts, held from entry to exit, earning NOTHING. Idle cash = equity - collateral of open spreads, earning the
T-bill rate. Spread P&L is booked on the exit day. One spread open at a time (the tested rule). No compounding of
spread size (size comes from the capital rule on the starting account), interest compounds daily.

## Fill model (secondary scenario; the primary run uses bid/ask fills, FILL = 1)
Each trade independently (seeded, 1,000 runs): 50% filled at a quarter of the way from mid to natural (FILL = 0.25),
35% filled halfway (FILL = 0.5), 15% not filled (trade skipped, collateral stays idle). The same FILL applies to the
entry and the exit of that trade. Reported: median and 5th-95th percentile of each metric across runs.

## Pass / fail: "the spread sleeve adds value over T-bills alone" (primary = FILL 1)
For each account size, on 2014-2026 calendar-year returns (2026 partial):
1. Mean yearly excess return (overlay - T-bills only) > 0 with the 95% bootstrap interval (5,000 resamples of
   years) above 0.
2. Excess positive in >= 70% of years.
3. Max drawdown of the overlay account better than -30% and worst month better than -25%.
4. Deflated Sharpe ratio of the monthly excess returns >= 0.95, counting n_trials = 200 (a conservative count of
   every configuration tried in this project so far: 48 + 144 + filters + widths + ETF runs).
All four -> PASS for that size. The headline TOTAL return (with T-bills) is reported but is NOT evidence of skill:
the T-bill part is the risk-free rate.

## Tail stress (reported for every candidate; not pass/fail)
- Parametric: per-trade full-loss probability p drawn from the Jeffreys posterior Beta(0.5, n + 0.5) given 0 full
  losses in n trades (p resampled each simulated year); trades per year ~ Poisson(observed rate); a non-loss trade
  draws its return on risk from the observed trades. 20,000 simulated years. Report mean, 5% VaR, 1% CVaR of the
  yearly account return (spread sleeve only, 10% risk per trade), P(year < 0), P(year < -10%).
- Historical: the SAME structure with the crash guard switched OFF (shock entries only) during 2018-01..2018-12,
  2020-02..2020-04, 2022-01..2022-12 and 2025-03..2025-04: list each trade's return on risk; the worst one is the
  "guard fails once" scenario applied to each account.
- Candidates stressed: capital rule v2 at the four sizes; the return branch's hindsight best
  (T10+guard s0.15/l0.05 21d, sizing D) LABELLED hindsight; the return branch's walk-forward OOS account.

```json
{"trades": "capital_v2", "capitals": [2500, 5000, 10000, 25000], "start": "2014-01-01", "rate": "US3MT=RR",
 "fill_model": {"0.25": 0.50, "0.5": 0.35, "skip": 0.15, "runs": 1000}, "checks": {"boot": 5000, "years_pos": 0.70,
 "max_dd": -0.30, "worst_month": -0.25, "dsr_min": 0.95, "n_trials": 200},
 "stress": {"prior": "jeffreys", "years": 20000, "windows": [["2018-01-01", "2018-12-31"], ["2020-02-01", "2020-04-30"],
 ["2022-01-01", "2022-12-31"], ["2025-03-01", "2025-04-30"]]}}
```
