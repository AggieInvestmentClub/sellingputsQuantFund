# Overnight optimization report: 19 Nov 2026

Branch: `overnight-quant-optimization-20261119` (local commits, **not pushed**). Push it with
`git push -u origin overnight-quant-optimization-20261119`, then open a pull request into `main`.
Teammates' uncommitted edits in the working folder were not touched or committed.

## Bottom line

The only strategy that passes its pre-registered tests is still the **index branch** (SPY drop + crash guard, short
~0.10 delta, sold on XSP). It now has a corrected sizing rule and passes at $2,500 / $5,000 / $10,000 / $25,000.
**Every attempt to raise returns FAILED its pre-registered test tonight:**

- a wider family of SPY rules
- the same rule on 28 ETFs
- the SPY hindsight-best rule moved to those ETFs
- the agent's valuation and confidence filters

Putting the idle cash in T-bills raises the total account return to about 2.2-2.5% a year (2014-2026). Almost all of
that is the risk-free rate, not skill. The spread sleeve's own contribution (+0.1% to +0.5% a year) does not clear
the deflated-Sharpe bar.

## Scoreboard

| Workstream | Pre-registration (sha256 prefix) | Sample | Key result (95% CI) | Max DD | DSR / PBO | Verdict |
|---|---|---|---|---|---|---|
| Index branch walk-forward (baseline, earlier) | preregistered_index_branch.md (4d5605e152a14547) | 53 OOS trades 2014-26 | +2.42% [+1.96, +2.82] per trade | −0.6% | — | PASS (0 full losses seen) |
| Capital rule v2, $2,500 | preregistered_capital_fix.md (0843ea5b32fc279a) | 35 trades 2010-26 | +0.97% [+0.06, +1.80] | −1.3% | — | PASS (borderline) |
| Capital rule v2, $5,000 | same | 43 | +1.85% [+1.21, +2.46] | −0.8% | — | PASS |
| Capital rule v2, $10,000 | same | 46 | +2.14% [+1.77, +2.53] | −0.1% | — | PASS |
| Capital rule v2, $25,000 | same | 46 | +1.95% [+1.75, +2.16]; max risk 10.4%/trade (was 22%) | 0.0% | — | PASS |
| ETF branch, 29 ETFs | preregistered_etf_branch.md (1220b9d95e36bec8) | 54 / 68 / 72 trades, 2024-04..2026-10 | $2.5k −4.33% [−10.52, +0.53]; $5k −2.98% [−7.96, +0.84]; $10k −1.88% [−6.64, +1.74] | −21% / −21% / −16% | — | **FAIL** |
| Higher-return SPY family (144 candidates) | preregistered_return_branch.md (8a48b4d403e0b7fd) | walk-forward 2014-26 | CAGR +0.83% vs +0.85% current; yearly diff +0.39% [−4.67, +4.86]; 69% years positive | −31.0% | DSR(best) 1.00, PBO 0.37 | **FAIL** |
| ETF transfer of the SPY hindsight best | preregistered_etf_transfer.md (abc314297fba2418) | 131-135 trades | $5k −2.39% [−7.09, +1.27]; account −29% | −30.5% | — | **FAIL** |
| T-bill overlay: spread sleeve vs T-bills only | preregistered_portfolio_overlay.md (e627dcaa03c6777d) | 2014-26 daily | excess/yr $2.5k +0.13% [−0.09, +0.34]; $5k +0.38% [+0.12, +0.66]; $10k +0.48% [+0.23, +0.75]; $25k +0.37% [+0.19, +0.56] | −0.5% to 0.0% | DSR 0.00 / 0.09 / 0.75 / 0.93 (need 0.95) | **FAIL** (all sizes) |
| Agent ablation H1 (valuation gate within TAKEs) | preregistered_agent_ablation.md (b0cf3c86e503460b) | post-cutoff, hold-out n=4 | +13.05% vs +3.60% | — | DSR 0.62 | **FAIL** (n < 10) |
| Agent ablation H2 (confidence ≥ 0.7 TAKEs) | same | hold-out n=18 | +5.70% [−10.70, +13.50] vs −0.74% | — | DSR 0.67 | **FAIL** |
| Agent ablation H3 (TAKE & fv_pass vs rest) | same | hold-out n=4 | +13.05% vs −0.50% | — | DSR 0.62 | **FAIL** (n < 10) |

## Phase 1: data and bug fixes

- **SPY option file.** `data/processed/options/spy_puts_2024_2026.csv` was rebuilt from the OPRA day files.
  - It now runs 2024-04-01 to 2026-10-28: 1,750,256 rows over 648 days, 0 nulls and 0 malformed tickers.
  - It matches the truncated file row for row before 2026-08-17 (1,600,984 rows, none missing on either side).
  - The truncated copy is kept as `..._TRUNCATED_at_2026-08-17.csv`.
  - With the full file, the index branch shows no new entries between Aug and Oct 2026, so its results are unchanged.
- **$25,000 sizing bug.** Fixed as capital rule v2 (AG-28f, `--stage capital_v2`).
  - The long leg is now the narrower of $W and the 0.03-delta put.
  - The budget check applies at every account size. Max risk per trade at $25,000 fell from 16-22% to 10.4%.
  - `index_signal.py` now prints this rule and says to trade **XSP**, not SPY.
- **Legacy trade quarantined.** `trade_desk.py index`, the weekly 0.15-delta $1 XSP trade that failed, now exits
  with DEPRECATED, and `index_plan()` raises.
- **Tests:** `tests/test_overnight_fixes.py` (13).

## Phase 2: ETF branch and correlation

- **Trade frequency:** the rule fires far more often on 29 ETFs than on SPY alone: 21-28 trades a year against 1-2.
- **Results:** the trades lose money on average (table above). The losses come from equity ETFs:
  - Two full losses at each size: XLE (Dec 2024), plus QQQ (Jul 2024) or ARKK (Nov 2025).
  - Non-equity ETFs (GLD, SLV, TLT, HYG, GDX, USO) were roughly flat to slightly positive.
- **Correlation:** daily returns of the traded equity ETFs average 0.49 correlation, rising to **0.77 in stress
  windows** (Aug 2024, Mar-Apr 2025). Spreading across ETFs stacks the same equity risk rather than diversifying it.
  - Up to 3 spreads were open at once, using 26-29% of capital.
  - Source: `scripts/etf_correlation.py`, `data/results/etf_branch/correlation.json`.

## Phase 3: return ceiling, tail risk, idle cash

### Higher-return SPY family

| Account | CAGR | Max DD | Trades/yr | Full losses |
|---|---|---|---|---|
| Walk-forward pick (scored) | +0.83% | −31.0% | 12.2 | 3 |
| Current rule | +0.85% | −0.6% | 3.6 | 0 |
| Best in hindsight: T10+guard 0.15/0.05 21d, 20% risk | +3.81% | −5.4% | — | — |

The hindsight best looks attractive, but it was found after looking at all the results. Its only clean test, on
other ETFs, **failed**. Fills at a quarter of the way from mid do not change the verdict.

### Tail stress

The full-loss rate is drawn from a Beta(0.5, n+0.5) posterior, because 0 full losses were seen, over 20,000
simulated years:

| Candidate | Mean/yr | 5% VaR | 1% CVaR | P(losing year) |
|---|---|---|---|---|
| Capital rule v2, $2,500 | −0.13% | −0.69% | −10.29% | 23% |
| Capital rule v2, $5,000 | +0.06% | −0.43% | −11.49% | 15% |
| Capital rule v2, $10,000 | +0.19% | −0.04% | −11.30% | 5% |
| Capital rule v2, $25,000 | +0.13% | 0.00% | −10.74% | 3% |
| Return-branch hindsight best | +2.92% | −3.77% | −21.54% | 12% |
| Return-branch walk-forward pick | +0.52% | −18.29% | −39.54% | 37% |

With the guard switched off inside the crash windows (2018, Feb-Apr 2020, 2022, Mar-Apr 2025), there are 53 trades
and 2 full losses, each about −9.5% of the account. This is what the guard has avoided so far. The 2008 crash
cannot be tested: the option data starts in 2010.

### Idle cash in T-bills

- 3-month T-bills alone: +2.04% a year from 2014 to 2026.
- With the spread sleeve, at bid/ask fills: +2.17% / +2.41% / +2.52% / +2.41% a year at $2.5k / $5k / $10k / $25k.
- With the fill model (median of 1,000 runs): +2.31% / +2.56% / +2.54% / +2.39%.
- The sleeve adds measurable excess at $5k-$25k, but it fails the deflated-Sharpe check counting 200 tried
  configurations.
- **Practical takeaway:** keep idle cash in T-bills or a money-market fund. That is most of the account's return.

## Phase 4: AI agent

- **Audit:** 10 issues found; the full list is in `data/results/agent_diag/audit.md`.
  - 31 historical failures were network errors with no retry.
  - Text before the JSON silently dropped answers.
  - v1 masked ticker words such as ON, IT, ALL, NOW, KEY, LOW and CAT case-insensitively, garbling text in 73 shocks.
  - Dollar figures, executive names and peer names leaked into prompts.
  - A take-profit resting price of $0.00 was possible.
  - No look-ahead problem was found in the news or valuation timing.
- **`llm.py` hardening:**
  - Retries with exponential backoff and jitter.
  - A circuit breaker.
  - `safe_call_llm`: if the endpoint is unreachable, the verdict becomes "unavailable", so there is no TAKE and no hang. Verified on the laptop: 27 s, logged in `llm_failures.log`.
  - Robust JSON extraction.
- **Anonymization:** a strict version for v2 with case-safe tickers, peer names masked, titled people masked, and
  dollar figures replaced by buckets relative to market value.
- **Prompt v2** (`run_agent.py --version ag-v2`): structured sections for `<impairment>`, `<shock_type>`,
  `<vol_context>` and `<overreaction>`. v1 stays the default so the cached results remain comparable.
  - v2 has **not** been run against GPT-4.1 yet, because the TAMU endpoint is not reachable from the overnight
    environment. Run it from Windows; the command is below.
- **Ablation on post-cutoff shocks:** all three hypotheses fail; the hold-out samples are 4-18 trades.
- **Macro regime classifier** (`src/agent/macro.py`): a descriptive prototype printed by `daily.py`. It is not a
  trading rule. On 18 Nov it reads "unclear" because SPY is 0.3% below its 200-day average.
- **Execution:** the resting take-profit is never below $0.01, it closes at the natural price when that already
  meets the rule, it warns on wide markets, XSP is the default underlying, and SPY prints an assignment warning.
- **Tests:** suite of 861 passing (41 new in `tests/test_agent_upgrades.py`, 6 in `tests/test_portfolio_overlay.py`).

## What to do today (PowerShell, repo folder, after 15:00 CST, LSEG Workspace open)

```powershell
# 1. Daily run
py -3.12 -u scripts/daily.py --capital 5000 --risk-usd 500

# 2. Only if it prints "ENTRY SIGNAL": a teammate approves, then build the ticket from live XSP quotes
py -3.12 -u scripts/order_desk.py open --underlying XSP --expiry YYYY-MM-DD --short K1 --long K2 --sb .. --sa .. --lb .. --la .. --capital 5000

# 3. Log the fill, keep the take-profit order resting, check fill quality
py -3.12 -u scripts/order_desk.py fill ...
py -3.12 -u scripts/order_desk.py tp
py -3.12 -u scripts/order_desk.py report --capital 5000

# 4. Optional: try agent prompt v2 against GPT-4.1 on one event (Windows reaches the TAMU endpoint)
py -3.12 -u scripts/run_agent.py --version ag-v2 --ric MOH.N --date 2024-04-02

# 5. Publish the overnight work
git push -u origin overnight-quant-optimization-20261119
```

Nothing sends orders. Every trade is a ticket that a teammate approves and places by hand.

## Recommendations

1. **Trade only the index branch, small, on XSP, at $5,000 or more.** Treat it as a live test, not an income
   source, because its tail is unmeasured.
2. **Keep idle cash in T-bills or a money-market fund.**
3. **Stop searching on the same data.** Every new variant tonight failed, and each extra test adds to
   multiple-testing risk. Real progress now needs one of:
   - (a) OptionMetrics via WRDS, for real bid/ask and the 2008-2022 crashes;
   - (b) a forward paper-trading record of the index branch and the agent (v2), scored against a pre-registered plan.
4. **Single-stock TAKEs stay on paper** until there are about 100 post-cutoff TAKEs.
