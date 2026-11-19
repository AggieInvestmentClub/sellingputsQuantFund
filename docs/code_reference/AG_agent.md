# AG: shock-screening agent (`src/agent/`)

The backtest (11/12 update) found two things that shape this module: shocks don't beat random no-shock
days (+2.2% vs +4.1% return on risk), and the LLM filter's "not justified" label did no better than
"has Reuters news". So the agent's only job is to **pick which shocks to trade**, better than both.
It sees the same anonymized news as the LLM filter (LLM-01..03) **plus market evidence** an analyst would
check, and must make an explicit take / skip call. Trade mechanics stay as backtested: short delta 0.15
below the post-news low, enter by day +3, no stop, 2% of capital at risk.

Run with `scripts/run_agent.py` (history / one shock) and `scripts/live_scan.py` (today, needs LSEG).

## No cheating: what the agent adds to the LLM filter's rules

| Risk | How the code handles it |
|---|---|
| Future prices in the evidence | AG-02/03 slice prices at the shock date; trend returns stop the day BEFORE the shock. Tested in tests/test_agent.py. |
| News stamped earlier than it was published | DAT-09 cut **plus** AG-06 `_drop_late_ids`: drops rows whose storyId date is after the shock day. Found in the old `data/raw/news/events/` files: stories from weeks later stamped "shock day 00:00" with `before_decision=True` (PAYC.N 2024-05-31 had 4). |
| Model knows how the story ended | Same as LLM filter: use a model whose training data ends before the events (Llama 3 70B via OpenRouter: Dec 2023), or only events after `confirmed_cutoff` (`post_cutoff` column). |
| Bad answers traded | Unparseable or anything but "take" → skip (AG-09/10). |

---

### AG-01  load_close / abnormal_on / members_on
**File:** src/agent/evidence.py  **Status:** ✅ done

**What it does:** Adjusted close + volume from `data/raw/prices/daily/<RIC>_adj.parquet`; shock-day abnormal return with the EVT-01 beta (matches events.csv: PAYC.N 2024-05-31 −9.76%); RICs in the universe on a date (security_master span + 62-day grace, last_member is a month-end snapshot).

### AG-02  price_context
**File:** src/agent/evidence.py  **Status:** ✅ done

**What it does:** Returns over 5 / 21 / 63 days **before** the shock, distance from 52-week high and 200-day average, 60-day realized vol, shock-day volume vs 20-day average.

**Returns:** dict, or None if < 70 days of history or the date is not a trading day for the stock.

### AG-03  peer_context
**File:** src/agent/evidence.py  **Status:** ✅ done

**What it does:** Same-day returns of the stock's GICS industry (else industry group) members: median and share down ≥ 3%. Tells a stock-specific drop from a sector repricing.

**Gotchas:** Needs ≥ 3 peers with prices that day; reads one price file per peer (≤ 40).

### AG-04  earnings_context
**File:** src/agent/evidence.py  **Status:** ✅ done

**What it does:** Days since the last report; whether the next report falls within 35 days (inside a spread's life). Report dates are announced weeks ahead, so this is known at the decision.

### AG-05  market_cap_before
Same rule as run_llm_filter.cap_before: last value strictly before the shock day.

### AG-06  event_news
**File:** src/agent/evidence.py  **Status:** ✅ done

**What it does:** ING-23 / ING-24 files (`news/reuters_events`, `news/reuters_stories`, `<RIC>_<YYYYMMDD>.parquet`), first versions (DAT-08), cut at the close (DAT-09), late storyIds dropped.

### AG-07 / AG-08  context_block / build_agent_prompt
**File:** src/agent/screen.py  **Status:** ✅ done

**What it does:** LLM-03's anonymized prompt, its JSON spec replaced by the market evidence (rounded numbers only) and the decision spec (adds `market_read`, `decision`, `key_risk`). Tells the model that random-day put selling does as well, so "take" should be the minority.

### AG-09  parse_agent
LLM-05 plus `decision` / `market_read` / `key_risk`. Malformed or any decision other than take/skip → "skip".

### AG-10  final_call
**What it does:** `agent_take` only if: model said take, answer parsed, verdict "not justified", category tradeable (STR-2.3a), no earnings inside the window (STR-2.2a). Reports the overreaction ratio (STR-2.4a) and `ratio_ok` (≥ 3) without requiring it, so the backtest can test it as a separate filter. `why_not` lists every failed check.

### AG-11  scripts/live_scan.py
**What it does:** After the close, with LSEG open: appends missing days to the daily price files (current S&P 500 + SPY), finds shocks with the EVT-04 rule, skips < $5B, pulls Reuters news, runs the agent. `--days 3` also scans the 2 prior trading days (entry allowed to day +3). Output: `data/results/agent/live_<date>.csv` with post-news low and close for strike picking.

**Gotchas:** Before 16:00 New York the day's bar is not final (it warns). Strike, expiry and size are picked by a person in the broker; the agent never sends orders.

## How to test it before trading (backtest)

1. `python -u scripts/run_agent.py --start 2024-04-01` → `data/results/agent/agent_results.csv` (event_id = `RIC|YYYY-MM-DD`, same as the LLM filter).
2. In the backtest's event filter (BT-07 event_mask), add sets from that file: `agent_scored` (all screened), `agent_take` (agent_take True), `agent_skip`.
3. It works only if, out of sample: **agent_take > agent_skip** and **agent_take beats the random-entry run** (grid_random). With ~40-200 trades per setting, treat anything short of that as "not proven".

### AG-12 / AG-13  entry_plan / manage  (src/agent/trade_desk.py, scripts/trade_desk.py)
**What it does:** turns each TAKE into an order plan and tells you each day whether to close.
- `plan --capital N`: ENTER TOMORROW / WAIT / DROP per TAKE. Entry only day +1..+3, waits on a new low or volume > 1.5x normal (STR-3.5b), Friday expiry ~21 days with no earnings inside (STR-1.4b, 2.2a), short ~0.15 delta below the post-news low, long 5% lower, min credit (STR-3.4b, `--min-credit-pct`), contracts for 2% max loss (STR-3.8a).
- `manage`: for each open row of data/results/agent/positions.csv: CLOSE (take profit, STR-3.6a dynamic), HOLD (no stop: backtest), and on expiry day LET EXPIRE / CLOSE (buy back if between strikes) / LET SETTLE.
**Gotchas:** strikes and credit are Black-Scholes estimates; the broker chain decides. The buy-back estimate ignores IV crush and can be far off: put the broker's price in the `mark` column before acting on HOLD/CLOSE.

### AG-14 / AG-15 / AG-16  valuation_context / gate / valuation_lines  (src/agent/valuation.py)
Point-in-time fair-value evidence from data/raw/fundamentals (AG-30): forward P/E, its 3-year percentile, P/E vs GICS-industry median, 30-day NTM EPS revision, price-target gap, analyst count. `gate()` fails a shock if P/E is above its 3-year median, > 1.2x the industry, or EPS was cut > 2% in the 30 days before (thresholds from the literature, not tuned). The agent adds these lines to its prompt only when the data exists (old prompts and cache unchanged); `run_agent.py --fv-gate` lets the gate veto a TAKE. Off by default until it passes `score_signals.py` on the hold-out.

### AG-21 / AG-22  deflated_sharpe / pbo_cscv  (src/backtest/overfit.py)
Deflated Sharpe ratio (Bailey & Lopez de Prado 2014) and probability of backtest overfitting via CSCV (Bailey et al. 2015). Input: a (periods x settings) return matrix. Report both next to every grid result.

### AG-23  scripts/score_signals.py
One bull put spread per shock from REAL option closes (event_puts.parquet), held to expiry; every filter (agent TAKE, news, IV/RV, credit/width, drop size, fair-value gate) scored in-sample and once on the hold-out (default from 2026-06-01) with bootstrap intervals. A filter counts only if it beats the rest in-sample (interval above 0) and points the same way on the hold-out.

### AG-30  fetch_fundamentals  (src/ingest/lseg_fundamentals.py, scripts/download_fundamentals.py)
Weekly I/B/E/S history per RIC since 2021 (TR.EPSMean NTM/FY1, TR.EPSNumIncEstimates, TR.PriceTargetMean, TR.RecMean) into data/raw/fundamentals/<RIC>.parquet (field, date, value). Needs LSEG Workspace. `--limit 3` prints which fields came back empty: if one is empty, change its name in FIELDS.

### AG-17  index_plan  (src/agent/trade_desk.py) + `trade_desk.py index`
The core trade: a bull put spread on XSP (S&P 500 / 10, cash-settled European, no assignment), ~21 DTE Friday expiry, short ~0.15 delta, $1 wide by default, sized by `--risk-usd`, at most `--max-open` open. Strike is a Black-Scholes estimate from VIX + 3 pts (or 1.3x realized vol, or `--iv`); the broker's chain decides. `manage` handles XSP rows using SPY closes as the proxy.

### AG-18  scripts/daily.py
One daily agent run (after 15:00 CST): live_scan (prices incl. SPY, shocks, news, verdicts) -> index_signal.py (the tested index branch; act only on ENTRY SIGNAL, after a teammate approves) -> shock plans (paper only, logged to data/results/agent/paper_trades.csv) -> manage. The old weekly XSP step (trade_desk index) was removed from the daily run because it failed strategy/preregistered_spy.md.

### AG-24  scripts/spy_benchmark.py
Same spread rules on SPY from the OPRA day files (real closes), weekly or daily entries, `--exits hold|tp`. On 2024-04..2026-10: weekly, held to expiry, mean return on risk +1.8% (hold-out +4.7%) vs +0.2% (hold-out -3.0%) for single-stock shocks. Losses cluster in Mar-Apr 2025.

### AG-28a..d  scripts/index_branch.py  (index branch backtest, SPY options 2010-2026)
Kaggle EOD SPY chains 2010-2023 plus OPRA day bars 2024-2026; fills at short bid / long ask plus fees.
AG-28a candidate grid (entry x structure x exit). AG-28b every candidate's trades. AG-28c walk-forward: each year picks the
best candidate on earlier years only (strategy/preregistered_index_branch.md). AG-28d `--stage capital --capital N`:
width W = largest whole $ with W x 100 <= capital x 10% (W >= 15 uses the 0.03-delta long leg), same entry/short/exit,
same 4 checks (strategy/preregistered_index_branch_small.md, "Capital rule"). Results in data/results/index_branch/.

### AG-29  scripts/index_signal.py
Daily check of the walk-forward rule on SPY_adj: a 1-day <= -1.5% or 2-day <= -2.5% shock in the last 1-3 trading days,
close above the post-shock low, SPY above its 200-day average; prints the broker steps (9-day vs 30-day IV guard,
short -0.10 delta, long -0.03 delta or the capital-rule width, take-profit rule). Prints NO ENTRY otherwise.

### AG-30  src/ingest/lseg_fundamentals.py
I/B/E/S history for the valuation gate (AG valuation.py): EPS mean NTM/FY1, number of estimates, price target, recommendation.

### AG-28e  scripts/fill_sensitivity.py  (+ FILL in scripts/index_branch.py)
Reruns the capital rule with better fills: FILL = share of each leg's half bid/ask spread paid (1 = sell at bid / buy at
ask, as in every test; 0 = mid, an upper bound). A sensitivity for the execution agent, not a pass/fail test.
2010-2026, per year on the account: $2,500 $4 (bid/ask) -> $13 (quarter spread) -> $19 (mid); $5,000 $22 -> $36 -> $41;
$10,000 $53 -> $67 -> $71. Results in data/results/index_branch/fill_sensitivity.csv.

### AG-31 / AG-32  scripts/extract_etf_options.py / scripts/download_etf_prices.py
AG-31 pulls the puts of 29 liquid ETFs (SPY, QQQ, IWM, DIA, sector SPDRs, GLD, TLT, ...) out of the OPRA day files
already on disk (data/raw/options/day_aggs_v1) into data/processed/options/etf/puts_YYYY-MM.csv.gz; resumable.
AG-32 downloads their daily prices from LSEG (adj + unadj) into data/raw/prices_bulk, with the ticker->RIC map etf_rics.csv.

### AG-37..40  src/agent/execution.py + scripts/order_desk.py  (order-placing agent; never sends orders)
AG-37 ladder(): spread limit prices from the mid toward the natural price (4 steps x 5 min, at most halfway, rounded
toward the mid; skips if credit after fees <= min). AG-38 take_profit_price(): buy-to-close debit at which STR-3.6a fires
today (credit x (1 - clip(time used + 0.20, 0.50, 0.75)), rounded down) for a resting order, re-placed daily.
AG-39 fills.csv + realized fill quality (0 = mid, 1 = worst, same scale as index_branch FILL) and its $/yr from
fill_sensitivity.csv. AG-40 ticket(): plain-text combo order for a teammate to approve and place. Tests: tests/test_execution.py.
