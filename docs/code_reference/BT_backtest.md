# BT: backtest (`src/backtest/`)

Simulates one bull put spread per shock with the STR rules, crosses trade parameters with event
filters, and reports per quarter and walk-forward. Runner: `scripts/run_backtest.py`,
config: `strategy/backtest_grid.yaml`.

**How the combinations are tested.** Trade parameters (delta, width, DTE, entry trigger, exits)
change the trade, so each combination is simulated once over all events. Event filters (LLM verdicts,
category, shock size, market cap) only choose which events count, so they are applied to the finished
trades: every (combination × filter) pair is scored without re-simulating.

**Walk-forward.** Quarters are folds. For each test quarter, the candidate with the best score on
the previous `train_quarters` (rolling, or every earlier quarter if `expanding`) is chosen and scored on
the test quarter only. Training uses only trades that exited before the test quarter starts (purge).
The stitched test quarters are the out-of-sample result; compare them with always-baseline.

**Two pricers.** `panel` = real OPRA day-aggregate closes (EVT-12 output), `proxy` = Black-Scholes
stand-in. Neither has bid/ask quotes, so fills are close −/+ a modeled half spread (`half_spread`, min $0.025).

---

### BT-01  load_stock
**File:** src/backtest/market.py
**Status:** ✅ done

**What it does:** Daily bars for one stock: unadjusted close / low / high / volume (strikes are as traded)
and the split-adjusted daily return (for realized vol).

**Inputs:** `ric`  **Returns:** DataFrame indexed by date (close, low, high, volume, ret), or None if no price file.

**Uses:** ING-19 files (`<RIC>_unadj.parquet`, `<RIC>_adj.parquet`)  **Used by:** scripts/run_backtest.py

**Gotchas:** A split during a trade makes the unadjusted close jump; real strikes would be adjusted too.
Not handled (rare in a 2-5 week hold).

---

### BT-02  start
**File:** src/backtest/market.py (`ProxyPricer.start`)
**Status:** ✅ done

**What it does:** Sets up the proxy pricer for one event: ATM IV = `iv_mult` × the stock's 60-day realized
vol before the shock (floor `iv_floor`), skew `iv(K) = ATM × (1 + skew × ln(S/K))`, Friday expiries,
strike steps by price level.

**Inputs:** `event_id`, `stock` (BT-01), `shock_date`  **Returns:** None (state kept on the pricer)

**Uses:** OPT-07, OPT-09  **Used by:** BT-04 via scripts/run_backtest.py

**Gotchas:** For checking the code only. IV is constant through the trade (no post-shock IV spike or
crush), so the IV/RV entry rule and the vol-premium part of the P&L are not realistic.

---

### BT-03  start
**File:** src/backtest/market.py (`PanelPricer.start`)
**Status:** ✅ done

**What it does:** Sets up the real-panel pricer for one event: its puts from `data/processed/options/event_puts.parquet`.
ATM IV = median IV of 14-60 DTE puts with |delta| 0.35-0.65. A leg with no print on a day is re-priced
with Black-Scholes at its last implied vol.

**Inputs:** `event_id`, `stock`, `shock_date`  **Returns:** None

**Uses:** EVT-09..12 output, OPT-07, OPT-09  **Used by:** BT-04

**Gotchas:** Day aggregates have no open interest or quotes: STR-1.4a's OI and bid-ask checks can't run;
legs only need volume > 0 on the entry day.

---

### BT-04  simulate_event
**File:** src/backtest/engine.py
**Status:** ✅ done

**What it does:** The trade for one shock under one parameter set. Waits day +1 .. +max_wait for the entry
trigger (STR-3.5b: IV/RV, volume cooldown, no new low), picks the expiry (STR-1.4b), short (STR-3.2a,
below the post-news low) and long (STR-3.3a) strikes, requires the credit (STR-3.4b) and no earnings in
the window (STR-2.2a), then marks it each close until take-profit (STR-3.6a), stop (STR-3.7a) or expiry
(settled at intrinsic). If the stock closes between the strikes on expiry day, the spread is bought
back at that close instead (`close_between_at_expiry`, default on): otherwise the short put is
assigned, the long expires worthless, and you hold 100 shares per contract over the weekend.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| event | dict / row | event_id, ric, date | |
| stock | DataFrame | BT-01 | |
| pricer | ProxyPricer / PanelPricer | `start()` already called | |
| earnings_dates | list | report dates for the stock | |
| params | dict | overrides of `BASELINE` in engine.py | |

**Returns:** dict with `status` ("traded", or why not: "no entry signal", "earnings in window",
"price below floor", ...). Traded: entry / exit dates, entry_lag, strikes, width, credit, exit_cost, pnl,
max_loss, `ror` = pnl / max_loss (per share, after $0.65 per contract per leg), exit_reason, days_held.

**Example:** stock 100 → 90 on the shock, recovers to 103 → enters day +1, short 85 / long 80,
dynamic take-profit a few days later, ror ≈ +0.1.

**Uses:** STR-1.4b, 2.2a, 3.2a, 3.3a, 3.4b, 3.5a, 3.5b, 3.6a, 3.7a  **Used by:** scripts/run_backtest.py

**Gotchas:** Decisions and fills both at the close (end-of-day data). If the credit is too small on a day,
it keeps waiting (up to max_wait). Fill = sell short at modeled bid, buy long at modeled ask, both ways.
The expiry close pays at least intrinsic (short strike − close) plus the modeled spread, and 4 legs of fees
(settling at expiry pays 2). Below the long strike both legs exercise, so the loss is the width: no close needed.

**Random-entry control:** `scripts/make_random_events.py` picks random no-shock dates in the same stocks;
build their option panel with `build_option_panel.py --events ... --out data/processed/options_random` and
run `run_backtest.py --events ... --panel ... --config strategy/backtest_random.yaml`.

---

### BT-05  expand_grid
**File:** src/backtest/grid.py
**Status:** ✅ done

**What it does:** Every combination of the chosen parameter dimensions (full product); others stay at baseline.

**Inputs:** `space` {name: [values]}, `baseline`, `dims` (None = all)  **Returns:** list of dicts

**Example:** dims short_delta (3) × width_pct (4) → 12 dicts

**Used by:** scripts/run_backtest.py `--mode grid`

**Gotchas:** The full spec grid is millions of combinations; cross only 2-3 dimensions (spec 7.5).

---

### BT-06  one_at_a_time
**File:** src/backtest/grid.py
**Status:** ✅ done

**What it does:** The baseline plus one variant per non-baseline value of each parameter.

**Returns:** list of (label, params); labels "baseline", "short_delta=0.15", ...

**Used by:** scripts/run_backtest.py `--mode oat`

---

### BT-07  event_mask
**File:** src/backtest/grid.py
**Status:** ✅ done

**What it does:** Which events one event filter keeps: LLM event set (`all`, `has_news`, `nj_both`,
`nj_both_samecat`, `nj_opus`, `nj_gpt4o`, `not_j_both`, `j_both`), Opus category, minimum overreaction ratio
(market value lost ÷ Opus mid impact, STR-2.4a), shock size and market cap floor.

**Inputs:** `events` (table built by run_backtest.py), `event_set="all"`, `category="any"`, `min_ratio=None`,
`max_shock=-0.05`, `mcap_floor=5e9`  **Returns:** boolean Series

**Gotchas:** LLM labels exist only for 2024-04-08 .. 2026-09-15; earlier events have no verdict, so every
set except `all` is empty there.

---

### BT-08  event_filters
**File:** src/backtest/grid.py
**Status:** ✅ done

**What it does:** Every event filter from the `event_filters` block of the config (product of its lists).

**Returns:** list of dicts for BT-07

---

### BT-09  trade_metrics
**File:** src/backtest/grid.py
**Status:** ✅ done

**What it does:** Per-trade results: n, n_dates, win_rate, mean / median / sum ror, worst5_ror (5th
percentile), max_loss_share (ror ≤ −0.9), t_by_date, avg_days.

**Gotchas:** `t_by_date` averages the trades entered on the same day first, so a day with 20 macro shocks
counts once (same idea as EVT-08). Use it, not the raw t-stat.

---

### BT-10  portfolio_curve
**File:** src/backtest/grid.py
**Status:** ✅ done

**What it does:** Runs trades as one book: each risks `risk_pct` of starting capital (no compounding),
STR-5b blocks a second position in a ticker or a sector above 30% of capital-at-risk, P&L booked on exit.

**Returns:** (taken trades with dollar_pnl, daily equity, stats: total_return, max_drawdown, sharpe, n_taken, n_blocked)

**Uses:** STR-5b

---

### BT-11  quarterly_table
**File:** src/backtest/walk_forward.py
**Status:** ✅ done

**What it does:** BT-09 metrics per group and entry quarter, e.g. every parameter variant × quarter on
all events (`quarterly_params.csv`) or every event filter × quarter at baseline (`quarterly_filters.csv`).

---

### BT-12  quarter_folds
**File:** src/backtest/walk_forward.py
**Status:** ✅ done

**What it does:** Walk-forward folds: (train quarters, test quarter), rolling `train` quarters or expanding.

**Example:** 2024Q1..2024Q4, train 2 → ([Q1, Q2], Q3), ([Q2, Q3], Q4)

---

### BT-13  walk_forward
**File:** src/backtest/walk_forward.py
**Status:** ✅ done

**What it does:** For each fold, picks the candidate (combination | filter) with the best `score` in training
(at least `min_trades`), then records its test-quarter metrics and the baseline's.

**Returns:** (per-fold table, stitched out-of-sample trades)

**Gotchas:** Purge: training only uses trades that exited before the test quarter. The more candidates,
the more the training pick is luck: compare the stitched out-of-sample mean with always-baseline, and
watch how often the pick changes.
