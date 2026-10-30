# EVT: event study (`src/events/`)

Testing order step 1: find every non-earnings shock in the S&P 500 and measure what the stock did
afterwards. **Price-only**: news categories (2.3) and the LLM filter (2.4) come later.

Same rules as STR-2.1 (beta, abnormal return, shock) and STR-2.2 (earnings buffer), vectorized
over all days. `tests/test_events.py` checks that EVT-01 gives exactly STR-2.1a's beta.

Run it: `python -u scripts/overnight.py` (downloads first, then the study). Output goes to
`data/results/event_study/`: `events_raw.csv`, `events.csv` (deduped), `summary.txt`, `summary.csv`.

---

### EVT-01  rolling_beta
**File:** src/events/build_events.py
**Status:** ✅ done

**What it does:** Beta for every day, from the 250 days ending 5 days before it. The same
calculation as STR-2.1a `estimate_beta`, but for all days at once.

**Inputs:** `stock_ret`, `spy_ret` (date-indexed returns), `window` (250), `gap` (5), `min_obs` (200)

**Returns:** Series of betas aligned to `stock_ret`; NaN where there's too little history

**Uses:** —  **Used by:** EVT-04

---

### EVT-02  member_mask
**File:** src/events/build_events.py
**Status:** ✅ done

**What it does:** True on the days a RIC was in the S&P 500, using the latest month-end snapshot
before each day (same rule as ING-13 `members_before`).

**Inputs:** `dates`, `ric`, `membership` (ING-12 table)

**Returns:** Series of True / False

**Example:** member in the 2016-01-31 snapshot → True for every day of February 2016

**Uses:** ING-12 data  **Used by:** EVT-04

**Gotchas:** Lags up to a month: a stock added mid-month counts from the next month.

---

### EVT-03  near_earnings
**File:** src/events/build_events.py
**Status:** ✅ done

**What it does:** True for days within ±3 weekdays of any earnings date (STR-2.2 shock buffer).

**Inputs:** `dates`, `earnings_dates`, `buffer_days` (3)

**Returns:** Series of True / False

**Example:** earnings Monday 2016-03-07 → the Friday before and Thursday 03-10 are near; Friday 03-11 isn't

**Uses:** ING-08 data  **Used by:** EVT-04

**Gotchas:** The ±3 days also covers the unverified after-close flag in ING-08.

---

### EVT-04  find_shocks
**File:** src/events/build_events.py
**Status:** ✅ done

**What it does:** Every shock day for one stock, with its abnormal return and what happened over
the next 1, 5, 10 and 20 trading days.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| ric | str | stock | — |
| close | Series | split-adjusted close, date index (ING-19) | — |
| spy_ret | Series | SPY daily returns | — |
| membership | DataFrame | ING-12 table | — |
| earnings_dates | list | ING-08 report dates | — |
| threshold | float | abnormal return cutoff | -0.05 |
| max_spy_drop | float | skip if SPY fell more | -0.02 |

**Returns:** one row per shock: `ric`, `date`, `ret`, `spy_ret`, `beta`, `abnormal`,
`car_1/5/10/20` (sum of abnormal returns on days +1..+h, using the pre-shock beta),
`ret_1/5/10/20` (raw compounded return), `earnings_known`. Always has these columns, even with no shocks.

**Example:** planted -12% day followed by five +1% days with SPY flat → `car_5` = +5%

**Uses:** EVT-01, EVT-02, EVT-03  **Used by:** scripts/overnight.py

**Gotchas:**
- `car_h` is NaN when the stock has fewer than h days of data afterwards (e.g. delisted). Those
  events drop out of that horizon, which slightly favors survivors.
- `earnings_known` = False means LSEG returned no earnings dates for that stock, so the earnings
  filter couldn't be applied. The summary reports how many.

---

### EVT-05  dedupe_events
**File:** src/events/build_events.py
**Status:** ✅ done

**What it does:** One event per company. If several RICs of one company shock on the **same day**
(share classes like GOOG/GOOGL, or one stock under two RICs), keeps the RIC with the longest S&P 500
membership (ties alphabetical). Then drops shocks within 20 trading days after the company's
previous kept event, so overlapping windows aren't counted twice.

Events on **different days** are kept even if the RIC differs. The security master found that
some "same company" groups aren't share classes but one stock under a different RIC over time:
exchange moves (BKR.N → BKR.OQ, CPB, HST, KMB) and relistings (CHK → EXE). Keeping only one RIC
per company would throw away the other period's events.

**Inputs:** `events` (EVT-04 rows), `master` (ING-18), `cooldown` (20)

**Returns:** deduped events with `company_id`

**Example:** GOOG and GOOGL both shocked on 2020-03-02 → one event; another GOOG shock on 03-10 → dropped

**Uses:** ING-18 output  **Used by:** scripts/overnight.py

---

### EVT-06  summarize
**File:** src/events/build_events.py
**Status:** ✅ done

**What it does:** Average abnormal return after shocks, per shock size (-5% / -10% / -15%,
cumulative) and horizon.

**Inputs:** `events` (EVT-05 output), `thresholds`

**Returns:** DataFrame `threshold`, `horizon`, `n`, `mean`, `median`, `t_stat`, `pct_positive`

**Uses:** —  **Used by:** scripts/overnight.py

**Gotchas:** `t_stat` assumes events are independent. Shocks cluster in time (e.g. March 2020),
so treat it as optimistic until the Monte Carlo block bootstrap.

---

### EVT-07  entry_returns
**File:** src/events/analysis.py
**Status:** ✅ done

**What it does:** Returns from a realistic entry. EVT-04's `car_h` enters at the shock-day close;
the spec (3.5) enters on day +1 at the earliest, once volume settles. This re-measures each event
entering at the close of day +1 and day +3, and adds the worst close during the hold, which is
what decides whether a put spread loses.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| events | DataFrame | one stock's events with `date`, `beta` (EVT-04 / EVT-05) | |
| close | Series | split-adjusted close, date index (ING-19) | |
| spy_ret | Series | SPY daily returns | |
| lags | tuple | entry days after the shock | (1, 3) |
| holds | tuple | holding days after entry | (5, 10, 20) |

**Returns:** `events` plus, for each lag L and hold H: `car_eL_H` (abnormal, pre-shock beta),
`ret_eL_H` (raw, compounded), `low_eL_H` (worst close in the hold ÷ entry close − 1). NaN if the
stock has no data that far.

**Example:** shock close 90, entry close (+1) 92, next closes 95, 93, 99 →
`ret_e1_3 = 99/92 − 1 = 0.076`, `low_e1_3 = 93/92 − 1 = 0.011`

**Uses:** EVT-04 / EVT-05 output  **Used by:** scripts/analyze_events.py

**Gotchas:** With lag 0 it gives exactly EVT-04's `car_h` / `ret_h` (tested). `low` uses closes
only; an intraday low can be worse.

---

### EVT-08  robust_summary
**File:** src/events/analysis.py
**Status:** ✅ done

**What it does:** Mean, median and t-stat per column, plus a t-stat that averages same-day events
first (so a crash day with 40 shocks counts once) and the share of events that lost `tail` or more.

**Inputs:** `events` (with `date`), `cols` (list of columns), `tail` (−0.10)

**Returns:** DataFrame `column`, `n`, `n_dates`, `mean`, `median`, `t_stat`, `t_by_date`,
`pct_positive`, `pct_tail`

**Example:** four events of −0.20 on 2020-03-09 and +0.05, +0.03 on other days → `n = 6`,
`n_dates = 3`, `mean = −0.12`, `pct_tail = 0.67`, `t_by_date` from the three daily means

**Uses:** —  **Used by:** scripts/analyze_events.py

**Gotchas:** `t_by_date` fixes same-day clustering only. Shocks a few days apart in the same
selloff are still treated as independent; the block bootstrap planned for the backtest handles that.
