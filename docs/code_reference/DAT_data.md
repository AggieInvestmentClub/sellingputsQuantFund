# DAT: data lookups and cleaning (`src/data/`)

Pure functions that work on downloaded tables. Nothing here calls an API.

---

### DAT-01  lseg_option_root
**File:** src/data/security_master.py
**Status:** ✅ done

**What it does:** The root LSEG uses in option RICs for a stock RIC.

**Inputs:** `ric` (str)  **Returns:** str

**Example:** `"META.OQ"` → `"META"`, `"BRKb.N"` → `"BRKB"`, `"SIVB.OQ^C23"` → `"SIVB"`

**Uses:** —  **Used by:** ING-18, later option downloads

**Gotchas:** LSEG uses today's root for all history (META, not FB, for 2022 options).

---

### DAT-02  massive_ticker_from_ric
**File:** src/data/security_master.py
**Status:** ✅ done

**What it does:** Fallback Massive ticker built from a RIC, used when Massive has no ticker history.

**Inputs:** `ric` (str)  **Returns:** str

**Example:** `"META.OQ"` → `"META"`, `"BRKb.N"` → `"BRK.B"`, `"SIVB.OQ^C23"` → `"SIVB"`

**Uses:** —  **Used by:** ING-18

**Gotchas:** LSEG writes share classes as a lowercase letter (`BRKb`); Massive uses a dot (`BRK.B`).

---

### DAT-03  massive_ticker_on
**File:** src/data/security_master.py
**Status:** ✅ done

**What it does:** The Massive ticker for a RIC on a given date. Massive lists option contracts
under the ticker valid at the time, so this is needed for every Massive contract lookup.

**Inputs:** `massive_tickers` (table from ING-18), `ric`, `date`

**Returns:** ticker (str), or None if the RIC is unknown. Before the first recorded change, the
earliest known ticker.

**Example:** `META.OQ` on 2022-05-13 → `"FB"`; on 2022-06-15 → `"META"`

**Uses:** ING-18 output  **Used by:** later option-contract downloads

---

### DAT-04  companies
**File:** src/data/security_master.py
**Status:** ✅ done

**What it does:** Groups RICs by company, so share classes count once (events, positions,
statistics).

**Inputs:** `rics` (iterable), `master` (security master from ING-18)

**Returns:** `{company_id: [rics]}`

**Example:** `["GOOG.OQ", "GOOGL.OQ", "AAPL.OQ"]` →
`{"5030853586": ["GOOG.OQ", "GOOGL.OQ"], "4295905573": ["AAPL.OQ"]}`

**Uses:** ING-18 output  **Used by:** later EVT (one event per company), STR-5b (one position per company)

**Gotchas:** Which class to *trade* isn't decided here. Plan: the class with higher options open
interest on the event date (point-in-time). RICs missing from the master become their own group.

---

### DAT-05  price_issues
**File:** src/data/validate.py
**Status:** ✅ done

**What it does:** Checks one stock's daily price table for problems that would distort returns:
missing or non-positive closes, duplicate or unsorted dates, long gaps, huge one-day moves and
long runs of the same close (stale data).

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| px | DataFrame | ING-19 output (date index) | |
| close_col | str | close column | "TRDPRC_1" |
| max_gap_days | int | weekdays missing between two rows before it counts as a gap | 7 |
| max_abs_ret | float | daily return (decimal) that counts as a big move | 0.5 |
| stale_days | int | days in a row with the same close that count as stale | 5 |

**Returns:** dict with rows, first, last, nan_close, nonpositive, duplicate_dates, unsorted,
max_gap_days, big_moves, big_move_dates (first 5), longest_stale and `problems` (text, `""` if
clean). An empty table returns `problems = "empty"`.

**Example:** closes 103 → 210 on 2024-01-09 → `big_moves = 1`, `big_move_dates = "2024-01-09"`,
`problems = "1 daily moves over 50%"`

**Uses:** ING-19 output  **Used by:** scripts/validate_data.py

**Gotchas:** Holidays count as 1 missing weekday, so small gaps are normal. A big move in the
*adjusted* file is either a real crash or a split LSEG didn't adjust: compare with DAT-07.

---

### DAT-06  membership_issues
**File:** src/data/validate.py
**Status:** ✅ done

**What it does:** Checks the month-end S&P 500 membership table for blank or duplicate RICs,
missing months and odd sizes, and, given the join / leave events, that every RIC that drops out
has a matching event and every event matches the snapshots around it.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| membership | DataFrame | date, ric (ING-12) | |
| changes | DataFrame | ric, change ("Joiner"/"Leaver"), date (ING-14) | None |
| min_size | int | smallest normal snapshot | 450 |
| max_size | int | largest normal snapshot | 510 |

**Returns:** DataFrame (date, kind, ric, detail), one row per problem, empty if clean. Kinds:
`blank_ric`, `duplicate_ric`, `missing_month`, `size`; with changes: `removed_without_leaver`,
`leaver_still_member`, `leaver_not_member` (and `added_without_joiner`, `joiner_already_member`,
`joiner_not_member` when the changes include joiners).

**Example:** A.N listed on 2024-02-29, gone on 2024-03-31, no Leaver event →
`(2024-03-31, removed_without_leaver, A.N, between 2024-02-29 and 2024-03-31)`

**Uses:** ING-12, ING-14 output  **Used by:** scripts/validate_data.py

**Gotchas:** LSEG's change list (ING-14) only has leavers, so joins can't be checked yet.
`removed_without_leaver` is usually a RIC change, not a real exit (exchange move like BKR.N →
BKR.OQ, or a rename like ARNC → HWM). `leaver_not_member` is usually a stock that joined and left
within one month (spin-offs), which month-end snapshots can't see, or a leaver whose RIC suffix
differs from the snapshot's.

---

### DAT-07  split_factors
**File:** src/data/validate.py
**Status:** ✅ done

**What it does:** Finds price adjustments by comparing adjusted and unadjusted closes. LSEG
adjusts for splits and capital changes such as spin-offs (not regular dividends), so
unadjusted ÷ adjusted stays constant between those events and jumps on their ex-date.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| adj_close | Series | adjusted close, date index (ING-19, default) | |
| unadj_close | Series | unadjusted close, date index (ING-19, "unadjusted") | |
| tol | float | relative change in the ratio that counts as an adjustment | 0.005 |

**Returns:** DataFrame (date, ratio_before, ratio_after, split), one row per split; empty if none.
`split` = ratio_before ÷ ratio_after: 4.0 = 4-for-1, 0.1 = 1-for-10 reverse split.

**Example:** AAPL: unadjusted 506.09, adjusted 126.52 on 2020-08-28, then equal from
2020-08-31 → `(2020-08-31, 4.0, 1.0, 4.0)`

**Uses:** ING-19 output  **Used by:** scripts/validate_data.py; later the price floor and strike
checks (they need unadjusted prices on the trade date)

**Gotchas:** The date is the first day at the new ratio (the ex-date). A round `split` (2, 3,
4, 0.1) is a split; a non-round one is usually a spin-off (A.N 2014-11-03 Keysight 1.40, BAX
2015-07-01 Baxalta 1.84, CAG 2016-11-10 Lamb Weston 1.29) or a special distribution. That
adjustment is what we want for returns: unadjusted, a spin-off day looks like a -30% shock.
Ratios that jump back and forth (AIV.N late 2020) point to a bad adjusted series.
