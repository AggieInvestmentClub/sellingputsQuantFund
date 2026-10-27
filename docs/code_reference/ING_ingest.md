# ING: download from LSEG (`src/ingest/`)

These functions pull raw data from LSEG Workspace and save it to `data/raw/` as parquet.
They need Workspace open and signed in, and must run inside `lseg_session()` (ING-01).
Nothing here cleans the data; that's the job of DAT (not built yet).

**Gotchas that apply to every ING function:**
- RICs are LSEG tickers: `AAPL.O` (Nasdaq), `JPM.N` (NYSE). Delisted names end in `^` + a code.
- Saved timestamps are **UTC with a timezone**, in a column named `event_time`. Strategy
  functions (STR) expect a plain date **index without timezone**, so data must be converted first.
- `data/` is gitignored: everyone downloads their own copy.

---

### ING-01  lseg_session
**File:** src/ingest/lseg_client.py
**Status:** ✅ done

**What it does:** Opens a connection to LSEG Workspace, hands it to your code, and always
closes it afterwards, even if your code crashes. Uses `LSEG_APP_KEY` from `.env`; if that's
empty, falls back to `lseg-data.config.json`.

**Inputs:** none

**Returns:** the `lseg.data` module (`ld`), with an open session

**Example:**
```python
with lseg_session():
    px = fetch_prices("AAPL.O", "2026-09-01", "2026-10-15")
```

**Uses:** —  **Used by:** every script that downloads data

**Gotchas:** Workspace must be running on your machine; this is a desktop session, not a cloud one.

---

### ING-02  fetch_prices
**File:** src/ingest/lseg_prices.py
**Status:** ✅ done

**What it does:** Downloads price bars (daily by default) for one stock between two dates.
Starts 1 day before `start` so the first day's return can be computed.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| ric | str | LSEG ticker, e.g. `AAPL.O` | — |
| start, end | str | dates, `"YYYY-MM-DD"` | — |
| interval | str | bar size | `"daily"` |

**Returns:** DataFrame with `event_time` (UTC), LSEG price columns (`TRDPRC_1` = close,
`ACVOL_UNS` = volume, ...) and `ric`. Empty DataFrame if LSEG returns nothing.

**Example:** `fetch_prices("AAPL.O", "2026-09-01", "2026-10-15")` → 32 daily rows

**Uses:** ING-01 session  **Used by:** smoke_test, depth tests, membership checks; later DAT

**Gotchas:**
- Prices are **split-adjusted** (good for returns). For the $20 price floor (STR-1.1a) you need
  **unadjusted** prices: `ld.get_history(..., adjustments="unadjusted")`, not yet wired in here.
- Going back 1 *calendar* day can land on a weekend, so the first return may still be missing
  when `start` is a Monday.

---

### ING-03  save_prices
**File:** src/ingest/lseg_prices.py
**Status:** ✅ done

**What it does:** Saves a price DataFrame to `data/raw/prices/<RIC>_<interval>.parquet`
(dots in the RIC become underscores, e.g. `AAPL_O_daily.parquet`).

**Inputs:** `ric` (str), `interval` (str), `df` (DataFrame from ING-02)

**Returns:** the file path written

**Uses:** —  **Used by:** smoke_test

---

### ING-04  _get_page
**File:** src/ingest/lseg_news.py
**Status:** ✅ done (internal helper)

**What it does:** Requests one page of up to 100 headlines. On an error it waits and retries
(5s, 10s, 20s), and gives up after 4 tries.

**Inputs:** `query` (LSEG news query), `page` (max rows), `start_iso` / `end_iso` (ISO times), `retries` (4)

**Returns:** DataFrame of headlines, or raises the last error

**Uses:** ING-01 session  **Used by:** ING-05

---

### ING-05  _fetch_window
**File:** src/ingest/lseg_news.py
**Status:** ✅ done (internal helper)

**What it does:** Gets every headline in one time window. LSEG returns the newest 100 first,
so it keeps stepping the end time back to just before the oldest headline it got, until it
reaches the window start.

**Inputs:** `query`, `start_ts` / `end_ts` (UTC Timestamps), `page` (100)

**Returns:** list of DataFrames (one per page)

**Uses:** ING-04  **Used by:** ING-06

---

### ING-06  fetch_headlines
**File:** src/ingest/lseg_news.py
**Status:** ✅ done

**What it does:** Downloads all English headlines tagged with one stock between two dates.
Splits the range into 7-day windows (ING-05), joins them, removes duplicate `storyId`s and
sorts oldest to newest.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| ric | str | LSEG ticker | — |
| start, end | str | dates | — |
| page | int | headlines per request | 100 |
| lang | str | language code | `"LEN"` (English) |
| window_days | int | size of each chunk | 7 |

**Returns:** DataFrame with `event_time` (UTC), `headline`, `storyId`, `sourceCode`,
`data_available_time`, `ric`

**Example:** AAPL, Sept 1 – Oct 15 2026 → about 7,000 headlines

**Uses:** ING-05  **Used by:** smoke_test, depth tests; later DAT news cleaning

**Gotchas:**
- **Most headlines are junk.** Only Reuters (`sourceCode == "NS:RTRS"`) has reliable times.
- `data_available_time` is set equal to `event_time` for **every** source; that's only safe
  for Reuters. Using other sources risks look-ahead bias.
- Reuters posts several versions of one story (`storyId` ending `:1`, `:2`, ...). These are
  **not** merged here; the first version should be treated as the news time.

---

### ING-07  save_news
**File:** src/ingest/lseg_news.py
**Status:** ✅ done

**What it does:** Saves headlines to `data/raw/news/<RIC>.parquet`.

**Inputs:** `ric` (str), `df` (DataFrame from ING-06)

**Returns:** the file path written

**Uses:** —  **Used by:** smoke_test

---

### ING-08  fetch_earnings_dates
**File:** src/ingest/lseg_earnings.py
**Status:** ✅ done

**What it does:** Downloads past earnings report dates for a list of stocks, 50 at a time.
Adds `report_date` (the day) and `after_close` (reported at or after 4pm).

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| rics | list[str] | LSEG tickers | — |
| start, end | str | date range | `"2017-01-01"`, `"2027-01-31"` |
| batch | int | stocks per request | 50 |

**Returns:** DataFrame with `ric`, `report_time`, `report_date`, `after_close`. A failed batch
is printed and skipped.

**Example:** AAPL, 2018 on → about 4 rows per year

**Uses:** ING-01 session  **Used by:** checks; later STR-2.2a earnings_clear

**Gotchas:**
- The **next upcoming** earnings date doesn't come back, so live trading needs another source.
- `after_close` assumes the time is US Eastern. If LSEG returns UTC or date-only values,
  the flag is wrong. Not yet verified.

---

### ING-09  save_earnings
**File:** src/ingest/lseg_earnings.py
**Status:** ✅ done

**What it does:** Saves earnings dates to `data/raw/earnings/earnings_dates.parquet`.

**Inputs:** `df` (DataFrame from ING-08)  **Returns:** nothing

---

### ING-10  fetch_members
**File:** src/ingest/lseg_universe.py
**Status:** ✅ done

**What it does:** Lists the RICs that were in the S&P 500 on one date.

**Inputs:** `as_of` (str date), `index` (`".SPX"`)

**Returns:** sorted list of RICs (about 500)

**Example:** `fetch_members("2018-02-01")` → ~505 RICs, including ones since delisted

**Uses:** ING-01 session  **Used by:** ING-12, membership scripts

**Gotchas:** Returns nothing before **April 2015**, so the backtest can't start earlier.

---

### ING-11  save_membership
**File:** src/ingest/lseg_universe.py
**Status:** ✅ done

**What it does:** Saves membership snapshots to `data/raw/universe/sp500_membership.parquet`.

**Inputs:** `df` (columns `date`, `ric`)  **Returns:** nothing

---

### ING-12  build_membership
**File:** src/ingest/lseg_universe.py
**Status:** ✅ done

**What it does:** Calls ING-10 for every month-end between `start` and `end`, retrying failed
months, and saves the result (ING-11). Prints progress each December.

**Inputs:** `start` (`"2000-01-31"`), `end` (`"2026-09-30"`), `index` (`".SPX"`), `retries` (3)

**Returns:** DataFrame with one row per (month-end date, RIC)

**Uses:** ING-10, ING-11  **Used by:** scripts/build_membership.py

**Gotchas:** Use `start="2015-04-30"`; earlier months come back empty. Some months come back
short, and `scripts/check_membership.py` re-downloads those.

---

### ING-13  members_before
**File:** src/ingest/lseg_universe.py
**Status:** ✅ done

**What it does:** Returns the S&P 500 members from the latest month-end snapshot strictly
before a shock date. This is the point-in-time universe check for STR-1.1a.

**Inputs:** `snapshots` (DataFrame from ING-12), `shock_date`

**Returns:** set of RICs

**Example:** shock on 2024-03-20 → members as of 2024-02-29

**Uses:** —  **Used by:** later EVT / STR-1.1a

**Gotchas:** Lags by up to a month: a stock added mid-month is missed until the next
month-end. ING-14's join / leave dates could make it exact.

---

### ING-14  fetch_index_changes
**File:** src/ingest/lseg_universe.py
**Status:** ✅ done

**What it does:** Downloads every S&P 500 join and leave event with its date.

**Inputs:** `index` (`".SPX"`), `start` (`"2000-01-01"`), `end` (`"2026-10-22"`)

**Returns:** DataFrame with `ric`, `name`, `change` (e.g. `"Leaver"` for removals), `date`

**Uses:** ING-01 session  **Used by:** scripts/check_universe.py

**Gotchas:** The default `end` date is hard-coded; pass a newer one.

---

### ING-15  save_changes
**File:** src/ingest/lseg_universe.py
**Status:** ✅ done

**What it does:** Saves join / leave events to `data/raw/universe/sp500_changes.parquet`.

**Inputs:** `df` (DataFrame from ING-14)  **Returns:** nothing

---

### ING-16  fetch_company_ids
**File:** src/ingest/security_master.py
**Status:** ✅ done

**What it does:** For each RIC, gets LSEG's permanent company ID (`TR.OrganizationID`), company
name, ticker and CUSIP, 100 RICs per request.

**Inputs:** `rics` (list of str), `batch` (100)

**Returns:** DataFrame with `ric`, `company_id`, `name`, `ticker`, `cusip`

**Example:** GOOG.OQ and GOOGL.OQ → both `5030853586` (one company, two share classes)

**Uses:** ING-01 session  **Used by:** ING-18

**Gotchas:** RICs LSEG can't resolve get a placeholder ID `RIC:<ric>`, so they're never merged
with another company by mistake.

---

### ING-17  fetch_ticker_history
**File:** src/ingest/security_master.py
**Status:** ✅ done

**What it does:** Massive's ticker history for one security, looked up by **CUSIP**
(`/vX/reference/tickers/{cusip}/events`).

**Inputs:** `cusip` (str)

**Returns:** list of `(ticker, valid_from)`, oldest first; `[]` if Massive has none

**Example:** Meta's CUSIP 30303M102 → `[("FB", 2012-05-18), ("META", 2022-06-09)]`

**Uses:** Massive REST (`POLYGON_API_KEY`)  **Used by:** ING-18

**Gotchas:** Looked up by CUSIP, not ticker, because delisted tickers get reused by other
companies. Delisted companies (e.g. SVB) return nothing; ING-18 falls back to the RIC.

---

### ING-18  build_security_master
**File:** src/ingest/security_master.py
**Status:** ✅ done

**What it does:** For every RIC ever in the S&P 500 membership table, records its company ID,
membership span, LSEG option root and the Massive ticker valid on each date.

**Inputs:** `membership` (DataFrame `date`, `ric` from ING-12)

**Returns / saves:**
- `data/raw/universe/security_master.parquet`: `ric`, `first_member`, `last_member`,
  `company_id`, `name`, `ticker`, `cusip`, `delisted`, `lseg_option_root`
- `data/raw/universe/massive_tickers.parquet`: `ric`, `massive_ticker`, `valid_from`, `source`

`source` is `"events"` (Massive history), `"ric_fallback"` (ticker from the RIC) or
`"events_overridden"` (Massive history disagreed with the current ticker, so the RIC was used).

**Uses:** ING-16, ING-17, DAT-01, DAT-02  **Used by:** scripts/build_security_master.py; later EVT, option downloads

**Gotchas (tested Oct 2026):**
- **LSEG uses today's RIC and option root for all history** (`META.OQ`, `METAR...^F22` work for
  2022 when the ticker was FB). **Massive uses the ticker at the time** (FB before 2022-06-09).
  So only Massive lookups need DAT-03.
- Massive's ticker history can be wrong (GOOGL's said `GOOG` from 2015). For live stocks, if the
  latest Massive ticker isn't the current one, the history is ignored (`events_overridden`).

---

### ING-19  fetch_daily_history
**File:** src/ingest/lseg_prices.py
**Status:** ✅ done

**What it does:** Daily bars for one RIC with a plain date index (no timezone), the format STR
and EVT code expect.

**Inputs:** `ric`, `start`, `end`, `adjustments` (None = split-adjusted, `"unadjusted"` = as traded)

**Returns:** DataFrame indexed by `date`: `TRDPRC_1` (close), `OPEN_PRC`, `HIGH_1`, `LOW_1`,
`ACVOL_UNS` (volume), `TRNOVR_UNS` ($ volume), `VWAP`, `BID`, `ASK`, ...

**Example:** AAPL 2020-08-28: adjusted 124.81, unadjusted 499.23 (4-for-1 split on 08-31)

**Uses:** ING-01 session  **Used by:** scripts/overnight.py, EVT-04

**Gotchas:**
- LSEG's adjusted prices are **split-only, not dividend-adjusted**, so returns exclude dividends
  (small). Split dates can be found where adjusted ÷ unadjusted jumps.
- One request covers 2014–2026 (3,223 rows tested), so there's no need to split the range.

---

### ING-20  fetch_market_cap
**File:** src/ingest/lseg_reference.py
**Status:** ✅ done

**What it does:** Daily company market cap for one RIC (`TR.CompanyMarketCap`).

**Inputs:** `ric`, `start` ("2014-01-01"), `end` (today)

**Returns:** Series `market_cap` with a date index

**Uses:** ING-01 session  **Used by:** scripts/overnight.py; later STR-1.1a, STR-2.4a

**Gotchas:** Company-level, so GOOG and GOOGL show the same number. Don't add them together.

---

### ING-21  fetch_dividends
**File:** src/ingest/lseg_reference.py
**Status:** ✅ done

**What it does:** Cash dividends for many RICs, 50 per request.

**Inputs:** `rics`, `start`, `end`, `batch` (50)

**Returns:** DataFrame `ric`, `ex_date`, `amount` (gross per share, unadjusted), `div_type`

**Uses:** ING-01 session  **Used by:** scripts/overnight.py; OPT-02 / OPT-04 dividend inputs

---

### ING-22  fetch_treasury_yields
**File:** src/ingest/lseg_reference.py
**Status:** ✅ done

**What it does:** Daily 1/3/6-month Treasury bill yields, the risk-free rate for Black-Scholes.

**Inputs:** `rics` (`US1MT=RR`, `US3MT=RR`, `US6MT=RR`), `start`, `end`

**Returns:** DataFrame, one column per RIC, **already in decimals** (`B_YLD_1` / 100)

**Uses:** ING-01 session  **Used by:** scripts/overnight.py; OPT-04 `r` input
