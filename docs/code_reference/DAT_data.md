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
