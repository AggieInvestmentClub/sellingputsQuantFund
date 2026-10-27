# OPT: option math (`src/options/`)

Calculations on option data that no provider gives us directly. Nothing here downloads data.

**Why this exists:** LSEG's historical `DELTA` field is empty before 2022 (tested on COF:
0% filled 2018–2021, 100% from 2022). The strategy picks strikes by delta, so for
2018–2021 we compute delta from LSEG's `IMP_VOLT`, which *is* available.

**Validation (Oct 2026):** compared to LSEG's own `DELTA` where both exist (2022+):
- **Top 10 S&P stocks by market cap**, June expiries 2022–2025, puts at 100% / 95% / 90% of
  the stock price, 35 days before expiry, days with no bid excluded: **2,364 option-days**.
  - mean |error| **0.0036**, median 0.0020, 95th percentile 0.0116, max 0.093
  - **93.5% within 0.01, 98.8% within 0.02**
  - in our trading range (|delta| 0.10–0.30): mean 0.0037, 95th percentile 0.0102
  - every stock's mean error was under 0.005 (NVDA, AAPL, MSFT, GOOG, GOOGL, AMZN, META, AVGO, TSLA, LLY)
- The few large misses are an **LSEG data timing problem**, not a formula problem. On some
  fast-moving days LSEG's `DELTA` was calculated from an intraday stock price instead of
  the close. Backing out the price LSEG used: TSLA 2022-05-16 → $762.2 for all three strikes
  (close $724.37, opened $768); MSFT 2023-05-25 → $320.59 (close $325.92); NVDA 2022-06-08 →
  $191.49 (close $186.48). On a normal day (AAPL 2022-05-18) it matches the close ($140.85 vs
  $140.82). **So compute delta ourselves for every year** and use LSEG's `DELTA` only as a
  cross-check. LSEG's `IMP_VOLT` may have the same timing issue on those days (not yet checked).
- Earlier spot check: COF $105/$115 and AAPL $130/$140/$145 puts in 2022, mean error ~0.001.

**Gotchas that apply to every OPT function:**
- **Decimals only:** IV 0.25 = 25%, rate 0.0191 = 1.91%. LSEG's `IMP_VOLT` and Treasury
  `B_YLD_1` are in **percent**, so divide by 100 first.
- **Unadjusted prices:** option strikes are as-traded, so the stock price must be too.
- **Stale IV:** on days with no bid, LSEG repeats the last IV (e.g. 52.32 five days in a
  row). Don't compute delta from those days; skip or re-check the option.
- Tests: `tests/test_greeks.py`.

---

### OPT-01  years_to_expiry
**File:** src/options/greeks.py
**Status:** ✅ done

**What it does:** Time from a date to expiry in years, counting calendar days / 365.

**Inputs:** `as_of` (date), `expiry` (date)

**Returns:** float years; 0 if already expired

**Example:** 2022-05-18 → 2022-06-17 = 30 / 365 = 0.0822

**Uses:** —  **Used by:** OPT-04

---

### OPT-02  pv_dividends
**File:** src/options/greeks.py
**Status:** ✅ done

**What it does:** Present value of the cash dividends that go ex after `as_of` and on or before
expiry. These are subtracted from the stock price, because the stock drops by the dividend
on the ex-date.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| as_of, expiry | date | window | — |
| dividends | list of (ex_date, amount) | per-share cash dividends (LSEG `TR.DivExDate` + `TR.DivUnadjustedGross`, or Massive `/v3/reference/dividends`) | — |
| r | float | risk-free rate, decimal | — |

**Returns:** float $ per share

**Example:** COF $0.60 going ex 2022-05-12, as_of 2022-05-01, r = 0 → 0.60. As_of on the
ex-date itself → 0 (already paid).

**Uses:** —  **Used by:** OPT-04

---

### OPT-03  bs_delta
**File:** src/options/greeks.py
**Status:** ✅ done

**What it does:** Black-Scholes delta of a European put or call.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| S | float | stock price (dividend-adjusted if using discrete dividends) | — |
| K | float | strike | — |
| T | float | years to expiry | — |
| r | float | risk-free rate, decimal | — |
| vol | float | implied vol, decimal | — |
| q | float | continuous dividend yield, decimal | 0.0 |
| kind | str | `"put"` or `"call"` | `"put"` |

**Returns:** delta (put: −1 to 0, call: 0 to 1). At or after expiry, the payoff delta
(put −1 if S < K, else 0). None if an input is missing or ≤ 0.

**Example:** S = K = 100, T = 1, r = 5%, vol = 20% → call 0.6368, put −0.3632

**Uses:** —  **Used by:** OPT-04

**Gotchas:** European formula; listed US stock options are American. For the out-of-the-money
puts we trade the difference is tiny (validated above). Unknown `kind` raises ValueError.

---

### OPT-04  delta_from_iv
**File:** src/options/greeks.py
**Status:** ✅ done

**What it does:** The function to call when LSEG's `DELTA` is missing. Takes the option's IV,
subtracts upcoming dividends from the stock price, and returns Black-Scholes delta.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| S | float | **unadjusted** stock close on as_of | — |
| K | float | strike | — |
| as_of, expiry | date | pricing date, option expiry | — |
| iv | float | LSEG `IMP_VOLT` / 100 | — |
| r | float | LSEG `US1MT=RR` `B_YLD_1` / 100 | — |
| dividends | list of (ex_date, amount) | as OPT-02 | () |
| kind | str | `"put"` or `"call"` | `"put"` |

**Returns:** delta, or None if inputs are unusable

**Example:** COF $115 put on 2022-05-18 → about −0.489 (LSEG reported −0.4884)

**Uses:** OPT-01, OPT-02, OPT-03  **Used by:** later STR-3.2a / STR-3.3a inputs (pre-2022), BT

**Gotchas:** Raises ValueError when `iv > 5` or `r > 0.25`, which almost always means a percent
was passed. It can't catch every case: in 2021, yields were around 0.05%, so passing `0.05`
by mistake looks like a valid 5%. Always divide `B_YLD_1` by 100.
