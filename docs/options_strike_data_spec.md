# Option strike data for the options team

**Status (2026-11-05):** data pulled; panel built with `scripts/build_option_panel.py` (1,286 events tradable on day +1).
**Owner of this file:** data side (Juan Pablo). **Readers:** options / backtest team.

The panel holds, for every shock in the options period, every put the strategy could trade, with its
daily price, implied vol and delta from the shock day until expiry. The options team computes the
trade-level numbers (strike choice, credit, spread value, exits) from it, using the STR functions.

## 1. Coverage

| Item | Value |
|---|---|
| Source | OPRA day aggregates (Massive/Polygon flat files), copied from darkpool `/srv/data/options_model_features/FlatFiles/us_options_opra/day_aggs_v1` |
| Dates | 2024-04-01 .. 2026-10-28 (648 trading days, no gaps besides market holidays) |
| Events | shocks from `data/results/event_study/events.csv` dated 2024-04-01 or later (1,507) |
| Per event | puts on the event's option root, expiry between day +1 + 14 days and day +10 + 35 days, every day from the shock to expiry |
| Strikes | **all traded strikes** in that expiry range; no strike filter, so every delta / width rule in section 4 of the spec can be tested |

Build of 2026-11-05 (`data/processed/options/coverage.txt`):

| | 2024 (Apr–Dec) | 2025 | 2026 (Jan–Oct) | Total |
|---|---|---|---|---|
| Events in the options period | 294 | 456 | 757 | **1,507** |
| Full window inside the data | 294 | 456 | 639 | 1,389 |
| Any put rows | 289 | 454 | 756 | 1,499 |
| **Traded put on day +1, 14–35 DTE, delta −0.10..−0.30** | 242 | 386 | 658 | **1,286 (85%)** |

- 2,940,751 put rows; 89% have an implied vol; every row has a stock close.
- IV percentiles (5/25/50/75/95%): 0.31 / 0.50 / 0.69 / 0.98 / 1.87. High because these are out-of-the-money puts right after a shock; the top end is cheap, stale far-out puts.
- Events with puts but nothing tradable on day +1 are mostly names with thin options (REITs such as EXR, PSA, SBAC; insurers such as ACGL, AIZ, EG): the liquidity filter at work.
- The 118 events without a full window are from Sep–Oct 2026: their expiries run past 2026-10-28. Refresh the flat files from darkpool to complete them.

## 2. What the strategy needs from the chain (spec section 3)

| Rule | Spec | What it reads from the panel |
|---|---|---|
| Expiry (STR-1.4b) | 14–35 DTE at entry, target 21 | `expiry` on the entry day |
| Short put (STR-3.2a) | delta 0.15 / 0.20* / 0.25, strike below the post-news low | `delta`, `strike`; post-news low from stock prices |
| Long put (STR-3.3a) | width 2.5 / 5* / 7.5 / 10% of price, or delta 0.05 / 0.10 | `strike`, `delta`, `S` |
| Credit (STR-3.4a/b) | ≥ 25%* of width (15–33%) | `close` of both legs, fill model (section 5) |
| Entry (STR-3.5b) | 30-day ATM IV / realized vol ≥ 1.5*, max wait 10 days | ATM IV built from `iv` (section 5) |
| Profit / stop (STR-3.6a, 3.7a) | daily cost to close, short-leg delta | `close`, `delta` of both legs each day |
| Liquidity (STR-1.4a) | OI ≥ 500 / 250, name volume ≥ 5,000/day, short-leg width ≤ 10% of mid | `volume`, `transactions`; `root_volume.parquet` (no OI, no bid/ask: section 4) |

So the strikes that matter run from roughly delta −0.05 to −0.30: in practice about 70%–100% of the
stock price for 2–5 week puts after a shock (IV is high, so the 0.20-delta strike sits further out than usual).

## 3. Files and columns (`data/processed/options/`)

**`event_windows.parquet`**, one row per event (EVT-09)

| Column | Meaning |
|---|---|
| `event_id` | `RIC\|YYYY-MM-DD` (shock date), join key everywhere |
| `ric`, `date` | stock and shock date |
| `root` | OCC option root on that date (FB before 2022-06-09, BRKB for BRK.B) |
| `min_expiry`, `max_expiry` | expiry range kept |
| `window_end` | last date kept (= `max_expiry`) |
| `complete` | False if the window runs past 2026-10-28 (recent events) |

**`event_puts.parquet`**, one row per event × day × put (EVT-10, EVT-12)

| Column | Unit | Meaning |
|---|---|---|
| `event_id`, `ric`, `date` | | event and trading day |
| `ticker` | | OCC symbol, e.g. `O:AAPL250502P00200000` |
| `root`, `expiry`, `strike` | $ | parsed from the symbol (OPT-05) |
| `open`, `high`, `low`, `close` | $ per share | trade prices that day (×100 per contract) |
| `volume`, `transactions` | contracts, trades | that day |
| `S` | $ | **unadjusted** stock close that day (strikes are as traded) |
| `T` | years | (expiry − date) calendar days / 365 |
| `r` | decimal | 1-month T-bill (US1MT=RR), forward-filled |
| `pv_div` | $ per share | PV of cash dividends going ex in (date, expiry] |
| `iv` | decimal | implied vol from `close` (OPT-08); NaN if no vol fits |
| `delta` | | put delta from `iv` (OPT-09), negative |

**`root_volume.parquet`**, one row per root × day (EVT-11): `opt_volume` (calls + puts), `put_volume`,
`contracts_traded`. Average it over 20 days for STR-1.4a's name-level options volume.

## 4. What is NOT in the data

| Missing | Why it matters | Proposed handling |
|---|---|---|
| **Bid / ask** | credit, fill price, bid-ask width filter | fill model (section 5); optional: LSEG bid/ask for the chosen legs only (a few requests per trade) |
| **Open interest** | STR-1.4a OI ≥ 500 / 250 | replace with traded volume on the entry day (e.g. ≥ 50 contracts per leg) and 20-day name volume; optional LSEG OI for chosen legs |
| **Strikes with no trades that day** | a held leg can be missing on some days | mark it from the neighbouring strikes' IV (same expiry), or carry the last price; decide which |
| **Before 2024-04-01** | the 2015–2024 events have no option data | backtest period is Apr 2024 – Oct 2026; 2023 folders on darkpool are empty (ask Jorge) |
| **IV history for IV rank** | section 4 filter "IV rank ≥ 50" needs a year of ATM IV | possible from 2025-04 on, by running the panel logic on every day (not built) |

## 5. What the options team computes

1. **Fill model** (spec section 6, open). Proposal to start: sell at `close` × (1 − 5%), buy at `close` × (1 + 5%) per leg, minimum $0.01 haircut; commissions $0.65 per contract per leg. Replace with LSEG bid/ask once tested.
2. **30-day ATM IV per stock per day** (entry rule): for each day, interpolate `iv` at delta −0.50 on the two expiries around 30 days, then in time to exactly 30 days. Proposed ID OPT-10 `atm_iv_30d`.
3. **Realized vol since the shock** from stock returns: STR-3.5a already does this.
4. **Spread value each day** (cost to close) with the missing-leg rule above. Proposed ID OPT-11 `spread_mark`.
5. **Strike choice and credit**: STR-3.2a, STR-3.3a, STR-3.4a/b on the entry-day rows (`chain` = that day's rows for the chosen expiry, columns `strike`, `delta`).

## 6. Data caveats

- `close` is the **last trade**, which can be well before the 4 pm stock close on thin strikes: IV and delta are noisy there. Prefer strikes with `transactions` ≥ 5 that day.
- IV is from a **European** model on American puts. Fine for the out-of-the-money puts we trade; deep in-the-money puts can come out NaN.
- Events on crowded days (5+ shocks across the S&P on one day) are market moves, not company news; consider dropping them (event-study analysis, 2026-10-29).
- Check our IV against Jorge's `options_data/iv` on darkpool for a sample before relying on it.

## 7. How to rebuild

```powershell
py -3.12 -u scripts/build_option_panel.py
```
No LSEG needed. Needs the flat files in `data/raw/options/day_aggs_v1/` plus the outputs of
`overnight.py` and `build_security_master.py`. About 5–10 minutes; resumable by month.
