# Code reference

Plain-English docs for every function. Each entry is headed by the function's ID; search
the ID with **Ctrl+Shift+F** to jump between these docs, the code and the spec.
Rules for adding entries are in [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Pipeline map

```
ING (download) ──► DAT (clean) ──► EVT (find shocks) ──► STR (rules) ──► BT (simulate) ──► LIVE
                                                         ▲
                                   OPT (RICs, Greeks) ───┘
```

| Prefix | Folder | Docs file | Status |
|---|---|---|---|
| ING | `src/ingest/` | [ING_ingest.md](ING_ingest.md) | ✅ built |
| DAT | `src/data/` | — | ⬜ not started |
| OPT | `src/options/` | [OPT_options.md](OPT_options.md) | 🚧 delta built |
| STR | `strategy/strategy_logic.py` | [STR_strategy.md](STR_strategy.md) | ✅ built, except 2.3 / 2.4 |
| EVT | `src/events/` | — | ⬜ not started |
| BT | `src/backtest/` | — | ⬜ not started |
| LIVE | `src/live/` | — | ⬜ not started |

Status key: ✅ done · 🚧 in progress · ⬜ planned

## Index

### ING: download from LSEG (`src/ingest/`)

| ID | Function | File | Purpose | Status |
|---|---|---|---|---|
| ING-01 | lseg_session | lseg_client.py | Open / close an LSEG session | ✅ |
| ING-02 | fetch_prices | lseg_prices.py | Daily price bars for one stock | ✅ |
| ING-03 | save_prices | lseg_prices.py | Save price bars to parquet | ✅ |
| ING-04 | _get_page | lseg_news.py | One page of headlines, with retries | ✅ |
| ING-05 | _fetch_window | lseg_news.py | All headlines in one time window | ✅ |
| ING-06 | fetch_headlines | lseg_news.py | All headlines for one stock | ✅ |
| ING-07 | save_news | lseg_news.py | Save headlines to parquet | ✅ |
| ING-08 | fetch_earnings_dates | lseg_earnings.py | Earnings report dates | ✅ |
| ING-09 | save_earnings | lseg_earnings.py | Save earnings dates to parquet | ✅ |
| ING-10 | fetch_members | lseg_universe.py | S&P 500 members on one date | ✅ |
| ING-11 | save_membership | lseg_universe.py | Save membership snapshots | ✅ |
| ING-12 | build_membership | lseg_universe.py | Month-end membership history | ✅ |
| ING-13 | members_before | lseg_universe.py | Members just before a shock date | ✅ |
| ING-14 | fetch_index_changes | lseg_universe.py | Every S&P 500 join / leave | ✅ |
| ING-15 | save_changes | lseg_universe.py | Save joins / leaves to parquet | ✅ |

### OPT: option math (`src/options/`)

| ID | Function | File | Purpose | Status |
|---|---|---|---|---|
| OPT-01 | years_to_expiry | greeks.py | Years from a date to expiry | ✅ |
| OPT-02 | pv_dividends | greeks.py | PV of dividends before expiry | ✅ |
| OPT-03 | bs_delta | greeks.py | Black-Scholes delta | ✅ |
| OPT-04 | delta_from_iv | greeks.py | Delta from LSEG IV when DELTA is missing | ✅ |

### STR: strategy rules (`strategy/strategy_logic.py`)

| ID | Function | Spec | Purpose | Status |
|---|---|---|---|---|
| STR-1.1a | passes_universe | 1.1–1.3 | Can we trade this stock at all? | ✅ |
| STR-1.4a | passes_options_liquidity | 1.4 | Are the options liquid enough? | ✅ |
| STR-1.4b | pick_expiry | 1.4 | Expiry closest to target DTE | ✅ |
| STR-2.1a | estimate_beta | 2.1 | Stock's beta vs SPY before the shock | ✅ |
| STR-2.1b | abnormal_return | 2.1 | Part of the move the market doesn't explain | ✅ |
| STR-2.1c | is_shock | 2.1 | Does the day count as a shock? | ✅ |
| STR-2.2a | earnings_clear | 2.2 | No earnings / splits near the trade | ✅ |
| STR-2.3a | categories | 2.3 | News category list + toggles | ⬜ |
| STR-2.4a | overreaction_ratio | 2.4 | Market cap lost ÷ estimated impact | ⬜ |
| STR-2.4b | is_overreaction | 2.4 | Is the reaction overdone? | ⬜ |
| STR-3.2a | pick_short_strike | 3.2 | Short put by delta, below post-news low | ✅ |
| STR-3.3a | pick_long_strike | 3.3 | Long put by width or delta | ✅ |
| STR-3.4a | spread_credit | 3.4 | Credit at mid prices | ✅ |
| STR-3.4b | credit_ok | 3.4 | Credit ≥ min % of width? | ✅ |
| STR-3.5a | realized_vol | 3.5 | Realized vol since the shock | ✅ |
| STR-3.5b | entry_ready | 3.5 | Enter, wait or skip today? | ✅ |
| STR-3.6a | should_take_profit | 3.6 | Close for a profit? | ✅ |
| STR-3.7a | should_stop | 3.7 | Close for a loss? | ✅ |
| STR-3.8a | position_size | 3.8 | Number of spreads to trade | ✅ |
| STR-5a | sector_exposure | 5 | Share of risk in each sector | ✅ |
| STR-5b | portfolio_ok | 5 | Can we add this trade? | ✅ |

## Entry template

Copy this into the right docs file for every new function:

```markdown
### XXX-nn  function_name
**File:** path/to/file.py
**Spec:** strategy/universe_and_parameters.txt x.x   (strategy functions only)
**Status:** ✅ done / 🚧 in progress / ⬜ planned

**What it does:** One or two plain-English sentences.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|

**Returns:** what comes back, including None / error cases

**Example:** concrete numbers in → result out

**Uses:** IDs it calls  **Used by:** IDs that call it

**Gotchas:** units, timezones, look-ahead risks, anything easy to get wrong
```
