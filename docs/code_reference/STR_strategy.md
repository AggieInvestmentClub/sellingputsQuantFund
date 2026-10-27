# STR: strategy rules (`strategy_logic.py`)

The rules from `universe_and_parameters.txt`, written as small functions. Each takes plain
inputs (numbers, Series, DataFrames) and returns an answer. None of them download anything,
so they work the same on backtest data and live data.

**IDs** = spec section + a letter per function (`STR-2.1a`, `STR-2.1b`, ...).

**Gotchas that apply to every STR function:**
- Returns are decimals: `0.012` = 1.2%, `-0.05` = -5%.
- Option prices, credits and widths are **per share** (x100 for one contract).
- Put deltas can be negative (LSEG style) or positive; functions use `abs()`.
- Date-indexed inputs need a **date index without timezone** (ING data must be converted first).

---

### STR-1.1a  passes_universe
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 1.1, 1.2, 1.3
**Status:** ✅ done

**What it does:** Checks whether a stock can be traded at all: in the S&P 500 on the event
date, not excluded (ETF, pending M&A, binary event), and above the price, market cap and
volume floors.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| in_sp500 | bool | member **on the event date** (point-in-time) | — |
| price | float | **unadjusted** price that day | — |
| market_cap | float | $ market cap that day | — |
| avg_dollar_volume | float | 20-day avg $ volume before the shock | — |
| avg_share_volume | float | 20-day avg share volume before the shock | — |
| is_etf, pending_mna, binary_event | bool | exclusions | False |
| min_price | float | price floor | 20 |
| min_market_cap | float | market cap floor | 5e9 |
| min_dollar_volume | float | $ volume floor | 50e6 |
| min_share_volume | float | share volume floor | 1e6 |

**Returns:** True / False

**Example:** in S&P, $150, $50B cap, $500M/day, 3M shares/day → True. Same at $15 → False.

**Uses:** —  **Used by:** later EVT

**Gotchas:** Must use the **as-traded** price. Split-adjusted history makes old prices look
smaller and would wrongly fail the $20 floor.

---

### STR-1.4a  passes_options_liquidity
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 1.4
**Status:** ✅ done

**What it does:** Checks open interest at both strikes, the name's daily options volume, and
that the short put's bid-ask width is at most 10% of mid or $0.15, whichever is larger.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| short_oi, long_oi | int | open interest at each strike | — |
| avg_options_volume | int | avg daily options volume for the name | — |
| short_bid, short_ask | float | short put quote | — |
| min_short_oi / min_long_oi | int | OI floors | 500 / 250 |
| min_options_volume | int | volume floor | 5000 |
| max_spread_pct / max_spread_abs | float | bid-ask limits | 0.10 / 0.15 |

**Returns:** True / False

**Example:** OI 800 / 300, 20k volume, quote 1.20 / 1.30 → True. Quote 1.00 / 1.50 → False.

**Uses:** —  **Used by:** later BT

---

### STR-1.4b  pick_expiry
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 1.4
**Status:** ✅ done

**What it does:** From the listed expiries, picks the one closest to the target days-to-expiry,
within the allowed window.

**Inputs:** `expiries` (list of dates), `entry_date`, `target_dte` (21), `min_dte` (14), `max_dte` (35)

**Returns:** the chosen expiry (Timestamp), or None if none fits

**Example:** entry 2024-03-15, expiries Mar 22 / Apr 5 / Apr 19 / May 17 → Apr 5 (21 days)

**Uses:** —  **Used by:** later BT

---

### STR-2.1a  estimate_beta
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 2.1
**Status:** ✅ done (tested on synthetic data)

**What it does:** Measures how much the stock normally moves per 1% move in SPY, using about
a year of daily returns that ends a week before the shock.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| stock_returns | Series | daily returns, date index | — |
| spy_returns | Series | SPY daily returns, date index | — |
| shock_date | date / str | first day that reacted to the news | — |
| window | int | days used | 250 |
| gap | int | days skipped right before the shock | 5 |
| min_obs | int | minimum usable days | 200 |

**Returns:** beta (float), or None if there's too little history

**Example:** synthetic stock built as 1.5 × SPY + noise → 1.493

**Uses:** —  **Used by:** STR-2.1b

**Gotchas:** Pass **returns, not prices**; prices give a wrong beta with no error. The shock
day and later days are never used (no look-ahead).

---

### STR-2.1b  abnormal_return
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 2.1
**Status:** ✅ done

**What it does:** The part of the day's move that the market doesn't explain:
stock return − beta × SPY return.

**Inputs:** `stock_ret` (float), `spy_ret` (float), `beta` (float or None)

**Returns:** abnormal return (float), or None if beta is None

**Example:** stock -8%, SPY -1%, beta 1.5 → -0.08 − (1.5 × -0.01) = -6.5%

**Uses:** STR-2.1a output  **Used by:** STR-2.1c, STR-2.4a

---

### STR-2.1c  is_shock
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 2.1
**Status:** ✅ done

**What it does:** Decides if a day counts as a shock: the stock-specific drop is big enough
AND the whole market didn't sell off.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| abnormal_ret | float / None | from STR-2.1b | — |
| spy_ret | float | SPY return that day | — |
| threshold | float | min drop to count (test -0.05 / -0.10 / -0.15) | -0.05 |
| max_spy_drop | float | skip if SPY fell more than this | -0.02 |

**Returns:** True / False

**Example:** abnormal -6.5%, SPY -1% → True. Same day at a -10% threshold → False. SPY -3% → False.

**Uses:** STR-2.1b  **Used by:** later EVT

**Gotchas:** None in → False out. Exactly -5% counts as a shock; SPY at exactly -2% does not.

---

### STR-2.2a  earnings_clear
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 2.2
**Status:** ✅ done

**What it does:** Rejects the event if earnings fall within ±3 trading days of the shock, or
if earnings, a split or a special dividend fall between entry and expiry.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| shock_date, entry_date, expiry | date | trade timeline | — |
| earnings_dates | list of dates | from ING-08 | — |
| other_event_dates | list of dates | splits, special dividends | () |
| buffer_days | int | trading days around the shock | 3 |

**Returns:** True (clear) / False

**Example:** shock Mar 15, earnings Mar 12 → False. Earnings Apr 25, trade ends Apr 5 → True.
Split on Mar 28 inside the trade → False.

**Uses:** ING-08 data  **Used by:** later EVT / BT

**Gotchas:** Trading days are counted as weekdays; holidays are ignored.

---

### STR-2.3a  categories
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 2.3
**Status:** ⬜ planned

**What it does:** The list of news categories (downgrade, lawsuit, recall, ...) and on/off
toggles so each category can be traded alone or excluded in tests.

---

### STR-2.4a  overreaction_ratio
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 2.4
**Status:** ⬜ planned

**What it does:** Market cap lost (abnormal return × pre-shock market cap) divided by the
LLM's estimated $ impact of the news.

---

### STR-2.4b  is_overreaction
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 2.4
**Status:** ⬜ planned

**What it does:** True when the overreaction ratio is at or above the threshold
(2x / 3x / 5x) and the LLM says the reaction is not justified.

---

### STR-3.2a  pick_short_strike
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 3.2
**Status:** ✅ done

**What it does:** From one expiry's put chain, picks the put closest to the target delta among
strikes below the post-announcement low.

**Inputs:** `chain` (DataFrame: `strike`, `delta`, `bid`, `ask`), `post_news_low` (float),
`target_delta` (0.20; test 0.15 / 0.25)

**Returns:** the chosen row (Series), or None if no strike is below the low

**Example:** low $93, chain 80–95 → the 90 strike (delta -0.20)

**Uses:** —  **Used by:** STR-3.3a, later BT

---

### STR-3.3a  pick_long_strike
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 3.3
**Status:** ✅ done

**What it does:** Picks the long put below the short strike, either by width (% of stock
price, snapped to a listed strike) or by delta.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| chain | DataFrame | same as STR-3.2a | — |
| short_strike | float | from STR-3.2a | — |
| stock_price | float | current price | — |
| method | str | `"width"` or `"delta"` | `"width"` |
| width_pct | float | test 0.025 / 0.05 / 0.075 / 0.10 | 0.05 |
| target_delta | float | test 0.05 / 0.10 | 0.10 |

**Returns:** the chosen row (Series), or None

**Example:** short 90, stock $100, 5% width → target 85 → the 85 strike

**Uses:** STR-3.2a output  **Used by:** later BT

**Gotchas:** An unknown `method` raises ValueError.

---

### STR-3.4a  spread_credit
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 3.4
**Status:** ✅ done (placeholder fill model)

**What it does:** The credit for selling the spread at mid prices: short mid − long mid.

**Inputs:** `short_bid`, `short_ask`, `long_bid`, `long_ask` (floats)

**Returns:** credit per share (float)

**Example:** short 1.30 / 1.40, long 0.50 / 0.60 → 1.35 − 0.55 = 0.80

**Uses:** —  **Used by:** STR-3.4b, later BT

**Gotchas:** Mid fills are optimistic. Replace once the fill model (spec section 6) is decided.

---

### STR-3.4b  credit_ok
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 3.4
**Status:** ✅ done

**What it does:** Checks that the credit is at least a minimum share of the spread width.

**Inputs:** `credit`, `short_strike`, `long_strike`, `min_credit_pct` (0.25; test 0.15 / 0.20 / 0.33)

**Returns:** True / False (False if width ≤ 0)

**Example:** $5 wide, 0.25 → need ≥ $1.25. A $0.80 credit → False.

**Uses:** STR-3.4a  **Used by:** later BT

---

### STR-3.5a  realized_vol
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 3.5
**Status:** ✅ done

**What it does:** Annualized realized volatility over the days since the shock, not counting
the shock day itself.

**Inputs:** `returns` (Series, date index), `shock_date`, `as_of_date`, `window` (3)

**Returns:** annualized vol (float, e.g. 0.25 = 25%), or None if no days have passed yet

**Example:** returns +1%, -0.5% on the two days after the shock → 0.125 (12.5%)

**Uses:** —  **Used by:** STR-3.5b

**Gotchas:** Uses the zero-mean formula √(mean(r²) × 252) so it works with just 1–3 days.

---

### STR-3.5b  entry_ready
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 3.5
**Status:** ✅ done

**What it does:** Decides whether to enter today. Enters when IV is still rich versus realized
vol, volume has cooled, and the stock didn't make a new low. Skips the event after too long.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| days_since_shock | int | trading days since the shock | — |
| iv | float | 30-day ATM implied vol | — |
| rv | float / None | from STR-3.5a | — |
| volume | float | today's volume | — |
| avg_volume_20d | float | 20-day avg **before** the shock | — |
| made_new_low | bool | traded below the post-news low today | — |
| min_days / max_wait | int | earliest entry / give-up day | 1 / 10 |
| iv_rv_min | float | test 1.2 / 1.5 / 2.0 | 1.5 |
| max_volume_mult | float / None | test 1.5 / 2.0 / None (off) | 1.5 |

**Returns:** `"enter"`, `"wait"` or `"skip"`

**Example:** day 2, IV 60% vs RV 12.5%, normal volume, no new low → "enter". Day 11 → "skip".

**Uses:** STR-3.5a  **Used by:** later BT

---

### STR-3.6a  should_take_profit
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 3.6
**Status:** ✅ done

**What it does:** Decides whether to close for a profit. Dynamic mode never closes below 50%
of the credit; above that, it closes if profit is running ahead of time decay, or at 75%.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| credit | float | credit received | — |
| cost_to_close | float | cost to buy the spread back now | — |
| days_held, dte_at_entry | int | time used / total | — |
| mode | str | `"dynamic"`, `"fixed"`, `"hold"` | `"dynamic"` |
| floor | float | minimum profit to close | 0.50 |
| outpace_margin | float | test 0.10 / 0.20 / 0.30 | 0.20 |
| final_target | float | always close here | 0.75 |
| fixed_target | float | for `"fixed"` mode | 0.50 |

**Returns:** True / False

**Example:** $1 credit, $0.40 to close (60% kept) after 5 of 20 days → 0.60 − 0.25 = 0.35 ≥ 0.20 → True.
Same profit after 15 of 20 days → False.

**Uses:** —  **Used by:** later BT

---

### STR-3.7a  should_stop
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 3.7
**Status:** ✅ done

**What it does:** Checks the chosen stop-loss rules in order and returns the first that fires.

**Inputs:**
| name | type | meaning | default |
|---|---|---|---|
| credit, cost_to_close, width | float | trade P&L inputs | — |
| stock_close | float | today's close | — |
| short_strike, post_news_low | float | price levels | — |
| short_delta | float | short put delta now | — |
| llm_says_justified | bool | LLM re-check after new news | False |
| rules | tuple of str | which stops are on; `()` = none | `("credit_multiple",)` |
| credit_multiple | float | test 1.5 / 2 / 3 | 2.0 |
| max_loss_pct | float | test 0.50 / 0.75 | 0.50 |
| max_short_delta | float | delta limit | 0.50 |

Rule names: `credit_multiple`, `max_loss_pct`, `below_short_strike`, `below_post_news_low`,
`short_delta`, `llm_recheck`.

**Returns:** name of the rule that fired, or None

**Example:** $1 credit, costs $3 to close → loss $2 = 2× credit → `"credit_multiple"`

**Uses:** —  **Used by:** later BT

**Gotchas:** "2x credit" means a **loss** of 2× the credit (buyback cost = 3× credit), not a
spread price of 2× credit. An unknown rule name raises ValueError.

---

### STR-3.8a  position_size
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 3.8
**Status:** ✅ done

**What it does:** Number of spreads so that the max possible loss stays within the risk budget.

**Inputs:** `capital`, `width`, `credit`, `risk_pct` (0.02; test 0.01 / 0.03), `multiplier` (100)

**Returns:** whole number of spreads (0 if even one is too big)

**Example:** $100k, 2% → $2,000 budget; $5 wide, $1.25 credit → $375 max loss each → 5 spreads

**Uses:** —  **Used by:** later BT

---

### STR-5a  sector_exposure
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 5
**Status:** ✅ done

**What it does:** Each sector's share of total capital-at-risk (sum of max losses).

**Inputs:** `positions`: list of dicts with `ticker`, `sector`, `max_loss` ($)

**Returns:** dict `{sector: share}`, e.g. `{"Tech": 0.67, "Energy": 0.33}`; `{}` if no risk

**Uses:** —  **Used by:** STR-5b

---

### STR-5b  portfolio_ok
**File:** strategy_logic.py
**Spec:** universe_and_parameters.txt 5
**Status:** ✅ done

**What it does:** The hard backstop before adding a trade: one position per ticker, and no
sector above 30% of capital-at-risk once 4+ positions are open.

**Inputs:** `positions` (as STR-5a), `ticker`, `sector`, `max_loss`, `max_sector_pct` (0.30),
`min_positions` (4)

**Returns:** `(ok, reason)`, e.g. `(False, "Tech would be 75% of capital-at-risk")`

**Uses:** STR-5a  **Used by:** later BT / LIVE

**Gotchas:** With 1–3 positions any sector is automatically 33–100%, so the cap only applies
from 4 positions. The exposure-review agent from spec section 5 is separate.
