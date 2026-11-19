# Pre-registered test: SPY bull put spread (frozen 2026-11-18, before seeing any pre-2024 data)

Written down BEFORE running on the 2010-2023 SPY chains (Kaggle "dudesurfin/spy-options-eod-volatility-surface-2010-2023",
OptionsDX EOD data with bid/ask). Nothing below may change after the run. The rules were chosen on 2024-04 .. 2026-10
data, so 2010-2023 is a clean out-of-sample test. (First draft named 2014-2024 via Massive; that plan has no access
before 2024-04, so the window moved to the free 2010-2023 data before any of it was seen.)

## Rules (scripts/spy_backtest_quotes.py)
- Entry: first trading day of each week, at the end-of-day snapshot.
- Expiry: listed SPY expiry 14-35 calendar days out, closest to 21.
- Short put: strike below spot with delta closest to -0.15 (the dataset's put delta).
- Long put: listed strike closest to short - 5% of spot.
- Fills: sell at the bid, buy at the ask, $0.65 per contract per leg.
- Exit: dynamic take-profit (STR-3.6a on mid prices: keep >= 50% of the credit and ahead of time by 20 points, or >= 75%),
  paid at the ask/bid; otherwise hold to expiry and settle at intrinsic on the chain's underlying price.
- Account view: one position at a time, 10% of capital at risk per trade.

## Pass / fail (decided now)
Pass only if ALL hold on 2010-2023, net of costs:
1. Mean return on risk per trade > 0, with the 95% bootstrap interval above 0.
2. Positive in at least 70% of calendar years.
3. Worst calendar month at 10% risk per trade better than -25% of the account.
4. Drawdown better than -30% in each crisis window: 2011-08, 2015-08, 2018-02, 2020-03, 2022.
Fail any -> the strategy is not proven, and we say so. No re-runs with changed rules.

Trials already made on 2024-2026 data (count for the deflated Sharpe): shock filters x8, SPY hold/tp x weekly/daily = 4.
