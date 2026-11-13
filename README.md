# sellingputsQuantFund
Selling options strategy for quant fund

> **Team: read [CONTRIBUTING.md](CONTRIBUTING.md) after you pull.** Every function has an ID tag
> (e.g. `STR-2.1c`, `ING-06`). Ctrl+Shift+F an ID in VS Code to see the code, its docs in
> [docs/code_reference/](docs/code_reference/README.md), and its spec rule together. New or changed
> functions must be tagged and documented in the same commit.

Essentially the essence of the strategy is capitalizing on overreactions to stock specific news that doesn't change the companies valuation or add certain risk to the company.

Initial proposed pipeline: 

news feed->stock reaction -> use LLM agent council or 1 LLM to evaluate if the reaction is justified -> If not justified sell a put option for a long term hold and to gain money from increase in IV 

Considerations: strikes, put spreads instead of naked puts, how to exit positions, how to exit positions after being assigned shares, position sizing, bid-ask spreads, rules for stock screening 

To validate the strategy we must use statistical methods and live test / backtest this strategy. It is hard to backtest the strategy as news feeds aren't live in the past and it may be hard to backtest entries and exits 
considering bid asks and non-live news feeds. 

*** 10/21 update ***
Whats working: 
- We have access to LSEG Workspace through our TAMU accounts, and Python can connect to it.
- We can download daily prices and news. We tested it on five stocks (Apple, Microsoft, Nvidia, JPMorgan, Exxon) for Sept 1 to Oct 15, 2026.
- We looked closely at what came back, and that taught us a few things (below)

What was found in this data:
- **Most of the news is junk.** Each stock had thousands of headlines in six weeks, and a lot of them were patent filings and automated feeds. Real company news is a small slice.
- **Reuters is the one to trust.** It gives exact publication times. Some other sources only give a date, or a date that's later than when the news actually came out. If we used those, our backtest would "know" things before the market did and look better than it should.
- **One event, many headlines.** Reuters often posts several short items within minutes about the same story. We'll treat the first one as the moment the news broke.
- **Not everything mentions the company for a reason.** Some items only name-drop it, so we'll need a filter to keep the ones that are actually about the company.

*** 10/22 update ***

- Fixed the API connection with LSEG by using Python 3.12. This cus for some reason on one laptop, 3.14 packages were blocked by a Windows security policy.
- Daily prices go back to 2000 and are split-adjusted. We can also get unadjusted prices, which we need for the $20 price floor.
- Market cap is available daily.
- Earnings dates come back at about four per year (checked for Apple and Microsoft). Past dates work, but the next upcoming report does not show up, so live trading will need another source.
- Reuters news has exact publication times back to at least 2018. Most other news sources are patent filings, automated feeds, or date-only entries, so we ignore them.
- S&P 500 membership as of any past month works from April 2015 on (about 500 to 506 names each month). Companies that were later removed or acquired still have prices and earnings dates, which helps avoid survivorship bias.
- SPY works as the market return series. The `.SPX` index code failed.

**What doesn't work yet**
- Membership before April 2015 comes back empty.
- We haven't tested any source for historical options data (prices, bid/ask, implied vol). Without it we can test whether stocks rebound after shocks, but not the put spread trades themselves.
- Sector data (for the 30% sector limit) is untested.

**What was found**
- Raw news is mostly noise, so the plan is to find price shocks first and only look up Reuters news around those dates.
- Some sources have timestamps later than the real publication, which would make a backtest look better than it should, so those are dropped.

*** 10/29 update ***
- Built the historical S&P 500 membership dataset from April 2015 onward because LSEG returns empty results before that date.
- Found all S&P 500 companies represented across the full 2015-2026 period, with 138 monthly snapshots and 69,579 membership rows.
- Unique historical RICs: 736; 132 were marked as delisted; snapshot sizes stayed in the normal 500-506 range.
- The validation script confirmed 138 clean snapshots. The remaining `.SPX` LSEG error should be investigated separately.

- `src/ingest` has the code that downloads prices and news.
- `scripts/` has the runnable pipeline steps (downloads, builds, validation); every script says how to run it at the top.
- `python scripts/download_event_news.py --limit 20` is a quick LSEG connection test (20 events).

*** 11/12 update: backtest on real option prices ***

We now backtest the actual trade: one bull put spread per shock (≥5% abnormal drop, $5B+ market cap), using
real OPRA daily option prices for 2024-04 to 2026-11. Code is in `src/backtest/` (BT-01..13, docs in
[docs/code_reference/BT_backtest.md](docs/code_reference/BT_backtest.md)); results and charts are in
`data/results/backtest/` (open `index.html` for the dashboard). "Return on risk" = P&L ÷ the spread's max loss.

What we found:
- **Short delta 0.15 is best in every run; 0.25 is worst.** Further-out strikes survive the stock drifting lower after the news.
- **Enter within 3 days of the shock.** Entries on day +4 to +10 lost 6-14% on average; capping the wait took the baseline from −4.3% to −1.4%.
- **No stop-loss beats every stop we tried.** The long put already caps the loss; stops mostly sold at the low on a noisy close. With no stop the baseline is +2.2%.
- **New rule:** if the stock closes between the strikes on expiry day, buy the spread back instead of being assigned shares.
- **The shock itself is not an edge (yet).** Running the same rules on 3,014 random no-shock days in the same stocks earned as much or more (baseline +4.1% vs +2.2%; shocks win in only 31 of 145 settings). The profit so far is from selling put premium in general.
- **LLM "not justified" labels didn't beat simply "has Reuters news".**
- **vs SPY:** at 2% of capital risked per trade, no setting beats SPY's ~47% return over the period, but 24 settings beat its Sharpe ratio with drawdowns under 2%.

Caveats: one bull-market period (no 2020/2022-style crash in the option data), fills are modeled (day
aggregates have no bid/ask), 40-200 trades per setting, and many settings were tried, so the single best
setting is likely overfit. Next: test bigger shocks only, the valuation gate, and a pre-2024 stress test.

**How to rerun it** (from the repo root, Python venv with `requirements.txt`; no LSEG login needed for these steps).
Needs the local data the downloads produce (`data/raw/`: prices, earnings dates, GICS, market cap, rates, and
the OPRA day-aggregate files in `data/raw/options/day_aggs_v1/`) plus `data/results/event_study/events.csv`.
`data/` is gitignored except `data/results/backtest/`, so ask a teammate for the raw files.

```bash
# 1. option panel per shock from the OPRA files (~15 min)
python -u scripts/build_option_panel.py

# 2. backtests: --mode oat = one change at a time, --mode grid = full product of grid_dims
python -u scripts/run_backtest.py --pricer panel --mode oat  --start 2024-04-01 --workers 8 --out data/results/backtest/oat_panel_w3
python -u scripts/run_backtest.py --pricer panel --mode grid --start 2024-04-01 --workers 8 --out data/results/backtest/grid_panel_w3
#    no stop-loss, wait 1-3 days
python -u scripts/run_backtest.py --config strategy/backtest_nostop.yaml --pricer panel --mode oat  --start 2024-04-01 --workers 8 --out data/results/backtest/oat_nostop
python -u scripts/run_backtest.py --config strategy/backtest_nostop.yaml --pricer panel --mode grid --start 2024-04-01 --workers 8 --out data/results/backtest/grid_nostop

# 3. random-entry control: same rules on random no-shock days
python scripts/make_random_events.py
python -u scripts/build_option_panel.py --events data/results/event_study/random_events.csv --out data/processed/options_random
python -u scripts/run_backtest.py --config strategy/backtest_random.yaml --events data/results/event_study/random_events.csv \
    --panel data/processed/options_random/event_puts.parquet --pricer panel --mode grid --start 2024-04-01 --workers 8 --out data/results/backtest/grid_random
python scripts/compare_random.py --real data/results/backtest/grid_nostop --random data/results/backtest/grid_random

# 4. charts
python scripts/make_report.py --run data/results/backtest/grid_nostop          # one report per run
python scripts/plot_vs_spy.py --runs data/results/backtest/oat_nostop data/results/backtest/grid_nostop --out data/results/backtest/vs_spy.html
python scripts/make_dashboard.py                                                # data/results/backtest/index.html

# 5. view everything at http://127.0.0.1:8765/
python -m http.server 8765 --bind 127.0.0.1 --directory data/results/backtest

# tests
python -m pytest -q
```

Change parameters in `strategy/backtest_grid.yaml` (or a copy passed with `--config`): `baseline` is the
default trade, `space` the values to try, `grid_dims` which ones to cross in `--mode grid`, and
`event_filters` the event sets scored on the finished trades. `--limit 20` runs a quick smoke test.
