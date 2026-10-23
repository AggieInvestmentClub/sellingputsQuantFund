# sellingputsQuantFund
Selling options strategy for quant fund

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

`src/ingest has the code that downloads prices and news.
- scripts/smoke_test is the quick connection test.
- The other files in scripts are the checks used to understand the data.
