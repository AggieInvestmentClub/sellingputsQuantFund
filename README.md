# sellingputsQuantFund
Selling options strategy for quant fund

Essentially the essence of the strategy is capitalizing on overreactions to stock specific news that doesn't change the companies valuation or add certain risk to the company.

Initial proposed pipeline: 

news feed->stock reaction -> use LLM agent council or 1 LLM to evaluate if the reaction is justified -> If not justified sell a put option for a long term hold and to gain money from increase in IV 

Considerations: strikes, put spreads instead of naked puts, how to exit positions, how to exit positions after being assigned shares, position sizing, bid-ask spreads, rules for stock screening 

To validate the strategy we must use statistical methods and live test / backtest this strategy. It is hard to backtest the strategy as news feeds aren't live in the past and it may be hard to backtest entries and exits 
considering bid asks and non-live news feeds. 
