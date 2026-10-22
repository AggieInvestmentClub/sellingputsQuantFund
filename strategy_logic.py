"""
Strategy logic. Section numbers match universe_and_parameters.txt,
so ctrl+F a number (e.g. "2.1") to jump between the spec and the code.
"""
import pandas as pd


# ======================================================================
# 1.1 - 1.4 Universe, exclusions, liquidity filters
# ======================================================================
# TODO passes_universe: S&P 500 point-in-time, price >= $20, market cap,
#      $ volume, share volume, exclusions (ETFs, M&A, binary events)
# TODO passes_options_liquidity: open interest, options volume, bid-ask width


# ======================================================================
# 2.1 Price trigger
# ======================================================================

def estimate_beta(stock_returns, spy_returns, shock_date, window=250, gap=5, min_obs=200):
    """
    [2.1] beta for the market-model residual

    function is assuming data in two pandas(1 for spy, 1 for stock)
    date     return
    2024-xx-xx 0.12 (reflecting 12%)  something like this
    window is how many days, gap is how many trading days before the shock to leave out
    min_obs is a minimum requirement where beta can't be calculated if we don't have enough days

    beta is basically like correlation to spy 
    """
    # Line up both series on the same dates and drop days missing from either
    df = pd.concat([stock_returns, spy_returns], axis=1,
                   keys=["stock", "spy"]).dropna()

    # Only use days strictly before the shock (no look-ahead)
    df = df[df.index < pd.Timestamp(shock_date)]

    # Skip the `gap` days right before the shock (pre-news drift / leaks)
    if gap > 0:
        df = df.iloc[:-gap]

    df = df.iloc[-window:]
    if len(df) < min_obs:
        return None

    return df["stock"].cov(df["spy"]) / df["spy"].var()


def abnormal_return(stock_ret, spy_ret, beta):
    """
    [2.1] beta-adjusted abnormal 1-day return

    part of the stock's move the market doesn't explain
    abnormal = stock return - beta * spy return
    ex: stock -0.08, spy -0.01, beta 1.5 -> -0.08 - (1.5 * -0.01) = -0.065
    returns None if beta is None (not enough history)
    """
    if beta is None:
        return None
    return stock_ret - beta * spy_ret


def is_shock(abnormal_ret, spy_ret, threshold=-0.05, max_spy_drop=-0.02):
    """
    [2.1] shock threshold + SPY selloff filter

    True if the day counts as a shock event
    threshold: abnormal drop must be at least this big (test -0.05, -0.10, -0.15)
    max_spy_drop: skip market-wide selloffs where spy fell more than 2%
    """
    if abnormal_ret is None:
        return False

    # Stock-specific drop is big enough
    big_drop = abnormal_ret <= threshold

    # Market didn't sell off that day
    market_ok = spy_ret > max_spy_drop

    return big_drop and market_ok


# ======================================================================
# 2.2 Earnings exclusion
# ======================================================================
# TODO earnings_clear: no earnings within +/- 3 days of shock or between
#      entry and expiry (also splits / special dividends in the window)


# ======================================================================
# 2.3 News categories
# ======================================================================
# TODO category list + on/off toggles per category


# ======================================================================
# 2.4 LLM filter
# ======================================================================
# TODO overreaction_ratio: market cap lost / LLM estimated $ impact
# TODO is_overreaction: ratio >= threshold and LLM says "not justified"


# ======================================================================
# 3.2 Short strike
# ======================================================================
# TODO pick_short_strike: target delta (0.15 / 0.20 / 0.25), below post-news low


# ======================================================================
# 3.3 Long strike / spread width
# ======================================================================
# TODO pick_long_strike: by width (% of price) or by delta, rounded to listed strikes


# ======================================================================
# 3.4 Minimum credit
# ======================================================================
# TODO credit_ok: credit >= min % of spread width


# ======================================================================
# 3.5 Entry (IV lag / selling-exhaustion trigger)
# ======================================================================
# TODO realized_vol: RV since the shock, excluding the shock day
# TODO entry_ready: IV / RV >= threshold, volume cooled, no new low, max wait


# ======================================================================
# 3.6 Profit target
# ======================================================================
# TODO should_take_profit: 50% floor, close if captured % outpaces time elapsed


# ======================================================================
# 3.7 Stop loss
# ======================================================================
# TODO should_stop: credit multiple, % of max loss, price, delta, LLM re-check


# ======================================================================
# 3.8 Max loss per trade
# ======================================================================
# TODO position_size: contracts so max loss <= risk % of capital


# ======================================================================
# 5 Portfolio-level risk
# ======================================================================
# TODO sector_exposure: max loss by sector, hard 30% backstop
