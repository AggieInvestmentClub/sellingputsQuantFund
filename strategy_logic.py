"""
Strategy logic. Section numbers match universe_and_parameters.txt,
so ctrl+F a number (e.g. "2.1") to jump between the spec and the code.

Conventions:
- returns are decimals (0.012 = 1.2%)
- option prices, credits and widths are per share (x100 for one contract)
- put deltas can be passed negative (LSEG style) or positive, abs() is used
"""
import math

import numpy as np
import pandas as pd


# ======================================================================
# 1.1 - 1.4 Universe, exclusions, liquidity filters
# ======================================================================

def passes_universe(in_sp500, price, market_cap, avg_dollar_volume, avg_share_volume,
                    is_etf=False, pending_mna=False, binary_event=False,
                    min_price=20, min_market_cap=5e9,
                    min_dollar_volume=50e6, min_share_volume=1e6):
    """
    [1.1] [1.2] [1.3] can we trade this stock at all?

    in_sp500: was it in the S&P 500 ON THE EVENT DATE (point-in-time, not today)
    price, market_cap: as-traded (UNADJUSTED) values on the event date,
        split-adjusted prices make old prices look too small
    avg_dollar_volume, avg_share_volume: 20-day averages before the shock
    is_etf, pending_mna, binary_event: exclusions from 1.2
    """
    if not in_sp500 or is_etf or pending_mna or binary_event:
        return False

    return (price >= min_price
            and market_cap >= min_market_cap
            and avg_dollar_volume >= min_dollar_volume
            and avg_share_volume >= min_share_volume)


def passes_options_liquidity(short_oi, long_oi, avg_options_volume, short_bid, short_ask,
                             min_short_oi=500, min_long_oi=250, min_options_volume=5000,
                             max_spread_pct=0.10, max_spread_abs=0.15):
    """
    [1.4] are the options liquid enough to trade the spread?

    short_oi, long_oi: open interest at the short / long strike
    avg_options_volume: avg daily options volume for the whole name
    short_bid, short_ask: quote on the short put
    bid-ask width must be <= 10% of mid OR <= $0.15, whichever is larger
    """
    if short_oi < min_short_oi or long_oi < min_long_oi:
        return False
    if avg_options_volume < min_options_volume:
        return False

    mid = (short_bid + short_ask) / 2
    if mid <= 0:
        return False
    max_width = max(max_spread_pct * mid, max_spread_abs)
    return (short_ask - short_bid) <= max_width


def pick_expiry(expiries, entry_date, target_dte=21, min_dte=14, max_dte=35):
    """
    [1.4] pick the listed expiry closest to target_dte, within min_dte - max_dte

    expiries: list of listed expiry dates
    returns the chosen expiry (Timestamp) or None if nothing in the window
    """
    entry = pd.Timestamp(entry_date)
    best, best_gap = None, None
    for exp in expiries:
        exp = pd.Timestamp(exp)
        dte = (exp - entry).days
        if min_dte <= dte <= max_dte:
            gap = abs(dte - target_dte)
            if best_gap is None or gap < best_gap:
                best, best_gap = exp, gap
    return best


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

def earnings_clear(shock_date, entry_date, expiry, earnings_dates,
                   other_event_dates=(), buffer_days=3):
    """
    [2.2] no earnings near the shock or inside the trade window

    fails if:
    - an earnings date is within +/- buffer_days trading days of the shock
    - an earnings date falls between entry and expiry
    - any other_event_dates (splits, special dividends) fall between entry and expiry
    trading days are counted as weekdays (holidays ignored, close enough for a buffer)
    """
    shock = pd.Timestamp(shock_date).normalize()
    entry = pd.Timestamp(entry_date).normalize()
    exp = pd.Timestamp(expiry).normalize()

    for e in earnings_dates:
        e = pd.Timestamp(e).normalize()
        # Weekdays between the earnings date and the shock, either direction
        days_apart = abs(np.busday_count(min(e, shock).date(), max(e, shock).date()))
        if days_apart <= buffer_days:
            return False
        if entry <= e <= exp:
            return False

    for d in other_event_dates:
        if entry <= pd.Timestamp(d).normalize() <= exp:
            return False

    return True


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

def pick_short_strike(chain, post_news_low, target_delta=0.20):
    """
    [3.2] short put closest to target delta, strike below the post-announcement low

    chain: DataFrame of PUTS for one expiry, columns: strike, delta, bid, ask
    target_delta: 0.15 / 0.20 / 0.25
    returns the chosen row (pd.Series) or None
    """
    candidates = chain[chain["strike"] < post_news_low]
    if candidates.empty:
        return None
    gap = (candidates["delta"].abs() - target_delta).abs()
    return candidates.loc[gap.idxmin()]


# ======================================================================
# 3.3 Long strike / spread width
# ======================================================================

def pick_long_strike(chain, short_strike, stock_price, method="width",
                     width_pct=0.05, target_delta=0.10):
    """
    [3.3] long put below the short strike

    method "width": strike closest to short_strike - width_pct * stock_price
        (width_pct: 0.025 / 0.05 / 0.075 / 0.10), snaps to a listed strike
    method "delta": strike closest to target_delta (0.05 / 0.10)
    returns the chosen row (pd.Series) or None
    """
    candidates = chain[chain["strike"] < short_strike]
    if candidates.empty:
        return None

    if method == "width":
        target_strike = short_strike - width_pct * stock_price
        gap = (candidates["strike"] - target_strike).abs()
    elif method == "delta":
        gap = (candidates["delta"].abs() - target_delta).abs()
    else:
        raise ValueError(f"unknown method: {method}")

    return candidates.loc[gap.idxmin()]


# ======================================================================
# 3.4 Minimum credit
# ======================================================================

def spread_credit(short_bid, short_ask, long_bid, long_ask):
    """
    [3.4] credit for selling the spread at mid prices
    (fill model is TBD in section 6, swap this out once it's decided)
    """
    return (short_bid + short_ask) / 2 - (long_bid + long_ask) / 2


def credit_ok(credit, short_strike, long_strike, min_credit_pct=0.25):
    """
    [3.4] credit must be >= min_credit_pct of spread width (test 0.15 / 0.20 / 0.25 / 0.33)
    ex: $5 wide, 0.25 -> need at least $1.25 credit
    """
    width = short_strike - long_strike
    if width <= 0:
        return False
    return credit >= min_credit_pct * width


# ======================================================================
# 3.5 Entry (IV lag / selling-exhaustion trigger)
# ======================================================================

def realized_vol(returns, shock_date, as_of_date, window=3):
    """
    [3.5] annualized realized vol since the shock, excluding the shock day

    returns: pd.Series of daily returns indexed by date
    uses the last `window` days after the shock (fewer if not available yet, min 1)
    zero-mean formula sqrt(mean(r^2) * 252), works with just 1 day of data
    returns None if there are no days after the shock yet
    """
    shock = pd.Timestamp(shock_date)
    as_of = pd.Timestamp(as_of_date)
    post = returns[(returns.index > shock) & (returns.index <= as_of)].dropna()
    post = post.iloc[-window:]
    if post.empty:
        return None
    return math.sqrt((post ** 2).mean() * 252)


def entry_ready(days_since_shock, iv, rv, volume, avg_volume_20d, made_new_low,
                min_days=1, max_wait=10, iv_rv_min=1.5, max_volume_mult=1.5):
    """
    [3.5] should we enter today?

    returns "enter", "wait", or "skip" (waited too long, drop the event)
    iv: 30-day ATM implied vol, rv: from realized_vol (both annualized decimals)
    avg_volume_20d: average BEFORE the shock, so the shock days don't inflate it
    made_new_low: True if today traded below the post-announcement low
    iv_rv_min: 1.2 / 1.5 / 2.0
    max_volume_mult: 1.5 / 2.0 / None (None = volume check off)
    """
    if days_since_shock > max_wait:
        return "skip"
    if days_since_shock < min_days or rv is None:
        return "wait"

    # IV still rich vs what the stock is actually doing
    iv_rich = rv <= 0 or iv / rv >= iv_rv_min

    # Selling has cooled
    volume_ok = max_volume_mult is None or volume <= max_volume_mult * avg_volume_20d

    if iv_rich and volume_ok and not made_new_low:
        return "enter"
    return "wait"


# ======================================================================
# 3.6 Profit target
# ======================================================================

def should_take_profit(credit, cost_to_close, days_held, dte_at_entry, mode="dynamic",
                       floor=0.50, outpace_margin=0.20, final_target=0.75,
                       fixed_target=0.50):
    """
    [3.6] close for a profit?

    credit: credit received at entry, cost_to_close: what it costs to buy the spread back now
    captured = share of the credit we've kept, elapsed = share of the trade's time used up

    mode "dynamic": never below the 50% floor; close if captured is outpacing time
        by outpace_margin (0.10 / 0.20 / 0.30), or once final_target is hit
        ex: 60% captured after 25% of the time -> 0.60 - 0.25 = 0.35 >= 0.20 -> close
    mode "fixed": close at fixed_target (0.50 / 0.75)
    mode "hold": never take profit early, hold to expiry
    """
    captured = (credit - cost_to_close) / credit
    elapsed = days_held / dte_at_entry

    if mode == "hold":
        return False
    if mode == "fixed":
        return captured >= fixed_target
    if mode != "dynamic":
        raise ValueError(f"unknown mode: {mode}")

    if captured < floor:
        return False
    if captured >= final_target:
        return True
    return captured - elapsed >= outpace_margin


# ======================================================================
# 3.7 Stop loss
# ======================================================================

def should_stop(credit, cost_to_close, width, stock_close, short_strike, post_news_low,
                short_delta, llm_says_justified=False, rules=("credit_multiple",),
                credit_multiple=2.0, max_loss_pct=0.50, max_short_delta=0.50):
    """
    [3.7] close for a loss?

    returns the name of the first rule that fired, or None
    rules: any combination of the names below, () = no stop (hold to expiry)
      "credit_multiple"      loss >= credit_multiple x credit (1.5 / 2 / 3)
                             ex: $1 credit, 2x -> stop when buyback costs $3 ($2 loss)
      "max_loss_pct"         loss >= max_loss_pct of max spread loss (0.50 / 0.75)
      "below_short_strike"   stock closed below the short strike
      "below_post_news_low"  stock closed below the post-announcement low
      "short_delta"          short put delta above max_short_delta
      "llm_recheck"          new news, LLM re-ran 2.4 and now says justified
    """
    loss = cost_to_close - credit
    max_loss = width - credit

    checks = {
        "credit_multiple": loss >= credit_multiple * credit,
        "max_loss_pct": max_loss > 0 and loss >= max_loss_pct * max_loss,
        "below_short_strike": stock_close < short_strike,
        "below_post_news_low": stock_close < post_news_low,
        "short_delta": abs(short_delta) > max_short_delta,
        "llm_recheck": llm_says_justified,
    }

    for rule in rules:
        if rule not in checks:
            raise ValueError(f"unknown stop rule: {rule}")
        if checks[rule]:
            return rule
    return None


# ======================================================================
# 3.8 Max loss per trade
# ======================================================================

def position_size(capital, width, credit, risk_pct=0.02, multiplier=100):
    """
    [3.8] number of spreads so max loss <= risk_pct of capital (0.01 / 0.02 / 0.03)

    max loss per spread = (width - credit) * 100
    ex: $100k, 2% -> $2,000 budget; $5 wide, $1.25 credit -> $375 each -> 5 spreads
    returns 0 if even one spread is too big
    """
    max_loss_per_spread = (width - credit) * multiplier
    if max_loss_per_spread <= 0:
        return 0
    return int((capital * risk_pct) // max_loss_per_spread)


# ======================================================================
# 5 Portfolio-level risk
# ======================================================================

def sector_exposure(positions):
    """
    [5] share of total capital-at-risk in each sector

    positions: list of dicts with keys "ticker", "sector", "max_loss" ($ max loss of the position)
    returns {sector: share}, ex: {"Tech": 0.40, "Energy": 0.60}
    """
    total = sum(p["max_loss"] for p in positions)
    if total <= 0:
        return {}
    shares = {}
    for p in positions:
        shares[p["sector"]] = shares.get(p["sector"], 0) + p["max_loss"] / total
    return shares


def portfolio_ok(positions, ticker, sector, max_loss, max_sector_pct=0.30, min_positions=4):
    """
    [5] hard backstop before adding a new position (the exposure-review agent is separate)

    - one position per ticker
    - one sector <= max_sector_pct of total capital-at-risk, only enforced once there are
      min_positions open (with 1-3 positions any sector is automatically 33-100%)
    returns (ok, reason)
    """
    if any(p["ticker"] == ticker for p in positions):
        return False, "already holding this ticker"

    after = positions + [{"ticker": ticker, "sector": sector, "max_loss": max_loss}]
    if len(after) < min_positions:
        return True, None

    share = sector_exposure(after).get(sector, 0)
    if share > max_sector_pct:
        return False, f"{sector} would be {share:.0%} of capital-at-risk"
    return True, None
