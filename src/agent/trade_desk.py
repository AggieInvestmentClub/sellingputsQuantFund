"""
AG-12 / AG-13: what to do with a TAKE (entry plan) and with each open spread (hold / close).

Uses the strategy's own rules (strategy/strategy_logic.py) with the settings the 11/12 backtest chose:
  entry   day +1 .. +3 after the shock (later entries lost 6-14%), selling cooled, no new low (STR-3.5b)
  expiry  listed Friday closest to 21 days, 14-35 (STR-1.4b), no earnings inside (STR-2.2a)
  strikes short put ~0.15 delta BELOW the post-news low (STR-3.2a), long = short - 5% of price (STR-3.3a)
  credit  >= 25% of the width (STR-3.4b)          size  max loss <= 2% of capital (STR-3.8a)
  exit    dynamic take-profit (STR-3.6a: keep >= 50% of the credit and ahead of time by 20%, or 75%)
          NO stop-loss (every stop tested did worse; the long put already caps the loss)
          expiry day: stock between the strikes -> buy the spread back (avoid assignment)

Option prices here are Black-Scholes ESTIMATES (OPT-07) from one implied vol. Real strikes, credit and
the buy-back price come from the broker's chain: the plan says what to look for, the broker decides.
"""
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.options.greeks import bs_price
from strategy.strategy_logic import pick_expiry, position_size, should_take_profit

R = 0.04            # risk-free rate for the estimates
P = dict(entry_days=(1, 3), target_dte=21, min_dte=14, max_dte=35, short_delta=0.15, width_pct=0.05,
         min_credit_pct=0.25, risk_pct=0.02, vol_mult=1.5, max_volume_mult=1.5)


def fridays(start, n_weeks=8):
    d = pd.Timestamp(start).normalize()
    first = d + pd.Timedelta(days=(4 - d.weekday()) % 7)
    return [first + pd.Timedelta(weeks=k) for k in range(n_weeks)]


def strike_step(S):
    return 0.5 if S < 25 else 1.0 if S < 100 else 2.5 if S < 250 else 5.0


def strike_for_delta(S, T, vol, delta):
    """put strike whose |delta| = delta (BS)"""
    d1 = -norm.ppf(delta)
    return S * math.exp(-d1 * vol * math.sqrt(T) + (R + 0.5 * vol ** 2) * T)


def spread_value(S, Ks, Kl, T, vol):
    return float(bs_price(S, Ks, T, R, vol) - bs_price(S, Kl, T, R, vol))


def implied_spread_vol(value, S, Ks, Kl, T):
    """one flat vol that prices the spread at `value` (bisection); None if out of range"""
    lo, hi = 0.01, 5.0
    if not (spread_value(S, Ks, Kl, T, lo) <= value <= spread_value(S, Ks, Kl, T, hi)):
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if spread_value(S, Ks, Kl, T, mid) < value else (lo, mid)
    return (lo + hi) / 2


def entry_plan(ric, shock_date, today, px, earnings_dates, capital, iv=None):
    """
    [AG-12] concrete plan for one TAKE shock as of `today`'s close (orders go in the next session)
    px: AG-01 close/volume plus LOW_1 column; iv: the broker's ~30-day put IV if known (decimal)
    returns dict: action (ENTER TOMORROW / WAIT / DROP), why, expiry, strikes, min credit, contracts, est credit
    """
    shock, today = pd.Timestamp(shock_date).normalize(), pd.Timestamp(today).normalize()
    c = px["close"]
    days = len(c.loc[shock:today]) - 1                        # trading days since the shock
    fill_day = days + 1                                       # orders fill next session
    S = float(c.loc[today])
    low = float(px["low"].loc[shock:today].min())             # post-news low (shock day onward)
    r = c.pct_change()
    rv_pre = float(r.loc[:shock].iloc[-61:-1].std() * math.sqrt(252))
    vol = iv or max(rv_pre * P["vol_mult"], 0.15)
    out = {"ric": ric, "shock_date": shock.date(), "as_of": today.date(), "days_since_shock": days,
           "close": round(S, 2), "post_news_low": round(low, 2),
           "vol_used": round(vol, 3), "vol_source": "broker IV" if iv else f"{P['vol_mult']}x 60d realized (estimate)"}

    reasons = []
    if fill_day > P["entry_days"][1]:
        return {**out, "action": "DROP", "why": f"day +{fill_day} is past the day +3 entry limit (backtest: late entries lost)"}
    if days > 0 and float(px["low"].loc[today]) < float(px["low"].loc[shock:today].iloc[:-1].min()):
        reasons.append("made a new low today (selling not done)")
    vol20 = float(px["volume"].loc[:shock].iloc[-21:-1].mean())
    if days > 0 and np.isfinite(vol20) and float(px["volume"].loc[today]) > P["max_volume_mult"] * vol20:
        reasons.append(f"volume still {px['volume'].loc[today] / vol20:.1f}x normal")
    if reasons and fill_day < P["entry_days"][1]:
        return {**out, "action": "WAIT", "why": "; ".join(reasons) + f". Re-run tomorrow (last entry: day +{P['entry_days'][1]})"}
    if reasons:
        return {**out, "action": "DROP", "why": "; ".join(reasons) + " on the last allowed day"}

    entry = today + pd.offsets.BDay(1)
    exp = pick_expiry(fridays(entry), entry, P["target_dte"], P["min_dte"], P["max_dte"])
    ed = pd.to_datetime(pd.Series(earnings_dates))
    if exp is not None and ((ed >= entry) & (ed <= exp)).any():
        early = [f for f in fridays(entry) if P["min_dte"] <= (f - entry).days and not ((ed >= entry) & (ed <= f)).any()]
        exp = None if not early else min(early, key=lambda f: abs((f - entry).days - P["target_dte"]))
        if exp is not None and (exp - entry).days > P["max_dte"]:
            exp = None
    if exp is None:
        return {**out, "action": "DROP", "why": "no 14-35 day Friday expiry without earnings inside"}

    T = (exp - entry).days / 365
    step = strike_step(S)
    Ks = math.floor(min(strike_for_delta(S, T, vol, P["short_delta"]), low - 0.01) / step) * step
    Kl = (Ks - P["width"]) if P.get("width") else math.floor((Ks - P["width_pct"] * S) / step) * step
    width = Ks - Kl
    est = spread_value(S, Ks, Kl, T, vol)
    min_credit = P["min_credit_pct"] * width
    risk_pct = P["risk_usd"] / capital if P.get("risk_usd") else P["risk_pct"]
    n = position_size(capital, width, min_credit, risk_pct)   # sized at the minimum credit = most conservative
    return {**out, "action": "ENTER TOMORROW", "why": "inside day +1..+3, selling cooled, no new low",
            "expiry": exp.date(), "dte": (exp - entry).days, "short_strike": Ks, "long_strike": Kl, "width": width,
            "min_credit": round(min_credit, 2), "est_credit": round(est, 2),
            "credit_ok_est": bool(est >= min_credit), "contracts": n,
            "max_loss_usd": round(n * (width - min_credit) * 100),
            "loss_per_spread_usd": round((width - min_credit) * 100), "capital": capital}


def manage(pos, S, today, px_entry_close=None, mark=None):
    """
    [AG-13] HOLD or CLOSE one open bull put spread at today's close
    pos: dict ric, opened, expiry, short_strike, long_strike, credit, contracts
    S: stock close today; mark: broker's cost to buy the spread back (preferred); else BS estimate with the
    flat vol implied by the entry credit at the entry-day close (px_entry_close)
    """
    today = pd.Timestamp(today).normalize()
    opened, exp = pd.Timestamp(pos["opened"]).normalize(), pd.Timestamp(pos["expiry"]).normalize()
    Ks, Kl, credit = float(pos["short_strike"]), float(pos["long_strike"]), float(pos["credit"])
    width, n = Ks - Kl, int(pos["contracts"])
    dte_entry, held, left = (exp - opened).days, (today - opened).days, (exp - today).days
    out = {"ric": pos["ric"], "close": round(S, 2), "short": Ks, "long": Kl, "credit": credit, "days_left": left}

    if left <= 0:                                             # expiry day (or past it)
        if S >= Ks:
            return {**out, "action": "LET EXPIRE", "why": "above the short strike: both puts expire worthless, keep the credit",
                    "pnl_usd": round(credit * 100 * n)}
        if S <= Kl:
            return {**out, "action": "LET SETTLE (max loss)", "why": "below the long strike: both exercise, loss = width - credit",
                    "pnl_usd": round((credit - width) * 100 * n)}
        return {**out, "action": "CLOSE (buy back)", "why": "between the strikes on expiry day: buy back, don't take shares",
                "pnl_usd": round((credit - (Ks - S)) * 100 * n)}

    T = left / 365
    src = "broker"
    if mark is None or (isinstance(mark, float) and np.isnan(mark)):
        S0 = px_entry_close if px_entry_close else S
        v = implied_spread_vol(credit, S0, Ks, Kl, dte_entry / 365) or 0.4
        mark, src = spread_value(S, Ks, Kl, T, v), f"estimate (vol {v:.0%})"
    mark = max(float(mark), 0.0)
    captured = (credit - mark) / credit
    out.update({"cost_to_close": round(mark, 2), "mark_source": src, "captured": f"{captured:.0%}",
                "time_used": f"{held / dte_entry:.0%}", "pnl_now_usd": round((credit - mark) * 100 * n)})
    if should_take_profit(credit, mark, held, dte_entry, "dynamic", 0.50, 0.20, 0.75):
        why = "kept >= 75% of the credit" if captured >= 0.75 else "kept >= 50% and ahead of time by 20%"
        return {**out, "action": "CLOSE (take profit)", "why": why + " (STR-3.6a)"}
    note = "no stop-loss by design (the long put caps the loss)"
    if S < Ks:
        note = f"stock below the short strike; still HOLD, no stop (backtest). Max loss already capped at ${round((width - credit) * 100 * n)}"
    return {**out, "action": "HOLD", "why": note}


def index_plan(today, spy_px, capital, risk_usd, width=1.0, iv=None, vix=None, open_positions=0, max_open=1,
               short_delta=0.15, target_dte=21, min_credit_pct=0.10, _allow_deprecated=False):
    """
    [AG-17] the index trade (the benchmark that beat the single-stock shocks, scripts/spy_benchmark.py):
    a bull put spread on XSP (S&P 500 / 10, cash-settled, European: no early assignment, no shares at expiry)
    spy_px: SPY daily closes (XSP trades ~0.3% above SPY; the broker's XSP quote decides)
    iv: broker's ~0.15-delta put IV if known; else VIX/100 + 0.03 (puts below the money trade above VIX),
    else 1.3x 20-day realized vol. Returns dict with action ENTER TOMORROW / HOLD OFF and the order details.
    """
    if not _allow_deprecated:
        raise RuntimeError("index_plan (AG-17) is DEPRECATED: the weekly XSP 0.15-delta rule failed "
                           "strategy/preregistered_spy.md. Use scripts/index_signal.py. Pass _allow_deprecated=True "
                           "for research only.")
    today = pd.Timestamp(today).normalize()
    c = spy_px["close"].loc[:today]
    S = float(c.iloc[-1])
    rv = float(c.pct_change().iloc[-21:].std() * math.sqrt(252))
    if iv:
        vol, src = iv, "broker IV"
    elif vix:
        vol, src = vix / 100 + 0.03, f"VIX {vix:.1f} + 3 pts (estimate)"
    else:
        vol, src = max(1.3 * rv, 0.12), "1.3x 20-day realized (estimate)"
    out = {"ric": "XSP", "as_of": today.date(), "close_spy": round(S, 2), "vol_used": round(vol, 3), "vol_source": src}
    if open_positions >= max_open:
        return {**out, "action": "HOLD OFF", "why": f"{open_positions} spread(s) already open (max {max_open})"}
    entry = today + pd.offsets.BDay(1)
    exp = pick_expiry(fridays(entry), entry, target_dte, 14, 35)
    T = (exp - entry).days / 365
    Ks = math.floor(strike_for_delta(S, T, vol, short_delta))
    Kl = Ks - width
    est = spread_value(S, Ks, Kl, T, vol)
    min_credit = round(min_credit_pct * width, 2)
    n = position_size(capital, width, min_credit, risk_usd / capital)
    return {**out, "action": "ENTER TOMORROW" if n > 0 else "HOLD OFF",
            "why": "weekly index spread" if n > 0 else f"one spread risks ${round((width - min_credit) * 100)} > ${risk_usd:.0f} allowed",
            "expiry": exp.date(), "dte": (exp - entry).days, "short_strike": Ks, "long_strike": Kl, "width": width,
            "est_credit": round(est, 2), "min_credit": min_credit, "contracts": n,
            "loss_per_spread_usd": round((width - min_credit) * 100), "capital": capital}
