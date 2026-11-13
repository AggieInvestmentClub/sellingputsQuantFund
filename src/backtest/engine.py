"""
One bull put spread per event, using the STR rule functions in strategy/strategy_logic.py.
Daily (end-of-day) simulation: decide at the close, fill at the close with a modeled half spread.
"""
import numpy as np
import pandas as pd

from strategy.strategy_logic import (credit_ok, earnings_clear, entry_ready, pick_expiry,
                                     pick_long_strike, pick_short_strike, realized_vol,
                                     should_stop, should_take_profit)

# trade parameters (spec section 3 / 4). Event filters (LLM verdict, category, shock size,
# market cap) are applied afterwards, see src/backtest/grid.py
BASELINE = {
    "short_delta": 0.20, "long_method": "width", "width_pct": 0.05, "long_delta": 0.10,
    "min_credit_pct": 0.25, "target_dte": 21, "iv_rv_min": 1.5, "max_volume_mult": 1.5,
    "max_wait": 3, "tp_mode": "dynamic", "outpace_margin": 0.20, "fixed_target": 0.50,
    "stop_rules": ["credit_multiple"], "credit_multiple": 2.0, "max_loss_pct": 0.50,
    "min_price": 20.0, "commission": 0.65,
    "close_between_at_expiry": True,   # stock between the strikes on expiry day: buy the spread back
}                                      # at the close instead of being assigned on the short put


def _settle(S, short_k, long_k):
    return min(max(short_k - S, 0.0) - max(long_k - S, 0.0), short_k - long_k)


def simulate_event(event, stock, pricer, earnings_dates, params):
    """
    [BT-04] the trade for one shock under one parameter set: wait for the entry trigger (STR-3.5b),
    build the spread (STR-1.4b, 3.2a, 3.3a, 3.4b), check earnings (STR-2.2a), then mark it daily
    until take-profit (STR-3.6a), stop (STR-3.7a) or expiry (settled at intrinsic)

    event: row with event_id, ric, date. stock: BT-01 bars. pricer: BT-02 / BT-03 (start() already called)
    returns dict: status ("traded" or the skip reason) and, if traded, entry / exit dates, strikes,
    credit, exit_cost, pnl, max_loss and ror (P&L / max loss, all per share, after commissions)
    """
    p = {**BASELINE, **params}
    shock = pd.Timestamp(event["date"]).normalize()
    idx = stock.index
    if shock not in idx:
        return {"status": "no price on shock day"}
    s = idx.get_loc(shock)
    if stock["close"].iloc[s] < p["min_price"]:
        return {"status": "price below floor"}
    avg_vol = stock["volume"].iloc[max(0, s - 20):s].mean()
    lows = stock["low"].to_numpy()

    entry = None
    for k in range(1, p["max_wait"] + 2):
        if s + k >= len(idx):
            return {"status": "no data after shock"}
        d = idx[s + k]
        post_low = lows[s:s + k + 1].min()
        made_new_low = lows[s + k] < lows[s:s + k].min()
        rv = realized_vol(stock["ret"], shock, d)
        iv = pricer.atm_iv(event["event_id"], d)
        if iv is None:
            if k > p["max_wait"]:
                return {"status": "no entry signal"}
            continue
        go = entry_ready(k, iv, rv, stock["volume"].iloc[s + k], avg_vol, made_new_low,
                         max_wait=p["max_wait"], iv_rv_min=p["iv_rv_min"],
                         max_volume_mult=p["max_volume_mult"])
        if go == "skip":
            return {"status": "no entry signal"}
        if go != "enter":
            continue
        exp = pick_expiry(pricer.expiries(event["event_id"], d), d, target_dte=p["target_dte"])
        if exp is None:
            continue
        chain = pricer.chain(event["event_id"], d, exp)
        chain = chain[(chain["bid"] > 0) & (chain["volume"] > 0)]
        S = float(stock["close"].iloc[s + k])
        short = pick_short_strike(chain, post_low, p["short_delta"])
        if short is None:
            continue
        long = pick_long_strike(chain, short["strike"], S, p["long_method"], p["width_pct"], p["long_delta"])
        if long is None:
            continue
        credit = float(short["bid"] - long["ask"])
        if not credit_ok(credit, short["strike"], long["strike"], p["min_credit_pct"]):
            continue
        if not earnings_clear(shock, d, exp, earnings_dates):
            return {"status": "earnings in window"}
        entry = dict(k=k, date=d, expiry=pd.Timestamp(exp), S=S, post_low=post_low, iv=iv, rv=rv,
                     short_k=float(short["strike"]), long_k=float(long["strike"]), credit=credit)
        break
    if entry is None:
        return {"status": "no entry signal"}

    width = entry["short_k"] - entry["long_k"]
    dte = max((entry["expiry"] - entry["date"]).days, 1)
    j = s + entry["k"]
    exit_date, exit_cost, reason = None, None, None
    while j + 1 < len(idx) and idx[j + 1] <= entry["expiry"]:
        j += 1
        t = idx[j]
        S_t = float(stock["close"].iloc[j])
        if t == entry["expiry"]:
            break
        mid, bid, ask, delta = pricer.leg_marks(event["event_id"], t, entry["expiry"],
                                                [entry["short_k"], entry["long_k"]])
        cost = float(min(max(ask[0] - bid[1], 0.0), width))
        days = (t - entry["date"]).days
        if should_take_profit(entry["credit"], cost, days, dte, p["tp_mode"],
                              outpace_margin=p["outpace_margin"], fixed_target=p["fixed_target"]):
            exit_date, exit_cost, reason = t, cost, "take profit"
            break
        stop = should_stop(entry["credit"], cost, width, S_t, entry["short_k"], entry["post_low"],
                           delta[0], rules=tuple(p["stop_rules"]), credit_multiple=p["credit_multiple"],
                           max_loss_pct=p["max_loss_pct"])
        if stop:
            exit_date, exit_cost, reason = t, cost, f"stop: {stop}"
            break
    if exit_date is None:
        if idx[j] < entry["expiry"] and j + 1 >= len(idx):
            return {"status": "open at end of data"}
        S_x = float(stock["close"].iloc[j])
        exit_date, reason = idx[j], "expiry"
        exit_cost = _settle(S_x, entry["short_k"], entry["long_k"])
        if (p["close_between_at_expiry"] and idx[j] == entry["expiry"]
                and entry["long_k"] < S_x < entry["short_k"]):
            # short put in the money, long put worthless: closing avoids holding 100 shares per
            # contract over the weekend. Pay at least intrinsic, plus the modeled spread when quoted
            try:
                _, bid, ask, _ = pricer.leg_marks(event["event_id"], idx[j], entry["expiry"],
                                                  [entry["short_k"], entry["long_k"]])
                quoted = float(ask[0] - bid[1])
            except Exception:
                quoted = np.nan
            exit_cost = float(min(max(quoted, exit_cost) if np.isfinite(quoted) else exit_cost, width))
            reason = "closed at expiry (between strikes)"

    legs = 2 if reason == "expiry" else 4  # settled at expiry: no closing trades
    fees = p["commission"] * legs / 100
    pnl = entry["credit"] - exit_cost - fees
    max_loss = width - entry["credit"] + fees
    return {"status": "traded", "entry_date": entry["date"], "entry_lag": entry["k"],
            "expiry": entry["expiry"], "exit_date": exit_date, "exit_reason": reason,
            "S_entry": entry["S"], "short_k": entry["short_k"], "long_k": entry["long_k"],
            "width": width, "credit": entry["credit"], "exit_cost": exit_cost, "pnl": pnl,
            "max_loss": max_loss, "ror": pnl / max_loss if max_loss > 0 else np.nan,
            "iv_entry": entry["iv"], "rv_entry": entry["rv"], "days_held": (exit_date - entry["date"]).days}
