"""
AG-37..AG-40: the order-placing (execution) agent for the put credit spreads.

Every backtest filled at the worst price (sell at the bid, buy at the ask). This module works limit orders from
the mid-price toward that worst ("natural") price in small steps, rests the take-profit order in advance, and logs
every real fill so the club measures what its fills are worth (data/results/index_branch/fill_sensitivity.csv).

  ladder()             [AG-37] limit prices, mid -> natural, capped at max_concession, rounded to the tick
  take_profit_price()  [AG-38] buy-to-close debit at which the dynamic take-profit (STR-3.6a) fires today
  log_fill / report    [AG-39] fills.csv, realized FILL, and its $/yr from the fill sensitivity
  ticket()             [AG-40] plain-text combo-order ticket a teammate follows in the broker

Nothing here sends orders: a teammate approves and places every one by hand.
"""
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.strategy_logic import should_take_profit

FEE = 0.65 / 100        # $0.65 per contract per leg, per share (same as scripts/index_branch.py FEE)
TP = dict(floor=0.50, outpace_margin=0.20, final_target=0.75)    # STR-3.6a settings the backtests used
FILLS = Path("data/results/agent/fills.csv")
FILL_SENS = Path("data/results/index_branch/fill_sensitivity.csv")
FILL_COLS = ["trade_id", "logged_at", "side", "underlying", "expiry", "short_strike", "long_strike", "qty",
             "sb", "sa", "lb", "la", "mid", "natural", "limit", "fill", "step", "fill_q"]
EPS = 1e-9


# ---------------------------------------------------------------- AG-37 ladder
def spread_quotes(sb, sa, lb, la, side="open"):
    """mid and natural price of the 2-leg put spread, per share.
    open  (sell the spread for a credit): mid = short mid - long mid, natural = short bid - long ask (worst)
    close (buy it back for a debit):      mid = short mid - long mid, natural = short ask - long bid (worst)"""
    if side not in ("open", "close"):
        raise ValueError(f"side must be open or close, got {side}")
    for name, (b, a) in {"short": (sb, sa), "long": (lb, la)}.items():
        if b < 0 or a < b:
            raise ValueError(f"{name} leg: need 0 <= bid <= ask, got {b} / {a}")
    mid = (sb + sa) / 2 - (lb + la) / 2
    natural = sb - la if side == "open" else sa - lb
    return mid, natural


def _round(x, tick, up):
    k = x / tick
    return round((math.ceil(k - EPS) if up else math.floor(k + EPS)) * tick, 10)


def ladder(sb, sa, lb, la, side="open", steps=4, wait_min=5, max_concession=0.5, tick=0.01, min_credit=0.0,
           max_debit=None):
    """
    [AG-37] limit prices for one spread order, from the mid toward the natural price.

    Step i of n concedes (i / (n - 1)) x max_concession of the way from mid to natural (0.5 = stop halfway).
    Prices round toward the mid (a credit rounds up, a debit down) so rounding never concedes more than allowed;
    steps that round to the same price are merged. Opens drop steps whose credit minus fees (2 legs x $0.65) is
    below min_credit, and are skipped if none is left. Closes never go below $0.00 or above max_debit.
    Returns {"side", "mid", "natural", "steps": [{"step", "at_min", "limit", "net"}], "cancel_at_min", "skip"}.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not 0 <= max_concession <= 1:
        raise ValueError("max_concession must be between 0 and 1")
    mid, natural = spread_quotes(sb, sa, lb, la, side)
    gap = (mid - natural) if side == "open" else (natural - mid)       # >= 0: what the worst price costs us
    gap = max(gap, 0.0)
    out, seen = [], set()
    for i in range(steps):
        share = (i / (steps - 1)) * max_concession if steps > 1 else 0.0
        if side == "open":
            px = _round(mid - share * gap, tick, up=True)
            px = min(px, _round(mid, tick, up=True))
            net = px - 2 * FEE
        else:
            px = max(_round(mid + share * gap, tick, up=False), 0.0)
            if max_debit is not None:
                px = min(px, _round(max_debit, tick, up=False))
            net = px + 2 * FEE
        if px in seen:
            continue
        seen.add(px)
        out.append({"step": len(out) + 1, "at_min": i * wait_min, "limit": round(px, 2), "net": round(net, 4)})
    skip = None
    if side == "open":
        out = [s for s in out if s["net"] > max(min_credit, 0.0) + EPS]
        if not out:
            skip = (f"credit too small: mid {mid:.3f}, best limit minus fees <= {max(min_credit, 0):.2f}"
                    " -> no order")
        for k, s in enumerate(out):
            s["step"] = k + 1
    cancel = (out[-1]["at_min"] + wait_min) if out else 0
    return {"side": side, "mid": round(mid, 4), "natural": round(natural, 4), "steps": out,
            "cancel_at_min": cancel, "skip": skip}


def qty_for_risk(capital, width, credit, risk_pct=0.10):
    """contracts so that the max loss (width - credit) x 100 x qty stays within capital x risk_pct (AG-28d rule)"""
    loss = (width - credit) * 100
    if loss <= 0:
        raise ValueError("credit must be below the width")
    return int(math.floor(capital * risk_pct / loss + EPS))


# ---------------------------------------------------------------- AG-38 take-profit resting order
def take_profit_price(credit, opened, expiry, today, tick=0.01, **tp):
    """
    [AG-38] the buy-to-close debit at which the dynamic take-profit (STR-3.6a) fires TODAY.

    The rule closes when captured = (credit - cost) / credit is >= 50% AND (>= 75% OR captured - time_used >= 20%),
    time_used = days held / days at entry. So today it fires at captured >= clip(time_used + 0.20, 0.50, 0.75),
    i.e. cost <= credit x (1 - that). Rounded DOWN to the tick so a fill at the limit always satisfies the rule.
    The threshold rises every day: re-place (or modify) the resting order each morning. None on/after expiry day.
    """
    p = {**TP, **tp}
    opened, expiry, today = (pd.Timestamp(x).normalize() for x in (opened, expiry, today))
    dte, held = (expiry - opened).days, (today - opened).days
    if dte <= 0 or today >= expiry:
        return None
    elapsed = held / dte
    need = min(max(elapsed + p["outpace_margin"], p["floor"]), p["final_target"])
    px = _round(credit * (1 - need), tick, up=False)
    while px > 0 and not should_take_profit(credit, px, held, dte, "dynamic", p["floor"], p["outpace_margin"],
                                            p["final_target"]):
        px = round(px - tick, 10)
    px = max(px, 0.0)
    return {"limit": round(px, 2), "keep_pct": need, "time_used": elapsed, "days_held": held, "dte_entry": dte}


# ---------------------------------------------------------------- AG-39 fills
def fill_quality(fill, mid, natural, side="open"):
    """[AG-39] realized FILL: 0 = filled at the mid, 1 = at the natural (worst) price, < 0 = better than mid.
    Same scale as FILL in scripts/index_branch.py (share of the half bid/ask spread paid)."""
    gap = (mid - natural) if side == "open" else (natural - mid)
    if abs(gap) < EPS:
        return 0.0
    return (mid - fill) / gap if side == "open" else (fill - mid) / gap


def log_fill(trade_id, side, underlying, expiry, short_strike, long_strike, qty, sb, sa, lb, la, limit, fill,
             step, path=FILLS, when=None):
    """append one fill (gross price per share, before commissions) to fills.csv and return the row"""
    mid, natural = spread_quotes(sb, sa, lb, la, side)
    row = {"trade_id": trade_id, "logged_at": (when or datetime.now()).strftime("%Y-%m-%d %H:%M"), "side": side,
           "underlying": underlying, "expiry": str(pd.Timestamp(expiry).date()), "short_strike": short_strike,
           "long_strike": long_strike, "qty": int(qty), "sb": sb, "sa": sa, "lb": lb, "la": la,
           "mid": round(mid, 4), "natural": round(natural, 4), "limit": limit, "fill": fill, "step": step,
           "fill_q": round(fill_quality(fill, mid, natural, side), 4)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=FILL_COLS)
    pd.concat([old, pd.DataFrame([row])[FILL_COLS]], ignore_index=True).to_csv(path, index=False)
    return row


def value_of_fill(fill_q, capital, sens_path=FILL_SENS):
    """$/yr and %/yr the fill sensitivity backtest gives at this realized FILL (linear between its 0/0.25/0.5/1
    rows; clipped to 0..1; the nearest capital in the file if this one was not run)"""
    s = pd.read_csv(sens_path)
    caps = s["capital"].unique()
    cap = caps[np.argmin(np.abs(caps - capital))]
    g = s[s["capital"] == cap].sort_values("fill")
    f = float(np.clip(fill_q, 0, 1))
    return {"capital_used": float(cap), "fill": f,
            "per_year_$": float(np.interp(f, g["fill"], g["per_year_$"])),
            "per_year_%": float(np.interp(f, g["fill"], g["per_year_%"])),
            "at_worst_$": float(g.loc[g["fill"].idxmax(), "per_year_$"]),
            "at_mid_$": float(g.loc[g["fill"].idxmin(), "per_year_$"])}


def report(capital, path=FILLS, sens_path=FILL_SENS):
    """[AG-39] mean realized FILL (all, opens, closes) and what it is worth per year at this capital"""
    path = Path(path)
    if not path.exists() or pd.read_csv(path).empty:
        return {"n": 0}
    f = pd.read_csv(path)
    out = {"n": len(f), "n_open": int((f["side"] == "open").sum()), "n_close": int((f["side"] == "close").sum()),
           "fill_q": float(f["fill_q"].mean()),
           "fill_q_open": float(f.loc[f["side"] == "open", "fill_q"].mean()) if (f["side"] == "open").any() else None,
           "fill_q_close": float(f.loc[f["side"] == "close", "fill_q"].mean()) if (f["side"] == "close").any() else None,
           "avg_step": float(f["step"].mean())}
    out.update(value_of_fill(out["fill_q"], capital, sens_path))
    return out


# ---------------------------------------------------------------- AG-40 ticket
def ticket(lad, underlying, expiry, short_strike, long_strike, qty, trade_id=None, start="right after the open"):
    """[AG-40] plain-text order ticket for a teammate: combo order, legs, size, limit per step, timing, cancel"""
    side = lad["side"]
    exp = pd.Timestamp(expiry).strftime("%Y-%m-%d (%a)")
    lines = [f"ORDER TICKET {trade_id or ''}".rstrip(),
             f"  {'OPEN: SELL' if side == 'open' else 'CLOSE: BUY'} {qty} x {underlying} put spread "
             f"{short_strike:g}/{long_strike:g}, expiry {exp}, as ONE combo (vertical) order, LIMIT, DAY",
             f"    leg 1: {'SELL' if side == 'open' else 'BUY'} {qty} {underlying} {exp[:10]} {short_strike:g} put",
             f"    leg 2: {'BUY' if side == 'open' else 'SELL'} {qty} {underlying} {exp[:10]} {long_strike:g} put",
             f"  quotes now: mid {lad['mid']:.2f}, worst (natural) {lad['natural']:.2f} per share"]
    if lad["skip"]:
        lines.append(f"  NO ORDER: {lad['skip']}")
        return "\n".join(lines)
    word = "credit" if side == "open" else "debit"
    lines.append(f"  start {start}; if not filled, replace the limit at each step:")
    for s in lad["steps"]:
        lines.append(f"    step {s['step']}  t+{s['at_min']:>2d} min  limit {word} {s['limit']:.2f}"
                     f"  (= ${s['limit'] * 100 * qty:,.0f}; after fees ${s['net'] * 100 * qty:,.0f})")
    lines.append(f"  cancel at t+{lad['cancel_at_min']} min if still unfilled; never go past the last step "
                 f"(it is the most we give up vs the mid). Quotes moved > 10%? Re-run the ladder.")
    lines.append("  After the fill: scripts/order_desk.py fill ... (logs the fill quality).")
    lines.append("  A TEAMMATE MUST APPROVE THIS ORDER BEFORE IT IS PLACED. Nothing here sends orders.")
    return "\n".join(lines)
