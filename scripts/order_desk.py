"""
Order desk (AG-37..AG-40): order tickets for the put spreads, the resting take-profit price, and the fill log.
Nothing here sends orders: it prints a ticket, a teammate approves it and places it in the broker by hand.

Quotes come from the broker's chain at order time, per share: --sb/--sa short put bid/ask, --lb/--la long put bid/ask.

    # open: limit ladder from the mid toward the worst price, size from the 10%-of-capital risk rule
    py -3.12 -u scripts/order_desk.py open  --underlying XSP --expiry 2026-12-11 --short 655 --long 650 \
        --sb 0.62 --sa 0.66 --lb 0.40 --la 0.44 --capital 5000
    # close (expiry-day buy-back, or a take-profit that did not rest): same ladder for a debit
    py -3.12 -u scripts/order_desk.py close --underlying XSP --expiry 2026-12-11 --short 655 --long 650 \
        --sb 0.08 --sa 0.10 --lb 0.03 --la 0.05 --qty 1
    # resting take-profit (buy-to-close limit) for every open row of data/results/agent/positions.csv, today
    py -3.12 -u scripts/order_desk.py tp
    # after the broker fills: log it (open fills also add the row to positions.csv; close fills mark it closed)
    py -3.12 -u scripts/order_desk.py fill  --side open --underlying XSP --expiry 2026-12-11 --short 655 \
        --long 650 --qty 1 --sb 0.62 --sa 0.66 --lb 0.40 --la 0.44 --limit 0.21 --fill 0.21 --step 2
    # what our real fills are worth vs the backtest (data/results/index_branch/fill_sensitivity.csv)
    py -3.12 -u scripts/order_desk.py report --capital 5000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from src.agent import execution as X  # noqa: E402

OUT = Path("data/results/agent")
POS = OUT / "positions.csv"
POS_COLS = ["ric", "opened", "expiry", "short_strike", "long_strike", "credit", "contracts", "mark", "status"]


def trade_id(a, opened=None):
    d = pd.Timestamp(opened or pd.Timestamp.today()).strftime("%Y%m%d")
    return f"{a.underlying}-{pd.Timestamp(a.expiry):%Y%m%d}-{a.short:g}-{a.long:g}-{d}"


def save_ticket(tid, side, text):
    p = OUT / "tickets" / f"{tid}_{side}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text + "\n", encoding="utf-8")
    return p


SPY_WARNING = ("  WARNING: SPY options are American-style and can be ASSIGNED EARLY (shares, pin risk at expiry). "
               "Use XSP (European, cash-settled) unless a teammate decided otherwise.")


def warn_underlying(a):
    """[AG-52] the default underlying is XSP; SPY gets an early-assignment warning"""
    if str(getattr(a, "underlying", "") or "").upper() == "SPY":
        print(SPY_WARNING)


def cmd_open(a):
    warn_underlying(a)
    lad = X.ladder(a.sb, a.sa, a.lb, a.la, "open", a.steps, a.wait, a.max_concession, a.tick, a.min_credit)
    width = a.short - a.long
    if a.qty:
        qty = a.qty
    elif a.capital and lad["steps"]:
        worst = lad["steps"][-1]["net"]                  # size on the worst credit we would accept
        qty = X.qty_for_risk(a.capital, width, worst, a.risk_pct)
    else:
        qty = 1
    tid = trade_id(a)
    print(X.ticket(lad, a.underlying, a.expiry, a.short, a.long, max(qty, 1), tid))
    if a.capital and lad["steps"]:
        loss = (width - lad["steps"][-1]["net"]) * 100
        print(f"  size: max loss ${loss:,.0f} per spread vs budget ${a.capital * a.risk_pct:,.0f} "
              f"({a.risk_pct:.0%} of ${a.capital:,.0f}) -> {qty} contract(s)")
        if qty < 1:
            print("  ACCOUNT TOO SMALL for this spread under the 10% rule: do not place it (paper-trade it).")
    if not lad["skip"] and qty >= 1:
        print(f"  ticket saved: {save_ticket(tid, 'open', X.ticket(lad, a.underlying, a.expiry, a.short, a.long, max(qty, 1), tid))}")


def open_row(a):
    """the open positions.csv row for this spread, or None"""
    if not POS.exists():
        return None
    pos = pd.read_csv(POS)
    key = ((pos["ric"].astype(str) == a.underlying) & (pd.to_datetime(pos["expiry"]) == pd.Timestamp(a.expiry))
           & (pos["short_strike"].astype(float) == a.short) & (pos["long_strike"].astype(float) == a.long)
           & (pos["status"].fillna("open").str.lower() == "open"))
    return pos[key].iloc[0].to_dict() if key.any() else None


def cmd_close(a):
    warn_underlying(a)
    lad = X.ladder(a.sb, a.sa, a.lb, a.la, "close", a.steps, a.wait, a.max_concession, a.tick, max_debit=a.max_debit)
    row = open_row(a)
    tid = a.trade_id or trade_id(a, row["opened"] if row else None)
    text = X.ticket(lad, a.underlying, a.expiry, a.short, a.long, a.qty, tid)
    print(text)
    print(f"  ticket saved: {save_ticket(tid, 'close', text)}")


def cmd_tp(a):
    today = pd.Timestamp(a.today or pd.Timestamp.today()).normalize()
    if a.credit is not None:
        rows = [{"ric": a.underlying or "?", "opened": a.opened, "expiry": a.expiry, "short_strike": a.short,
                 "long_strike": a.long, "credit": a.credit, "contracts": a.qty or 1}]
    else:
        if not POS.exists():
            print(f"no {POS}: log the opening fill first (order_desk.py fill --side open ...)")
            return
        pos = pd.read_csv(POS)
        rows = pos[pos["status"].fillna("open").str.lower() == "open"].to_dict("records")
        if not rows:
            print("no open positions")
            return
    print(f"Resting take-profit orders for {today.date()} (buy-to-close LIMIT, DAY; re-place each morning, "
          f"the price rises as time passes):")
    q = {k: getattr(a, k, None) for k in ("sb", "sa", "lb", "la")} if a.credit is not None else {}
    for r in rows:
        if str(r.get("ric", "")).upper() == "SPY":
            print(SPY_WARNING)
        tp = X.resting_take_profit(float(r["credit"]), r["opened"], r["expiry"], today, tick=a.tick, **q)
        tomorrow = X.take_profit_price(float(r["credit"]), r["opened"], r["expiry"], today + pd.Timedelta(days=1), a.tick)
        legs = f"{r['ric']} {float(r['short_strike']):g}/{float(r['long_strike']):g} exp {pd.Timestamp(r['expiry']).date()}"
        if tp is None:
            print(f"  {legs}: expiry day or later -> no take-profit; use trade_desk.py manage (expire / buy back)")
            continue
        n = int(r["contracts"])
        if tp["action"] == "hold":
            print(f"  {legs}: NO RESTING ORDER - {tp['warning']}")
            continue
        verb = "BUY TO CLOSE NOW (take-profit already reached at the natural price):" if tp["action"] == "close now" \
            else "BUY TO CLOSE"
        print(f"  {legs}: {verb} {n} at {tp['limit']:.2f} or less (keeps {tp['keep_pct']:.0%} of the "
              f"{float(r['credit']):.2f} credit = ${(float(r['credit']) - tp['limit']) * 100 * n:,.0f}; "
              f"time used {tp['time_used']:.0%})" + (f"; tomorrow {tomorrow['limit']:.2f}" if tomorrow and tomorrow['limit'] >= 0.01 else ""))
        if tp["warning"]:
            print(f"    {tp['warning']}")
    print("  A TEAMMATE MUST APPROVE each order. Nothing here sends orders.")


def cmd_fill(a):
    row0 = open_row(a) if a.side == "close" else None
    tid = a.trade_id or trade_id(a, a.opened or (row0["opened"] if row0 else None))
    row = X.log_fill(tid, a.side, a.underlying, a.expiry, a.short, a.long, a.qty, a.sb, a.sa, a.lb, a.la,
                     a.limit, a.fill, a.step)
    print(f"logged {a.side} fill {a.fill:.2f} (mid {row['mid']:.2f}, worst {row['natural']:.2f}) -> "
          f"fill quality {row['fill_q']:+.2f} (0 = mid, 1 = worst) in {X.FILLS}")
    if a.no_position:
        return
    pos = pd.read_csv(POS) if POS.exists() else pd.DataFrame(columns=POS_COLS)
    key = ((pos["ric"].astype(str) == a.underlying) & (pd.to_datetime(pos["expiry"]) == pd.Timestamp(a.expiry))
           & (pos["short_strike"].astype(float) == a.short) & (pos["long_strike"].astype(float) == a.long)
           & (pos["status"].fillna("open").str.lower() == "open"))
    if a.side == "open":
        if key.any():
            print(f"  {POS}: an open row for this spread already exists, not added again")
            return
        new = {"ric": a.underlying, "opened": str(pd.Timestamp(a.opened or pd.Timestamp.today()).date()),
               "expiry": str(pd.Timestamp(a.expiry).date()), "short_strike": a.short, "long_strike": a.long,
               "credit": a.fill, "contracts": a.qty, "mark": "", "status": "open"}
        pos = pd.concat([pos, pd.DataFrame([new])], ignore_index=True)[POS_COLS]
        POS.parent.mkdir(parents=True, exist_ok=True)
        pos.to_csv(POS, index=False)
        print(f"  added the position to {POS} (credit {a.fill:.2f} x {a.qty})")
    else:
        if key.any():
            pos.loc[key, "status"] = "closed"
            pos.to_csv(POS, index=False)
            print(f"  marked closed in {POS}")
        else:
            print(f"  no open row for this spread in {POS}")


def cmd_report(a):
    r = X.report(a.capital)
    if not r.get("n"):
        print(f"no fills logged yet in {X.FILLS}")
        return
    print(f"{r['n']} fills ({r['n_open']} opens, {r['n_close']} closes), average step {r['avg_step']:.1f}")
    print(f"  realized fill quality {r['fill_q']:+.2f} (0 = mid, 1 = worst; backtests assumed 1.00)"
          + (f" | opens {r['fill_q_open']:+.2f}" if r["fill_q_open"] is not None else "")
          + (f" | closes {r['fill_q_close']:+.2f}" if r["fill_q_close"] is not None else ""))
    print(f"  at ${r['capital_used']:,.0f} the backtest earns ${r['per_year_$']:,.0f}/yr ({r['per_year_%']:+.2%}) at "
          f"this fill quality, vs ${r['at_worst_$']:,.0f}/yr at worst fills and ${r['at_mid_$']:,.0f}/yr at mid "
          f"(data/results/index_branch/fill_sensitivity.csv)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def legs(p, quotes=True):
        p.add_argument("--underlying", default="XSP")
        p.add_argument("--expiry", required=True)
        p.add_argument("--short", type=float, required=True, help="short put strike")
        p.add_argument("--long", type=float, required=True, help="long put strike")
        if quotes:
            for k, h in (("sb", "short bid"), ("sa", "short ask"), ("lb", "long bid"), ("la", "long ask")):
                p.add_argument(f"--{k}", type=float, required=True, help=f"{h} per share")

    def ladder_args(p):
        p.add_argument("--steps", type=int, default=4)
        p.add_argument("--wait", type=int, default=5, help="minutes between steps")
        p.add_argument("--max-concession", type=float, default=0.5, help="0 = mid only, 1 = up to the worst price")
        p.add_argument("--tick", type=float, default=0.01)

    po = sub.add_parser("open", help="ticket to sell a put spread")
    legs(po)
    ladder_args(po)
    po.add_argument("--qty", type=int)
    po.add_argument("--capital", type=float, help="size from the risk rule instead of --qty")
    po.add_argument("--risk-pct", type=float, default=0.10)
    po.add_argument("--min-credit", type=float, default=0.0, help="skip steps whose credit after fees is below this")
    po.set_defaults(fn=cmd_open)

    pc = sub.add_parser("close", help="ticket to buy a put spread back")
    legs(pc)
    ladder_args(pc)
    pc.add_argument("--qty", type=int, required=True)
    pc.add_argument("--max-debit", type=float, help="never pay more than this per share")
    pc.add_argument("--trade-id")
    pc.set_defaults(fn=cmd_close)

    pt = sub.add_parser("tp", help="resting take-profit price for each open position (or one given here)")
    pt.add_argument("--today")
    pt.add_argument("--tick", type=float, default=0.01)
    pt.add_argument("--credit", type=float, help="one trade instead of positions.csv: entry credit per share")
    pt.add_argument("--opened")
    pt.add_argument("--expiry")
    pt.add_argument("--underlying")
    pt.add_argument("--short", type=float)
    pt.add_argument("--long", type=float)
    pt.add_argument("--qty", type=int)
    for k, h in (("sb", "short bid"), ("sa", "short ask"), ("lb", "long bid"), ("la", "long ask")):
        pt.add_argument(f"--{k}", type=float, help=f"optional, with --credit: today's {h} (wide-market check)")
    pt.set_defaults(fn=cmd_tp)

    pf = sub.add_parser("fill", help="log a broker fill")
    pf.add_argument("--side", choices=["open", "close"], required=True)
    legs(pf)
    pf.add_argument("--qty", type=int, required=True)
    pf.add_argument("--limit", type=float, required=True, help="the limit that filled, per share")
    pf.add_argument("--fill", type=float, required=True, help="fill price per share, before commissions")
    pf.add_argument("--step", type=int, required=True, help="ladder step that filled")
    pf.add_argument("--opened", help="date of the opening fill (default today)")
    pf.add_argument("--trade-id")
    pf.add_argument("--no-position", action="store_true", help="do not touch positions.csv")
    pf.set_defaults(fn=cmd_fill)

    pr = sub.add_parser("report", help="realized fill quality and what it is worth")
    pr.add_argument("--capital", type=float, required=True)
    pr.set_defaults(fn=cmd_report)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
