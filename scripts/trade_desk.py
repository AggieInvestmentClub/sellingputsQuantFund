"""
Trade desk (AG-12 / AG-13): turns the agent's TAKEs into orders and tells you when to close.
Run every trading day AFTER scripts/live_scan.py (which updates prices), after 15:00 Matamoros.

    py -3.12 -u scripts/trade_desk.py plan   --capital 25000      # entry plan for every TAKE of the last 3 days
    py -3.12 -u scripts/trade_desk.py plan   --capital 25000 --ric TSLA.OQ --shock-date 2024-12-20 --today 2024-12-23
    py -3.12 -u scripts/trade_desk.py manage                      # HOLD / CLOSE for each open spread

Open positions live in data/results/agent/positions.csv (you add a row when the broker fills):
    ric,opened,expiry,short_strike,long_strike,credit,contracts,mark,status
    TSLA.OQ,2024-12-23,2025-01-17,380,360,4.10,1,,open
  credit = per share received (e.g. 4.10 = $410 per spread); mark = optional: the broker's current
  cost to buy the spread back (fill it in for the exact answer; empty = Black-Scholes estimate);
  status = open / closed (closed rows are ignored).
Nothing here sends orders: a teammate checks the broker's chain and approves each one.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
import pandas as pd

from src.agent.evidence import fname
from src.agent.trade_desk import entry_plan, index_plan, manage

RAW, OUT = Path("data/raw"), Path("data/results/agent")


def prices(ric):
    p = RAW / "prices" / "daily" / f"{fname(ric)}_adj.parquet"
    if not p.exists():
        return None
    px = pd.read_parquet(p)
    if "date" in px.columns:
        px = px.set_index("date")
    px.index = pd.to_datetime(px.index).normalize()
    px = px[~px.index.duplicated(keep="last")].sort_index()
    return pd.DataFrame({"close": px["TRDPRC_1"].astype(float), "low": px["LOW_1"].astype(float),
                         "volume": px["ACVOL_UNS"].astype(float)}).dropna(subset=["close"])


def show(df, cols):
    pd.set_option("display.width", 250, "display.max_colwidth", 95)
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))


def cmd_plan(a):
    from src.agent import trade_desk as TD
    TD.P["min_credit_pct"] = a.min_credit_pct
    TD.P["width"], TD.P["risk_usd"] = a.width, a.risk_usd
    earn = pd.read_parquet(RAW / "earnings" / "earnings_dates.parquet")
    if a.ric:
        takes = pd.DataFrame([{"ric": a.ric, "date": a.shock_date}])
    else:
        files = sorted(OUT.glob("live_*.csv"))
        if not files:
            raise SystemExit("no data/results/agent/live_*.csv: run scripts/live_scan.py first")
        live = pd.concat([pd.read_csv(f) for f in files[-4:]], ignore_index=True)
        takes = live[live.get("agent_take", False) == True].drop_duplicates(["ric", "date"])
        if takes.empty:
            print(f"no TAKE in the last scans ({[f.name for f in files[-4:]]}): nothing to open today")
            return
    rows = []
    for t in takes.itertuples():
        px = prices(t.ric)
        today = pd.Timestamp(a.today) if a.today else px.index[-1]
        ed = earn.loc[earn["ric"] == t.ric, "report_date"]
        rows.append(entry_plan(t.ric, t.date, today, px.loc[:today], ed, a.capital, a.iv))
    df = pd.DataFrame(rows)
    path = OUT / f"plan_{pd.Timestamp(rows[0]['as_of']):%Y%m%d}.csv"
    df.to_csv(path, index=False)
    show(df, ["ric", "shock_date", "days_since_shock", "action", "expiry", "dte", "short_strike", "long_strike",
              "min_credit", "est_credit", "contracts", "loss_per_spread_usd", "max_loss_usd", "post_news_low", "why"])
    print(f"\nsaved {path}. For each ENTER: in the broker pick the listed put closest to |delta| 0.15 BELOW post_news_low "
          f"on that expiry, long ~5% of the price lower; only sell if credit >= min_credit; contracts sized for "
          f"2% of ${a.capital:,.0f}. If est_credit < min_credit the trade likely won't qualify.")


def cmd_manage(a):
    p = OUT / "positions.csv"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["ric", "opened", "expiry", "short_strike", "long_strike", "credit", "contracts",
                              "mark", "status"]).to_csv(p, index=False)
        print(f"created empty {p}: add a row for each spread when it fills")
        return
    pos = pd.read_csv(p)
    pos = pos[pos["status"].fillna("open").str.lower() == "open"]
    if pos.empty:
        print("no open positions")
        return
    rows = []
    for r in pos.to_dict("records"):
        px = prices("SPY" if r["ric"] == "XSP" else r["ric"])   # XSP ~ SPY (S&P/10); confirm with the broker quote
        today = pd.Timestamp(a.today) if a.today else px.index[-1]
        opened = pd.Timestamp(r["opened"])
        e = px["close"].loc[:opened]
        rows.append({**manage(r, float(px["close"].loc[:today].iloc[-1]), today,
                              float(e.iloc[-1]) if len(e) else None, r.get("mark")), "as_of": today.date()})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"manage_{pd.Timestamp(rows[0]['as_of']):%Y%m%d}.csv", index=False)
    show(df, ["ric", "as_of", "close", "short", "long", "days_left", "cost_to_close", "mark_source", "captured",
              "time_used", "pnl_now_usd", "action", "why"])
    print("\nCLOSE = buy the spread back tomorrow (confirm the price in the broker). After it fills, set status=closed.")


def cmd_index(a):
    spy = prices("SPY")
    today = pd.Timestamp(a.today) if a.today else spy.index[-1]
    vix = None
    vp = RAW / "prices" / "daily" / "_VIX_adj.parquet"
    if vp.exists() and not a.iv:
        v = pd.read_parquet(vp)
        if "date" in v.columns:
            v = v.set_index("date")
        v.index = pd.to_datetime(v.index).normalize()
        v = v["TRDPRC_1"].astype(float).loc[:today].dropna()
        vix = float(v.iloc[-1]) if len(v) else None
    pos = pd.read_csv(OUT / "positions.csv") if (OUT / "positions.csv").exists() else pd.DataFrame(columns=["ric", "status"])
    n_open = int(((pos["ric"] == "XSP") & (pos["status"].fillna("open").str.lower() == "open")).sum())
    r = index_plan(today, spy, a.capital, a.risk_usd, a.width, a.iv, vix, n_open, a.max_open, _allow_deprecated=True)
    r["action"] = "DEPRECATED - DO NOT TRADE: " + str(r.get("action"))
    df = pd.DataFrame([r])
    df.to_csv(OUT / f"index_plan_{pd.Timestamp(r['as_of']):%Y%m%d}.csv", index=False)
    show(df, ["ric", "as_of", "action", "expiry", "dte", "short_strike", "long_strike", "width", "est_credit",
              "min_credit", "contracts", "loss_per_spread_usd", "vol_source", "why"])
    print("\nIn the broker: XSP puts on that expiry, short the strike closest to delta -0.15 (the estimate above is a "
          "starting point), long one width lower. Sell only at or above min_credit. XSP is cash-settled: nothing is "
          "assigned. When it fills, add a row to positions.csv with ric XSP.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--capital", type=float, required=True, help="fund capital in USD (sizing: 2%% max loss per trade)")
    pl.add_argument("--min-credit-pct", type=float, default=0.25, help="min credit / width (STR-3.4b); use the backtest's chosen value")
    pl.add_argument("--width", type=float, help="fixed spread width in $ (e.g. 1) instead of 5%% of the price")
    pl.add_argument("--risk-usd", type=float, help="max loss per trade in $ (overrides 2%% of capital)")
    pl.add_argument("--iv", type=float, help="broker's ~30-day put IV as a decimal (better strike estimate)")
    pl.add_argument("--ric")
    pl.add_argument("--shock-date")
    pl.add_argument("--today")
    ix = sub.add_parser("index")
    ix.add_argument("--capital", type=float, required=True)
    ix.add_argument("--risk-usd", type=float, required=True, help="max loss you accept on one spread, in $")
    ix.add_argument("--width", type=float, default=1.0, help="XSP strikes are $1 apart; $1 wide fits a small account")
    ix.add_argument("--iv", type=float, help="broker's put IV near 0.15 delta (decimal)")
    ix.add_argument("--max-open", type=int, default=1)
    ix.add_argument("--today")
    ix.add_argument("--i-know-this-failed", action="store_true", help="research only: run the deprecated failed rule")
    mg = sub.add_parser("manage")
    mg.add_argument("--today")
    a = ap.parse_args()
    if a.cmd == "index" and not a.i_know_this_failed:
        sys.exit("DEPRECATED (AG-17q): 'trade_desk.py index' is the weekly XSP 0.15-delta $1-wide spread that FAILED "
                 "strategy/preregistered_spy.md (2010-2023). Do not trade it. Use scripts/index_signal.py (the tested "
                 "index branch). For research only: add --i-know-this-failed (output is labelled DEPRECATED).")
    {"plan": cmd_plan, "index": cmd_index, "manage": cmd_manage}[a.cmd](a)


if __name__ == "__main__":
    main()
