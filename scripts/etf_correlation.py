"""
[AG-36b] ETF branch diagnostics (reported, not scored): does spreading the rule over 29 ETFs diversify, or does it
stack the same equity risk? Reads the frozen run's trades (data/results/etf_branch/run_<capital>_trades.csv).

  - trade frequency vs SPY alone
  - P&L and concurrent open risk inside stress windows (Aug 2024 yen-carry unwind, Mar-Apr 2025 tariff sell-off)
  - correlation of daily returns between the ETFs that were traded, all days vs stress days
  - share of losses coming from equity-index/sector ETFs vs the rest

    py -3.12 -u scripts/etf_correlation.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("data/results/etf_branch")
PX = Path("data/raw/prices_bulk")
STRESS = {"Aug 2024 (carry unwind)": ("2024-07-24", "2024-08-16"),
          "Mar-Apr 2025 (tariffs)": ("2025-03-03", "2025-04-30")}
EQUITY = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "SMH", "XBI",
          "KRE", "EEM", "EFA", "FXI", "EWZ", "ARKK", "KWEB", "XOP"}


def adj_returns():
    m = pd.read_csv(PX / "etf_rics.csv")
    out = {}
    for t, ric in zip(m["ticker"], m["ric"]):
        p = PX / "adj" / f"{ric.replace('.', '_')}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            s = d.set_index(pd.to_datetime(d["date"]).dt.normalize())["TRDPRC_1"].astype(float)
            out[t] = s[~s.index.duplicated()].sort_index()
    return pd.DataFrame(out).pct_change()


def open_risk_by_day(t):
    days = pd.bdate_range(t["date"].min(), t["exit"].max())
    r = pd.Series(0.0, index=days)
    n = pd.Series(0, index=days)
    for _, x in t.iterrows():
        m = (days >= x["date"]) & (days < x["exit"])
        r[m] += x["risk_usd"]
        n[m] += 1
    return r, n


def main():
    rets = adj_returns()
    res = {}
    for cap in (2500, 5000, 10000):
        f = OUT / f"run_{cap}_trades.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f, parse_dates=["date", "expiry", "exit"])
        yrs = (t["exit"].max() - pd.Timestamp("2024-04-01")).days / 365.25
        risk, n = open_risk_by_day(t)
        print(f"\n=== ${cap:,}: {len(t)} trades ({len(t) / yrs:.1f}/yr vs SPY alone {(t['root'] == 'SPY').sum() / yrs:.1f}/yr)")
        print(f"  max open positions {n.max()}, max open risk ${risk.max():,.0f} ({risk.max() / cap:.0%} of capital)")
        eq = t["root"].isin(EQUITY)
        print(f"  equity ETFs: {eq.sum()} trades, P&L ${t.loc[eq, 'pnl_usd'].sum():+,.0f} | "
              f"other (GLD, SLV, TLT, HYG, GDX, USO): {(~eq).sum()} trades, ${t.loc[~eq, 'pnl_usd'].sum():+,.0f}")
        big = t[t["ror_net"] <= -0.5]
        print(f"  trades losing >= 50% of risk: {len(big)} -> " +
              ", ".join(f"{r.root} {r.date:%Y-%m-%d} ({r.ror_net:+.0%})" for r in big.itertuples()))
        row = {"trades": len(t), "per_year": len(t) / yrs, "max_open": int(n.max()), "max_open_risk_pct": risk.max() / cap,
               "big_losses": len(big)}
        for name, (a, b) in STRESS.items():
            w = t[(t["exit"] >= a) & (t["date"] <= b)]
            print(f"  {name}: {len(w)} trades open, P&L ${w['pnl_usd'].sum():+,.0f}, "
                  f"max concurrent {int(n.loc[a:b].max()) if len(n.loc[a:b]) else 0}, roots {sorted(set(w['root']))}")
            row[name] = {"trades": len(w), "pnl": float(w["pnl_usd"].sum())}
        res[cap] = row
    traded = sorted(set(pd.concat([pd.read_csv(OUT / f"run_{c}_trades.csv")["root"] for c in (2500, 5000, 10000)
                                   if (OUT / f"run_{c}_trades.csv").exists()])))
    r = rets.loc["2024-04-01":, [c for c in traded if c in rets]]
    stress_days = pd.Index([])
    for a, b in STRESS.values():
        stress_days = stress_days.union(r.loc[a:b].index)
    up = np.triu_indices(r.shape[1], 1)
    c_all = r.corr().values[up]
    c_str = r.loc[stress_days].corr().values[up]
    eqc = [c for c in r.columns if c in EQUITY]
    up2 = np.triu_indices(len(eqc), 1)
    print(f"\nAverage pairwise correlation of daily returns, traded ETFs ({r.shape[1]}): all days {np.nanmean(c_all):.2f}, "
          f"stress days {np.nanmean(c_str):.2f}")
    print(f"  equity ETFs only ({len(eqc)}): all days {np.nanmean(r[eqc].corr().values[up2]):.2f}, "
          f"stress days {np.nanmean(r.loc[stress_days, eqc].corr().values[up2]):.2f}")
    res["corr"] = {"all": float(np.nanmean(c_all)), "stress": float(np.nanmean(c_str)),
                   "equity_all": float(np.nanmean(r[eqc].corr().values[up2])),
                   "equity_stress": float(np.nanmean(r.loc[stress_days, eqc].corr().values[up2]))}
    (OUT / "correlation.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"\nsaved {OUT / 'correlation.json'}")


if __name__ == "__main__":
    main()
