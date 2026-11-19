"""
Index benchmark (AG-24): the same bull put spread on SPY, entered on a fixed schedule, priced with REAL
option closes from the OPRA day-aggregate files. The bar every single-stock signal has to beat.

Same rules as scripts/score_signals.py: expiry 14-35 DTE closest to 21, short put closest to |delta| 0.15
below the close, long put ~5% of the price lower, held to expiry, ror = P&L / max loss, closes only (no bid/ask).
Delta from Black-Scholes on each put's own close (IV solved per strike), dividend yield ~1.3%.

    py -3.12 -u scripts/spy_benchmark.py                       # extracts SPY puts once (~5 min), then simulates
    py -3.12 -u scripts/spy_benchmark.py --puts spy_puts.csv   # reuse an extract
    py -3.12 -u scripts/spy_benchmark.py --every day           # enter every trading day (overlapping trades)

Writes data/results/agent/spy_trades.csv and prints in-sample vs hold-out next to the shock trades.
"""
import argparse
import gzip
import math
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from scipy.special import ndtr

AGGS = Path("data/raw/options/day_aggs_v1")
R, Q = 0.045, 0.013
FEE = 0.65 / 100          # $ per share per leg (0.65 per contract)


def hc(px):
    return max(0.01, 0.05 * px)       # half-spread haircut, same rule as the backtest fills (BT fills.py)


def extract(out):
    rows = []
    for f in sorted(AGGS.rglob("*.csv.gz")):
        d = f.name[:10]
        with gzip.open(f, "rt") as fh:
            next(fh)
            rows += [f"{d},{line}" for line in fh if line.startswith("O:SPY") and line[11] == "P"]
    Path(out).write_text("".join(rows))
    return out


def load_puts(path):
    p = pd.read_csv(path, header=None, names=["date", "ticker", "volume", "open", "close", "high", "low", "ws", "tx"])
    t = p["ticker"].str.extract(r"^O:SPY(\d{6})P(\d{8})$")
    p = p[t[0].notna()].copy()
    p["expiry"] = pd.to_datetime(t.loc[p.index, 0], format="%y%m%d")
    p["strike"] = t.loc[p.index, 1].astype(int) / 1000
    p["date"] = pd.to_datetime(p["date"])
    return p[["date", "expiry", "strike", "close", "volume"]]


def bs_put(S, K, T, vol):
    sq = vol * np.sqrt(T)
    d1 = (np.log(S / K) + (R - Q + 0.5 * vol ** 2) * T) / sq
    return K * np.exp(-R * T) * ndtr(-(d1 - sq)) - S * np.exp(-Q * T) * ndtr(-d1), d1


def implied(S, K, T, price, iters=60):
    lo, hi = np.full(len(K), 0.01), np.full(len(K), 3.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        v, _ = bs_put(S, K, T, mid)
        lo, hi = np.where(v < price, mid, lo), np.where(v < price, hi, mid)
    iv = (lo + hi) / 2
    ok = (price > bs_put(S, K, T, 0.011)[0]) & (price < bs_put(S, K, T, 2.99)[0])
    return np.where(ok, iv, np.nan)


def take_profit_exit(puts_ix, d, exp, ks, kl, credit, spot):
    """STR-3.6a dynamic take-profit on daily closes of both legs (days where a leg did not trade are skipped)"""
    from strategy.strategy_logic import should_take_profit
    dte = (exp - d).days
    for day in spot.loc[d:exp].index[1:]:
        if day >= exp:
            break
        try:
            a, b = puts_ix.loc[(day, exp, ks), "close"], puts_ix.loc[(day, exp, kl), "close"]
        except KeyError:
            continue
        cost = max(float(a) - float(b), 0.0)
        if should_take_profit(credit, cost, (day - d).days, dte, "dynamic", 0.50, 0.20, 0.75):
            return day, cost
    return None, None


def simulate(puts, spot, every="week", exits="hold"):
    puts_ix = puts.drop_duplicates(["date", "expiry", "strike"]).set_index(["date", "expiry", "strike"]).sort_index()
    dates = sorted(puts["date"].unique())
    if every == "week":
        s = pd.Series(dates, index=dates)
        dates = list(s.groupby(pd.DatetimeIndex(s.index).to_period("W")).min())
    by = dict(tuple(puts.groupby("date")))
    rows = []
    for d in dates:
        if d not in spot.index or d not in by:
            continue
        S = float(spot.loc[d])
        g = by[d]
        g = g[((g["expiry"] - d).dt.days.between(14, 35)) & (g["volume"] > 0) & (g["close"] > 0)]
        if g.empty:
            continue
        dte = (g["expiry"] - d).dt.days
        exp = g.loc[(dte - 21).abs().idxmin(), "expiry"]
        ch = g[g["expiry"] == exp].copy()
        T = (exp - d).days / 365
        ch["iv"] = implied(S, ch["strike"].values, T, ch["close"].values)
        _, d1 = bs_put(S, ch["strike"].values, T, ch["iv"].fillna(0.2).values)
        ch["delta"] = -np.exp(-Q * T) * ndtr(-d1)
        sh = ch[(ch["strike"] < S) & ch["iv"].notna() & ch["delta"].between(-0.30, -0.05)]
        if sh.empty:
            continue
        s = sh.loc[(sh["delta"] + 0.15).abs().idxmin()]
        lg = ch[ch["strike"] < s.strike]
        if lg.empty:
            continue
        l = lg.loc[(lg["strike"] - (s.strike - 0.05 * S)).abs().idxmin()]
        credit, width = float(s.close - l.close), float(s.strike - l.strike)
        if not (0 < credit < width):
            continue
        credit_net = credit - hc(float(s.close)) - hc(float(l.close)) - 2 * FEE
        settle = spot.loc[:exp]
        if settle.index[-1] < exp - pd.Timedelta(days=4):
            continue
        ST = float(settle.iloc[-1])
        loss = min(width, max(0.0, s.strike - ST))
        loss_net = loss + (2 * FEE if 0 < loss < width else 0.0)     # buy back between strikes at expiry
        exit_day = None
        if exits == "tp":
            exit_day, cost = take_profit_exit(puts_ix, d, exp, s.strike, l.strike, credit, spot)
            if exit_day is not None:
                loss = cost
                loss_net = cost + 2 * 0.01 + 2 * FEE                     # cross the spread on both legs to close
        rows.append({"date": d, "expiry": exp, "exit": exit_day, "S": S, "short": s.strike, "long": l.strike, "credit": credit,
                     "width": width, "credit_pct": credit / width, "short_delta": float(s.delta),
                     "ror": (credit - loss) / (width - credit),
                     "credit_net": credit_net, "ror_net": (credit_net - loss_net) / (width - credit_net)
                     if 0 < credit_net < width else np.nan})
    return pd.DataFrame(rows)


def line(name, x):
    return (f"{name:34s} n={len(x):4d}  mean ror {x.mean():+.1%}  median {x.median():+.1%}  "
            f"win {(x > 0).mean():.0%}  max-loss {(x <= -0.9).mean():.0%}")


def prereg(t, spot, risk=0.10):
    """the 4 checks of strategy/preregistered_spy.md, net of costs, one position at a time"""
    t = t.dropna(subset=["ror_net"]).sort_values("date").copy()
    t["out"] = pd.to_datetime(t["exit"]).fillna(t["expiry"]) if "exit" in t else t["expiry"]
    keep, free = [], pd.Timestamp("1900-01-01")
    for i, r in t.iterrows():
        if r.date > free:
            keep.append(i)
            free = r.out
    k = t.loc[keep]
    rng = np.random.default_rng(0)
    x = t["ror_net"].values
    boots = [rng.choice(x, len(x)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pnl = (k.set_index("out")["ror_net"] * risk).groupby(level=0).sum()
    idx = spot.loc[k["date"].min():k["out"].max()].index
    eq = (1 + pnl.reindex(idx).fillna(0)).cumprod()
    dd = eq / eq.cummax() - 1
    yearly = (k.set_index("out")["ror_net"] * risk).groupby(lambda d: d.year).sum()
    monthly = (k.set_index("out")["ror_net"] * risk).groupby(lambda d: (d.year, d.month)).sum()
    crisis = {w: float(dd.loc[s0:s1].min()) if len(dd.loc[s0:s1]) else float("nan")
              for w, (s0, s1) in {"2011-08": ("2011-07-15", "2011-10-31"), "2015-08": ("2015-08-01", "2015-10-31"),
                                  "2018-02": ("2018-01-15", "2018-04-30"), "2020-03": ("2020-02-15", "2020-05-31"),
                                  "2022": ("2022-01-01", "2022-12-31")}.items()}
    checks = [("1. mean ror > 0, 95% interval above 0", lo > 0, f"mean {x.mean():+.2%} [{lo:+.2%}, {hi:+.2%}], n={len(x)}"),
              ("2. positive in >= 70% of calendar years", (yearly > 0).mean() >= 0.7, f"{(yearly > 0).sum()} of {len(yearly)} years positive"),
              ("3. worst month better than -25%", monthly.min() > -0.25, f"worst month {monthly.min():+.1%}"),
              ("4. crises drawdown better than -30%", all(v > -0.30 for v in crisis.values() if v == v),
               ", ".join(f"{w} {v:+.1%}" for w, v in crisis.items()))]
    print(f"\nPRE-REGISTERED TEST {t['date'].min().date()} .. {t['date'].max().date()} (net of costs, 10% risk, 1 open)")
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    print(f"  account: total {eq.iloc[-1] - 1:+.1%}, max drawdown {dd.min():+.1%}, trades taken {len(k)}")
    print("  yearly (sum of ror x risk):", {y: f"{v:+.1%}" for y, v in yearly.items()})
    print(f"  VERDICT: {'PASS' if all(c[1] for c in checks) else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puts", help="SPY put extract (date,ticker,...); default: build data/processed/options/spy_puts.csv")
    ap.add_argument("--every", choices=["week", "day"], default="week")
    ap.add_argument("--holdout", default="2026-06-01")
    ap.add_argument("--exits", choices=["hold", "tp"], default="hold", help="tp = the strategy's dynamic take-profit")
    ap.add_argument("--prereg", action="store_true", help="score the 4 pass/fail checks in strategy/preregistered_spy.md")
    ap.add_argument("--out", default="data/results/agent/spy_trades.csv")
    a = ap.parse_args()
    path = a.puts or "data/processed/options/spy_puts.csv"
    if not Path(path).exists():
        print("extracting SPY puts from the OPRA day files ...", flush=True)
        extract(path)
    px = pd.read_parquet("data/raw/prices/daily/SPY_unadj.parquet")
    if "date" in px.columns:
        px = px.set_index("date")
    px.index = pd.to_datetime(px.index).normalize()
    spot = px["TRDPRC_1"].astype(float).sort_index()
    t = simulate(load_puts(path), spot, a.every, a.exits)
    Path("data/results/agent").mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, index=False)
    if a.prereg:
        return prereg(t, spot)
    ho = pd.Timestamp(a.holdout)
    print(f"SPY bull put spreads, entered every {a.every}, exits {a.exits}, {t['date'].min().date()} .. {t['date'].max().date()}")
    print(line("SPY all", t["ror"]))
    print(line("SPY in-sample", t.loc[t["date"] < ho, "ror"]))
    print(line("SPY hold-out", t.loc[t["date"] >= ho, "ror"]))
    st = Path("data/results/agent/signal_trades.csv")
    if st.exists():
        s = pd.read_csv(st, parse_dates=["date"])
        print(line("single-stock shocks all", s["ror"]))
        print(line("single-stock shocks in-sample", s.loc[s["date"] < ho, "ror"]))
        print(line("single-stock shocks hold-out", s.loc[s["date"] >= ho, "ror"]))
    print(f"median credit/width SPY {t['credit_pct'].median():.1%}, median short delta {t['short_delta'].median():.2f}")


if __name__ == "__main__":
    main()
