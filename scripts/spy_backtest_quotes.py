"""
Out-of-sample proof test (AG-26) on SPY option chains WITH bid/ask, 2010-2023, from the free Kaggle dataset
"dudesurfin/spy-options-eod-volatility-surface-2010-2023" (OptionsDX EOD chains, one file per year).

    kaggle datasets download -d dudesurfin/spy-options-eod-volatility-surface-2010-2023 -p data/raw/kaggle_spy --unzip
    py -3.12 -u scripts/spy_backtest_quotes.py --dir data/raw/kaggle_spy            # the pre-registered test
    py -3.12 -u scripts/spy_backtest_quotes.py --dir data/raw/kaggle_spy --inspect  # print the columns only

Rules: strategy/preregistered_spy.md, unchanged except that fills use the REAL quotes: sell at the bid, buy at
the ask, $0.65 per contract per leg; take-profit decided on mids, paid at ask/bid. Spot = the chain's own
underlying price. Writes data/results/agent/spy_trades_2010_2023.csv.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from scipy.special import ndtr

from strategy.strategy_logic import pick_expiry, should_take_profit

FEE = 0.65 / 100
NEED = {"quote_date", "expire_date", "strike", "p_bid", "p_ask", "underlying_last"}


def norm(c):
    return str(c).strip().strip("[]").strip().lower()


def load(d):
    files = sorted([*Path(d).rglob("*.parquet"), *Path(d).rglob("*.csv"), *Path(d).rglob("*.txt")])
    if not files:
        sys.exit(f"no parquet/csv files under {d}")
    out = []
    for f in files:
        want = NEED | {"p_delta", "p_volume"}
        if f.suffix == ".parquet":
            import pyarrow.parquet as pq
            names = pq.ParquetFile(f).schema.names
            df = pd.read_parquet(f, columns=[n for n in names if norm(n) in want])   # only what we use
        else:
            df = pd.read_csv(f, low_memory=False)
        df.columns = [norm(c) for c in df.columns]
        miss = NEED - set(df.columns)
        if miss:
            print(f"  skip {f.name}: missing {sorted(miss)}; has {list(df.columns)[:12]}...")
            continue
        keep = [c for c in ("quote_date", "expire_date", "strike", "p_bid", "p_ask", "p_delta", "p_volume",
                            "underlying_last") if c in df.columns]
        df = df[keep].copy()
        for c in keep:
            if c not in ("quote_date", "expire_date"):
                col = df[c].astype(str).str.strip() if df[c].dtype == object else df[c]   # raw values carry spaces
                df[c] = pd.to_numeric(col, errors="coerce")
        df["quote_date"] = pd.to_datetime(df["quote_date"].astype(str).str.strip(), errors="coerce")
        df["expire_date"] = pd.to_datetime(df["expire_date"].astype(str).str.strip(), errors="coerce")
        dte = (df["expire_date"] - df["quote_date"]).dt.days
        df = df[dte.between(0, 40) & (df["strike"] < df["underlying_last"] * 1.02)]
        out.append(df)
        print(f"  {f.name}: {len(df):,} put rows kept", flush=True)
    p = pd.concat(out, ignore_index=True).dropna(subset=["quote_date", "expire_date", "strike", "p_bid", "p_ask"])
    return p.rename(columns={"quote_date": "date", "expire_date": "expiry", "p_bid": "bid", "p_ask": "ask",
                             "p_delta": "delta", "p_volume": "volume", "underlying_last": "S"})


def bs_delta(S, K, T, vol, r=0.02, q=0.018):
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    return -np.exp(-q * T) * ndtr(-d1)


def simulate(p):
    spot = p.groupby("date")["S"].median().sort_index()
    ix = p.drop_duplicates(["date", "expiry", "strike"]).set_index(["date", "expiry", "strike"]).sort_index()
    weeks = pd.Series(spot.index, index=spot.index).groupby(spot.index.to_period("W")).min()
    by = dict(tuple(p.groupby("date")))
    rows = []
    for d in weeks:
        g = by[d]
        g = g[((g["expiry"] - d).dt.days.between(14, 35)) & (g["bid"] > 0) & (g["ask"] >= g["bid"])]
        if g.empty:
            continue
        exp = pick_expiry(sorted(g["expiry"].unique()), d, 21, 14, 35)
        ch = g[g["expiry"] == exp].copy()
        S, T = float(spot.loc[d]), (exp - d).days / 365
        if "delta" not in ch or ch["delta"].isna().all():
            mid = (ch["bid"] + ch["ask"]) / 2
            iv = np.clip(mid / (0.4 * S * math.sqrt(T)) * 2.5, 0.05, 2.0)      # rough; only to rank strikes
            ch["delta"] = bs_delta(S, ch["strike"].values, T, iv.values)
        sh = ch[(ch["strike"] < S) & ch["delta"].between(-0.30, -0.05)]
        if sh.empty:
            continue
        s = sh.loc[(sh["delta"] + 0.15).abs().idxmin()]
        lg = ch[ch["strike"] < s.strike]
        if lg.empty:
            continue
        l = lg.loc[(lg["strike"] - (s.strike - 0.05 * S)).abs().idxmin()]
        width = float(s.strike - l.strike)
        credit_mid = float((s.bid + s.ask) / 2 - (l.bid + l.ask) / 2)
        credit = float(s.bid - l.ask) - 2 * FEE                       # what we actually receive
        if not (0 < credit < width):
            continue
        settle = spot.loc[:exp]
        if settle.index[-1] < exp - pd.Timedelta(days=4):
            continue
        out, cost, how = exp, min(width, max(0.0, s.strike - float(settle.iloc[-1]))), "expiry"
        if 0 < cost < width:
            cost += 2 * FEE
        for day in spot.loc[d:exp].index[1:]:
            if day >= exp:
                break
            try:
                a, b = ix.loc[(day, exp, s.strike)], ix.loc[(day, exp, l.strike)]
            except KeyError:
                continue
            mid_cost = max(float((a.bid + a.ask) / 2 - (b.bid + b.ask) / 2), 0.0)
            if should_take_profit(credit_mid, mid_cost, (day - d).days, (exp - d).days, "dynamic", 0.50, 0.20, 0.75):
                out, cost, how = day, max(float(a.ask - b.bid), 0.0) + 2 * FEE, "take-profit"
                break
        rows.append({"date": d, "expiry": exp, "exit": out, "how": how, "S": S, "short": s.strike, "long": l.strike,
                     "width": width, "credit": credit, "credit_mid": credit_mid, "cost": cost,
                     "ror_net": (credit - cost) / (width - credit)})
    return pd.DataFrame(rows), spot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/kaggle_spy")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    if a.inspect:
        f = sorted([*Path(a.dir).rglob("*.parquet"), *Path(a.dir).rglob("*.csv")])[0]
        df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, nrows=5)
        print(f.name, [norm(c) for c in df.columns])
        print(df.head(3).to_string()[:1500])
        return
    p = load(a.dir)
    t, spot = simulate(p)
    Path("data/results/agent").mkdir(parents=True, exist_ok=True)
    t.to_csv("data/results/agent/spy_trades_2010_2023.csv", index=False)
    from scripts.spy_benchmark import prereg
    prereg(t, spot)
    print(t["how"].value_counts().to_dict())


if __name__ == "__main__":
    main()
