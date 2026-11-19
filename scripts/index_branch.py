"""
Index branch (AG-28): the same bull put spread, but on SPY and entered on market-wide signals, designed and
tested in three separate periods so a pass means something.

    design    2010-01 .. 2016-12   choose ONE candidate from a fixed list (below), by train t-stat
    validate  2017-01 .. 2020-12   the chosen candidate must stay positive here, else stop
    final     2021-01 .. 2026-10   run ONCE with the rules frozen in strategy/preregistered_index_branch.md

    py -3.12 -u scripts/index_branch.py --stage design
    py -3.12 -u scripts/index_branch.py --stage validate
    py -3.12 -u scripts/index_branch.py --stage final

Data: data/raw/kaggle_spy (SPY chains with bid/ask, 2010-2023), data/processed/options/spy_puts_2024_2026.csv
(OPRA closes, 2024-2026, costs modeled), data/raw/market_context/spy.parquet + move/us2y/us10y (LSEG, AG-27).

The same elements as the shock strategy: a shock trigger, day +1..+3 entry with "no new low", an implied-vs-
realized vol check, a bull put spread at ~0.15 delta, dynamic take-profit. What changes: the underlying (the index,
where the premium is documented), a crash guard (trend + vol term structure), and the hedge leg (by delta).
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from scipy.special import ndtr

from strategy.strategy_logic import pick_expiry, should_take_profit

FEE = 0.65 / 100
FILL = 1.0   # share of each leg's half bid/ask spread paid: 1 = sell at bid / buy at ask (all tests), 0 = mid (AG-28e)
PERIODS = {"design": ("2010-01-01", "2016-12-31"), "validate": ("2017-01-01", "2020-12-31"),
           "final": ("2021-01-01", "2026-12-31"), "walkforward": ("2010-01-01", "2026-12-31"),
           "small": ("2010-01-01", "2026-12-31"), "capital": ("2010-01-01", "2026-12-31"),
           "capital_v2": ("2010-01-01", "2026-12-31")}
OUT = Path("data/results/index_branch")
PREREG = Path("strategy/preregistered_index_branch.md")

# ---------------------------------------------------------------- candidates (fixed BEFORE any real run: 8 x 3 x 2 = 48)
# Priors carried over from the earlier backtests (not re-tested): short delta 0.15 was best and 0.25 worst in the
# 11/12 single-stock backtest (so 0.25 is dropped and 0.10 added), entry by day +3 with "no new low", no stop-loss,
# dynamic take-profit, buy back between the strikes at expiry. From the shock scoreboard: the biggest drops kept
# falling -> "cap" skips a shock day worse than -4%.
ENTRIES = ["weekly", "weekly+guard", "vrp", "vrp+guard", "shock", "shock+guard", "shock+vrp+guard", "shock+guard+cap"]
STRUCTURES = {"S1 short 0.15d / long 5% lower": (0.15, ("pct", 0.05)),
              "S2 short 0.15d / long 0.05d": (0.15, ("delta", 0.05)),
              "S3 short 0.10d / long 0.03d": (0.10, ("delta", 0.03))}
EXITS = ["tp", "tp+7dte"]
SHOCK_1D, SHOCK_2D, VRP_MIN = -0.015, -0.025, 0.02   # -1.5% day or -2.5% over 2 days; IV30 - RV20 >= 2 vol points


def norm(c):
    return str(c).strip().strip("[]").strip().lower()


# ---------------------------------------------------------------- data
def load_kaggle(d, max_dte=100):
    import pyarrow.parquet as pq
    want = {"quote_date", "expire_date", "strike", "p_bid", "p_ask", "p_delta", "p_iv", "underlying_last"}
    out = []
    for f in sorted(Path(d).glob("*.parquet")):
        names = pq.ParquetFile(f).schema.names
        df = pd.read_parquet(f, columns=[n for n in names if norm(n) in want])
        df.columns = [norm(c) for c in df.columns]
        for c in df.columns:
            if c in ("quote_date", "expire_date"):
                df[c] = pd.to_datetime(df[c].astype(str).str.strip(), errors="coerce")
            else:
                col = df[c].astype(str).str.strip() if df[c].dtype == object else df[c]
                df[c] = pd.to_numeric(col, errors="coerce")
        dte = (df["expire_date"] - df["quote_date"]).dt.days
        df = df[dte.between(0, max_dte) & (df["strike"] <= df["underlying_last"] * 1.03)
                & (df["strike"] >= df["underlying_last"] * 0.6)]
        out.append(df)
        print(f"  {f.name}: {len(df):,} put rows", flush=True)
    p = pd.concat(out, ignore_index=True).dropna(subset=["quote_date", "expire_date", "strike", "p_bid", "p_ask"])
    p = p.rename(columns={"quote_date": "date", "expire_date": "expiry", "p_bid": "bid", "p_ask": "ask",
                          "p_delta": "delta", "p_iv": "iv", "underlying_last": "S"})
    p["mid"] = (p["bid"] + p["ask"]) / 2
    p["src"] = "kaggle"
    return p


def bs_put(S, K, T, vol, r=0.045, q=0.013):
    sq = vol * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / sq
    return K * np.exp(-r * T) * ndtr(-(d1 - sq)) - S * np.exp(-q * T) * ndtr(-d1), d1


def implied(S, K, T, price):
    lo, hi = np.full(len(K), 0.01), np.full(len(K), 3.0)
    for _ in range(50):
        mid = (lo + hi) / 2
        v, _ = bs_put(S, K, T, mid)
        lo, hi = np.where(v < price, mid, lo), np.where(v < price, hi, mid)
    iv = (lo + hi) / 2
    ok = (price > bs_put(S, K, T, 0.011)[0]) & (price < bs_put(S, K, T, 2.99)[0])
    return np.where(ok, iv, np.nan)


def load_opra(csv, spot, start="2024-01-01"):
    """OPRA day aggregates (closes only): bid/ask modeled as close -/+ max($0.01, 5%) like the backtest fills"""
    p = pd.read_csv(csv, header=None, names=["date", "ticker", "volume", "open", "close", "high", "low", "ws", "tx"])
    t = p["ticker"].str.extract(r"^O:SPY(\d{6})P(\d{8})$")
    p = p[t[0].notna()].copy()
    p["expiry"] = pd.to_datetime(t.loc[p.index, 0], format="%y%m%d")
    p["strike"] = t.loc[p.index, 1].astype(int) / 1000
    p["date"] = pd.to_datetime(p["date"])
    p = p[(p["date"] >= start) & (p["volume"] > 0) & (p["close"] > 0)]
    p = p[(p["expiry"] - p["date"]).dt.days.between(0, 100)]
    p["S"] = p["date"].map(spot)
    p = p.dropna(subset=["S"])
    p = p[(p["strike"] <= p["S"] * 1.03) & (p["strike"] >= p["S"] * 0.6)].copy()
    T = ((p["expiry"] - p["date"]).dt.days.clip(lower=1)) / 365
    p["iv"] = implied(p["S"].values, p["strike"].values, T.values, p["close"].values)
    _, d1 = bs_put(p["S"].values, p["strike"].values, T.values, np.nan_to_num(p["iv"].values, nan=0.2))
    p["delta"] = -np.exp(-0.013 * T.values) * ndtr(-d1)
    h = np.maximum(0.01, 0.05 * p["close"])
    p["bid"], p["ask"], p["mid"] = (p["close"] - h).clip(lower=0), p["close"] + h, p["close"]
    p["src"] = "opra"
    return p[["date", "expiry", "strike", "bid", "ask", "mid", "delta", "iv", "S", "src"]]


def lseg_series(name, col_pref=("TRDPRC_1", "YLDTOMAT", "BID")):
    f = Path("data/raw/market_context") / f"{name}.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index).normalize()
    for c in col_pref:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").dropna().sort_index()
    return None


# ---------------------------------------------------------------- features (everything known at the close of day d)
def chain_iv(g, target_dte):
    """median put IV, |delta| 0.35..0.65, at the expiry closest to target_dte"""
    g = g[g["iv"].notna() & g["delta"].abs().between(0.35, 0.65) & (g["iv"] > 0)]
    if g.empty:
        return np.nan
    dte = (g["expiry"] - g["date"]).dt.days
    e = g.loc[(dte - target_dte).abs().idxmin(), "expiry"]
    return float(g.loc[g["expiry"] == e, "iv"].median())


def features(puts, spot_hist):
    rows = []
    for d, g in puts.groupby("date"):
        rows.append({"date": d, "S": float(g["S"].median()), "iv9": chain_iv(g, 9), "iv30": chain_iv(g, 30),
                     "iv60": chain_iv(g, 60)})
    f = pd.DataFrame(rows).set_index("date").sort_index()
    px = spot_hist.reindex(spot_hist.index.union(f.index)).ffill()
    r = px.pct_change()
    f["rv20"] = r.rolling(20).std().reindex(f.index) * math.sqrt(252)
    f["ma200"] = px.rolling(200).mean().reindex(f.index)
    f["ret1"] = r.reindex(f.index)
    f["ret2"] = px.pct_change(2).reindex(f.index)
    f["vrp"] = f["iv30"] - f["rv20"]
    f["guard_ok"] = (f["S"] > f["ma200"]) & ~(f["iv9"] > f["iv30"])   # uptrend and no panic term structure
    f["shock"] = (f["ret1"] <= SHOCK_1D) | (f["ret2"] <= SHOCK_2D)
    return f


def entry_dates(f, rule):
    """[AG-28a] the candidate entry rules; one position at a time is applied later"""
    d = f.index
    weekly = pd.Series(d, index=d).groupby(d.to_period("W")).min().values
    ok = f["guard_ok"].fillna(False)
    vrp_ok = (f["vrp"] >= VRP_MIN).fillna(False)
    if rule.startswith("weekly"):
        base = pd.DatetimeIndex(weekly)
    elif rule.startswith("vrp"):
        base = pd.DatetimeIndex(weekly)[vrp_ok.reindex(weekly).fillna(False).values]
    else:                                        # shock: enter day +1..+3, the first day with no new low
        base, i, idx = [], 0, list(d)
        shocks = list(f.index[f["shock"].fillna(False)])
        last_used = None
        for s in shocks:
            if last_used is not None and s <= last_used:
                continue
            k = idx.index(s)
            low = f["S"].iloc[k]
            for j in range(k + 1, min(k + 4, len(idx))):
                if f["S"].iloc[j] > low:         # held above the shock close: selling has paused
                    base.append(idx[j])
                    last_used = idx[min(k + 3, len(idx) - 1)]
                    break
                low = min(low, f["S"].iloc[j])
        base = pd.DatetimeIndex(base)
        if "vrp" in rule:
            base = base[vrp_ok.reindex(base).fillna(False).values]
        if "cap" in rule:                       # skip entries that follow a crash-size day (shock scoreboard prior)
            worst3 = f["ret1"].rolling(3).min()
            base = base[(worst3.reindex(base) > -0.04).fillna(False).values]
    if "guard" in rule:
        base = base[ok.reindex(base).fillna(False).values]
    return base


# ---------------------------------------------------------------- one trade
def trade(d, chain_by_day, ix, spot, structure, exit_rule):
    short_d, (lmode, lval) = structure
    g = chain_by_day.get(d)
    if g is None:
        return None
    g = g[((g["expiry"] - d).dt.days.between(14, 35)) & (g["bid"] > 0) & (g["ask"] >= g["bid"]) & g["delta"].notna()]
    if g.empty:
        return None
    exp = pick_expiry(sorted(g["expiry"].unique()), d, 21, 14, 35)
    ch = g[g["expiry"] == exp]
    S = float(spot.get(d, ch["S"].median()))
    sh = ch[(ch["strike"] < S) & ch["delta"].between(-0.45, -0.03)]
    if sh.empty:
        return None
    s = sh.loc[(sh["delta"] + short_d).abs().idxmin()]
    lg = ch[ch["strike"] < s.strike]
    if lg.empty:
        return None
    if lmode == "pct":
        l = lg.loc[(lg["strike"] - (s.strike - lval * S)).abs().idxmin()]
    elif lmode == "abs":
        l = lg.loc[(lg["strike"] - (s.strike - lval)).abs().idxmin()]
    elif lmode == "minw":                       # [AG-28f] narrower of ($W below the short) and the 0.03-delta put
        w_dollars, dlt = lval
        la = lg.loc[(lg["strike"] - (s.strike - w_dollars)).abs().idxmin()]
        ld = lg.loc[(lg["delta"] + dlt).abs().idxmin()]
        l = la if la.strike >= ld.strike else ld
    else:
        l = lg.loc[(lg["delta"] + lval).abs().idxmin()]
    width = float(s.strike - l.strike)
    credit_mid = float(s.mid - l.mid)
    credit = credit_mid - FILL * float((s.ask - s.bid) + (l.ask - l.bid)) / 2 - 2 * FEE   # FILL=1 -> s.bid - l.ask
    if not (0 < credit < width) or credit_mid <= 0:
        return None
    settle = spot.loc[:exp]
    if settle.empty or settle.index[-1] < exp - pd.Timedelta(days=4):
        return None
    out, how = exp, "expiry"
    cost = min(width, max(0.0, s.strike - float(settle.iloc[-1])))
    if 0 < cost < width:
        cost += 2 * FEE
    days = spot.loc[d:exp].index[1:]
    for day in days:
        if day >= exp:
            break
        try:
            a, b = ix.loc[(day, exp, s.strike)], ix.loc[(day, exp, l.strike)]
        except KeyError:
            continue
        mc = max(float(a["mid"] - b["mid"]), 0.0)
        pay = max(float(a["mid"] - b["mid"]) + FILL * float((a["ask"] - a["bid"]) + (b["ask"] - b["bid"])) / 2, 0.0) + 2 * FEE
        if should_take_profit(credit_mid, mc, (day - d).days, (exp - d).days, "dynamic", 0.50, 0.20, 0.75):
            out, cost, how = day, pay, "take-profit"
            break
        if exit_rule == "tp+7dte" and (exp - day).days <= 7:
            out, cost, how = day, pay, "7-dte"
            break
    return {"date": d, "expiry": exp, "exit": out, "how": how, "S": S, "short": s.strike, "long": l.strike,
            "width": width, "credit": credit, "cost": cost, "ror_net": (credit - cost) / (width - credit)}


def one_open(t):
    keep, free = [], pd.Timestamp("1900-01-01")
    for i, r in t.sort_values("date").iterrows():
        if r["date"] > free:
            keep.append(i)
            free = r["exit"]
    return t.loc[keep]


def stats(t, risk=0.10, spot=None, seed=0):
    if t is None or t.empty:
        return {"n": 0}
    x = t["ror_net"].values
    rng = np.random.default_rng(seed)
    b = [rng.choice(x, len(x)).mean() for _ in range(3000)] if len(x) > 4 else [np.nan]
    se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 2 else np.nan
    pnl = (t.set_index("exit")["ror_net"] * risk).groupby(level=0).sum()
    yearly = pnl.groupby(pnl.index.year).sum()
    monthly = pnl.groupby([pnl.index.year, pnl.index.month]).sum()
    eq = (1 + pnl).cumprod()
    return {"n": len(x), "mean": float(x.mean()), "t": float(x.mean() / se) if se else np.nan,
            "ci_lo": float(np.percentile(b, 2.5)), "ci_hi": float(np.percentile(b, 97.5)),
            "win": float((x > 0).mean()), "worst": float(x.min()), "total": float(eq.iloc[-1] - 1),
            "maxdd": float((eq / eq.cummax() - 1).min()), "years_pos": f"{int((yearly > 0).sum())}/{len(yearly)}",
            "years_pos_share": float((yearly > 0).mean()), "worst_month": float(monthly.min()),
            "yearly": {int(k): round(float(v), 4) for k, v in yearly.items()}}


# ---------------------------------------------------------------- capital rule v2
def capital_structure(capital, risk_pct=0.10):
    """[AG-28f] strategy/preregistered_capital_fix.md: W = floor(capital x risk / 100); long leg = narrower of $W and
    the 0.03-delta put. Returns (budget, W, structure); W < 1 -> structure None (account too small)."""
    budget = capital * risk_pct
    W = int(budget // 100)
    return budget, W, ((0.10, ("minw", (float(W), 0.03))) if W >= 1 else None)


def within_budget(t, budget, slack=1.15):
    """keep only spreads whose max loss fits the budget (every account size, unlike v1)"""
    if t is None or len(t) == 0:
        return t
    return t[(t["width"] - t["credit"]) * 100 <= budget * slack]


def capital_v2(f, by, ix, spot, capital, risk_pct=0.10):
    if not capital:
        sys.exit("--capital is required with --stage capital_v2")
    budget, W, struct = capital_structure(capital, risk_pct)
    if struct is None:
        sys.exit(f"${capital:,.0f} x {risk_pct:.0%} = ${budget:,.0f}: less than one $1-wide spread. Not tradeable.")
    import hashlib
    sha = hashlib.sha256(Path("strategy/preregistered_capital_fix.md").read_bytes()).hexdigest()[:16]
    rows = [trade(d, by, ix, spot, struct, "tp") for d in entry_dates(f, "shock+guard")]
    t = within_budget(pd.DataFrame([r for r in rows if r]), budget)
    t.to_csv(OUT / f"capital_v2_{int(capital)}_trades.csv", index=False)
    raw, acct = stats(t, risk_pct), stats(one_open(t), risk_pct)
    checks = [("1. mean ror > 0 with 95% interval above 0", raw["ci_lo"] > 0,
               f"{raw['mean']:+.2%} [{raw['ci_lo']:+.2%}, {raw['ci_hi']:+.2%}], n={raw['n']}"),
              ("2. positive in >= 70% of years with trades", acct["years_pos_share"] >= 0.7, acct["years_pos"]),
              ("3. worst month better than -25%", acct["worst_month"] > -0.25, f"{acct['worst_month']:+.1%}"),
              ("4. max drawdown better than -30%", acct["maxdd"] > -0.30, f"{acct['maxdd']:+.1%}")]
    risk_share = ((t["width"] - t["credit"]) * 100 / capital)
    print(f"\nCAPITAL TEST v2 (pre-reg sha {sha}): ${capital:,.0f}, budget ${budget:,.0f}, long leg = narrower of ${W} and 0.03d")
    for name, ok, det in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {det}")
    print(f"  account (one spread at a time): {acct['total']:+.1%}, max dd {acct['maxdd']:+.1%}, avg credit "
          f"${t['credit'].mean():.2f}, avg width ${t['width'].mean():.2f}, max risk per trade {risk_share.max():.1%} of capital")
    print(f"  yearly {acct['yearly']}")
    print(f"  VERDICT: {'PASS' if all(c[1] for c in checks) else 'FAIL'}")
    return t


# ---------------------------------------------------------------- pipeline
def build(period):
    s0, s1 = PERIODS[period]
    spy = lseg_series("spy")
    parts = []
    if pd.Timestamp(s0) <= pd.Timestamp("2023-12-31"):
        k = load_kaggle("data/raw/kaggle_spy")
        parts.append(k[(k["date"] >= s0) & (k["date"] <= min(pd.Timestamp(s1), pd.Timestamp("2023-12-31")) + pd.Timedelta(days=45))])
    if pd.Timestamp(s1) >= pd.Timestamp("2024-01-01"):
        sp = spy if spy is not None else None
        un = pd.read_parquet("data/raw/prices/daily/SPY_unadj.parquet")
        un = un.set_index("date") if "date" in un.columns else un
        un.index = pd.to_datetime(un.index).normalize()
        parts.append(load_opra("data/processed/options/spy_puts_2024_2026.csv", un["TRDPRC_1"].astype(float)))
    puts = pd.concat(parts, ignore_index=True)
    # settlement / spot: the chain's own underlying where we have it, LSEG SPY otherwise
    chain_spot = puts.groupby("date")["S"].median()
    hist = spy if spy is not None else chain_spot
    spot = chain_spot.combine_first(hist).sort_index()
    f = features(puts, hist.combine_first(chain_spot).sort_index())
    f = f[(f.index >= s0) & (f.index <= s1)]
    by = dict(tuple(puts.groupby("date")))
    ix = puts.drop_duplicates(["date", "expiry", "strike"]).set_index(["date", "expiry", "strike"]).sort_index()
    return f, by, ix, spot


def run_candidates(f, by, ix, spot, candidates):
    res, cache = [], {}
    for entry in sorted({c[0] for c in candidates}):
        cache[entry] = entry_dates(f, entry)
    for entry, sname, ex in candidates:
        rows = [trade(d, by, ix, spot, STRUCTURES[sname], ex) for d in cache[entry]]
        t = pd.DataFrame([r for r in rows if r])
        t1 = one_open(t) if len(t) else t
        st = stats(t)
        acct = stats(t1)
        res.append({"entry": entry, "structure": sname, "exit": ex, **{k: st.get(k) for k in
                    ("n", "mean", "t", "ci_lo", "ci_hi", "win", "worst")},
                    "acct_total": acct.get("total"), "acct_maxdd": acct.get("maxdd"),
                    "acct_years_pos": acct.get("years_pos"), "acct_worst_month": acct.get("worst_month"),
                    "yearly": json.dumps(acct.get("yearly", {}))})
        print(f"  {entry:18s} {sname:32s} {ex:8s} n={st.get('n', 0):4d} mean={st.get('mean', np.nan):+.2%} "
              f"t={st.get('t', np.nan):+.2f}", flush=True)
    return pd.DataFrame(res)


def all_candidate_trades(f, by, ix, spot):
    """[AG-28b] every candidate's trades over the whole period, one row per trade with its candidate id"""
    out, cache = [], {}
    for e in ENTRIES:
        cache[e] = entry_dates(f, e)
    for e in ENTRIES:
        for sname in STRUCTURES:
            for x in EXITS:
                rows = [trade(d, by, ix, spot, STRUCTURES[sname], x) for d in cache[e]]
                t = pd.DataFrame([r for r in rows if r])
                if len(t):
                    t["cand"] = f"{e} | {sname} | {x}"
                    out.append(t)
                print(f"  {e:18s} {sname:30s} {x:8s} n={len(t)}", flush=True)
    return pd.concat(out, ignore_index=True)


def walk_forward(tr, first=2014, last=2026, min_trades=20):
    """[AG-28c] each year: pick the candidate with the best per-trade t on trades exited before that year"""
    picks, stitched = [], []
    for y in range(first, last + 1):
        hist = tr[tr["exit"] < pd.Timestamp(f"{y}-01-01")]
        g = hist.groupby("cand")["ror_net"].agg(["size", "mean", "std"])
        g = g[g["size"] >= min_trades]
        g["t"] = g["mean"] / (g["std"] / np.sqrt(g["size"]))
        best = g["t"].idxmax()
        yr = tr[(tr["cand"] == best) & (tr["date"].dt.year == y)]
        picks.append({"year": y, "pick": best, "train_n": int(g.loc[best, "size"]), "train_t": float(g.loc[best, "t"]),
                      "trades": len(yr), "year_mean": float(yr["ror_net"].mean()) if len(yr) else np.nan})
        stitched.append(yr)
    return pd.DataFrame(picks), pd.concat(stitched, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["design", "validate", "final", "walkforward", "small", "capital", "capital_v2"], required=True)
    ap.add_argument("--capital", type=float, help="--stage capital: account size in $")
    ap.add_argument("--risk-pct", type=float, default=0.10, help="--stage capital: max loss per trade as share of capital")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f, by, ix, spot = build(a.stage)
    f.to_csv(OUT / f"features_{a.stage}.csv")
    if a.stage == "capital_v2":
        return capital_v2(f, by, ix, spot, a.capital, a.risk_pct)
    if a.stage == "capital":
        # [AG-28d] (v1, kept to reproduce the frozen result; it has the W >= 15 sizing bug -> use capital_v2) account-size rule (strategy/preregistered_index_branch_small.md, "Capital rule"):
        # W = largest whole $ with W x 100 <= capital x risk_pct; W >= 15 -> the tested delta-based long leg (0.03d)
        if not a.capital:
            sys.exit("--capital is required with --stage capital")
        budget = a.capital * a.risk_pct
        W = int(budget // 100)
        if W < 1:
            sys.exit(f"${a.capital:,.0f} x {a.risk_pct:.0%} = ${budget:,.0f}: less than one $1-wide spread (~$100). Not tradeable.")
        struct = (0.10, ("delta", 0.03)) if W >= 15 else (0.10, ("abs", float(W)))
        dates = entry_dates(f, "shock+guard")
        rows = [trade(d, by, ix, spot, struct, "tp") for d in dates]
        t = pd.DataFrame([r for r in rows if r])
        if W < 15:
            t = t[(t["width"] - t["credit"]) * 100 <= budget * 1.15]      # only spreads the account can carry
        t.to_csv(OUT / f"capital_{int(a.capital)}_trades.csv", index=False)
        raw, acct = stats(t, a.risk_pct), stats(one_open(t), a.risk_pct)
        checks = [("1. mean ror > 0 with 95% interval above 0", raw["ci_lo"] > 0,
                   f"{raw['mean']:+.2%} [{raw['ci_lo']:+.2%}, {raw['ci_hi']:+.2%}], n={raw['n']}"),
                  ("2. positive in >= 70% of years with trades", acct["years_pos_share"] >= 0.7, acct["years_pos"]),
                  ("3. worst month better than -25%", acct["worst_month"] > -0.25, f"{acct['worst_month']:+.1%}"),
                  ("4. max drawdown better than -30%", acct["maxdd"] > -0.30, f"{acct['maxdd']:+.1%}")]
        label = "delta-based long leg (0.03d)" if W >= 15 else f"${W}-wide spread"
        print(f"\nCAPITAL TEST: ${a.capital:,.0f}, max loss per trade {a.risk_pct:.0%} = ${budget:,.0f} -> {label}, 2010-2026")
        for name, ok, det in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: {det}")
        print(f"  account (one spread at a time): {acct['total']:+.1%}, max dd {acct['maxdd']:+.1%}, "
              f"avg credit ${t['credit'].mean():.2f}, avg width ${t['width'].mean():.2f}")
        print(f"  yearly {acct['yearly']}")
        print(f"  VERDICT: {'PASS' if all(c[1] for c in checks) else 'FAIL'}")
        if W in (1, 2, 3):
            print("  note: widths $1-$3 were run before this rule was written ($1 failed); treat $2-$3 as evidence, not proof.")
        return
    if a.stage == "small":
        pr = Path("strategy/preregistered_index_branch_small.md")
        spec = json.loads(pr.read_text().split("```json")[1].split("```")[0])
        dates = entry_dates(f, spec["entry"])
        res = {}
        for w in spec["widths"]:
            rows = [trade(d, by, ix, spot, (spec["short_delta"], ("abs", w)), spec["exit"]) for d in dates]
            t = pd.DataFrame([r for r in rows if r])
            t.to_csv(OUT / f"small_w{w:g}_trades.csv", index=False)
            raw, acct = stats(t, spec["risk"]), stats(one_open(t), spec["risk"])
            res[w] = (raw, acct, t)
            print(f"W=${w:g}: n={raw['n']} mean {raw['mean']:+.2%} [{raw['ci_lo']:+.2%}, {raw['ci_hi']:+.2%}] "
                  f"win {raw['win']:.0%} worst {raw['worst']:+.1%} | account {acct['total']:+.1%}, max dd {acct['maxdd']:+.1%}, "
                  f"years {acct['years_pos']}, avg credit ${t['credit'].mean():.3f}, avg width ${t['width'].mean():.2f}")
        raw, acct, t = res[spec["primary"]]
        checks = [("1. mean ror > 0 with 95% interval above 0", raw["ci_lo"] > 0,
                   f"{raw['mean']:+.2%} [{raw['ci_lo']:+.2%}, {raw['ci_hi']:+.2%}], n={raw['n']}"),
                  ("2. positive in >= 70% of years with trades", acct["years_pos_share"] >= 0.7, acct["years_pos"]),
                  ("3. worst month better than -25%", acct["worst_month"] > -0.25, f"{acct['worst_month']:+.1%}"),
                  ("4. max drawdown better than -30%", acct["maxdd"] > -0.30, f"{acct['maxdd']:+.1%}")]
        print(f"\nPRE-REGISTERED ACCOUNT-SIZE TEST, W = ${spec['primary']:g}, 2010-2026")
        for name, ok, det in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: {det}")
        print(f"  yearly {acct['yearly']}")
        print(f"  VERDICT: {'PASS' if all(c[1] for c in checks) else 'FAIL'}")
        print(f"  rules file sha256: {hashlib.sha256(pr.read_bytes()).hexdigest()[:16]}")
        return
    if a.stage == "walkforward":
        frozen = json.loads(PREREG.read_text().split("```json")[1].split("```")[0])
        tr = all_candidate_trades(f, by, ix, spot)
        tr.to_csv(OUT / "all_candidate_trades.csv", index=False)
        picks, oos = walk_forward(tr, frozen["first_year"], frozen["last_year"], frozen["min_trades"])
        picks.to_csv(OUT / "walkforward_picks.csv", index=False)
        oos.to_csv(OUT / "walkforward_oos_trades.csv", index=False)
        st = stats(one_open(oos), frozen["risk"])
        raw = stats(oos, frozen["risk"])
        checks = [("1. mean ror > 0 with 95% interval above 0", raw["ci_lo"] > 0,
                   f"{raw['mean']:+.2%} [{raw['ci_lo']:+.2%}, {raw['ci_hi']:+.2%}], n={raw['n']}"),
                  ("2. positive in >= 70% of calendar years", st["years_pos_share"] >= 0.7, st["years_pos"]),
                  ("3. worst month better than -25%", st["worst_month"] > -0.25, f"{st['worst_month']:+.1%}"),
                  ("4. max drawdown better than -30%", st["maxdd"] > -0.30, f"{st['maxdd']:+.1%}")]
        pd.set_option("display.width", 220, "display.max_colwidth", 70)
        print("\nWALK-FORWARD PICKS (chosen each year from earlier years only):")
        print(picks.to_string(index=False))
        print(f"\nPRE-REGISTERED WALK-FORWARD TEST {frozen['first_year']}..{frozen['last_year']} (net of costs, "
              f"{frozen['risk']:.0%} risk, one open)")
        for name, ok, det in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: {det}")
        print(f"  account: total {st['total']:+.1%}, max drawdown {st['maxdd']:+.1%}, trades taken {st['n']}, yearly {st['yearly']}")
        print(f"  VERDICT: {'PASS' if all(c[1] for c in checks) else 'FAIL'}")
        print(f"  rules file sha256: {hashlib.sha256(PREREG.read_bytes()).hexdigest()[:16]}")
        return
    if a.stage == "design":
        cands = [(e, s, x) for e in ENTRIES for s in STRUCTURES for x in EXITS]
        r = run_candidates(f, by, ix, spot, cands)
        r.to_csv(OUT / "design_candidates.csv", index=False)
        from src.backtest.overfit import deflated_sharpe
        ok = r[(r["n"] >= 40)].copy()
        best = ok.sort_values("t", ascending=False).iloc[0]
        sr = best["mean"] / (best["mean"] / best["t"] * math.sqrt(best["n"])) if best["t"] else 0
        dsr = deflated_sharpe(sr, int(best["n"]), len(r), float(np.nanvar(
            ok["mean"] / (ok["mean"] / ok["t"] * np.sqrt(ok["n"])))))
        pick = {"entry": best["entry"], "structure": best["structure"], "exit": best["exit"],
                "design_t": float(best["t"]), "design_mean": float(best["mean"]), "design_n": int(best["n"]),
                "deflated_sharpe_prob": dsr, "n_trials": len(r)}
        (OUT / "design_pick.json").write_text(json.dumps(pick, indent=1))
        print("\nDESIGN PICK:", json.dumps(pick, indent=1))
    elif a.stage == "validate":
        pick = json.loads((OUT / "design_pick.json").read_text())
        r = run_candidates(f, by, ix, spot, [(pick["entry"], pick["structure"], pick["exit"])])
        r.to_csv(OUT / "validate_result.csv", index=False)
        v = r.iloc[0]
        verdict = bool(v["n"] and v["mean"] > 0)
        print(f"\nVALIDATION {'PASS' if verdict else 'FAIL'}: mean {v['mean']:+.2%} [{v['ci_lo']:+.2%}, {v['ci_hi']:+.2%}] "
              f"n={v['n']}, account {v['acct_total']:+.1%}, max dd {v['acct_maxdd']:+.1%}, years {v['acct_years_pos']}")
    else:
        if not PREREG.exists():
            sys.exit(f"{PREREG} missing: freeze the rules first")
        frozen = json.loads(PREREG.read_text().split("```json")[1].split("```")[0])
        r = run_candidates(f, by, ix, spot, [(frozen["entry"], frozen["structure"], frozen["exit"])])
        r.to_csv(OUT / "final_result.csv", index=False)
        v = r.iloc[0]
        yearly = json.loads(v["yearly"])
        share = np.mean([x > 0 for x in yearly.values()]) if yearly else 0
        checks = [("1. mean ror > 0 with 95% interval above 0", v["ci_lo"] > 0,
                   f"{v['mean']:+.2%} [{v['ci_lo']:+.2%}, {v['ci_hi']:+.2%}], n={v['n']}"),
                  ("2. positive in >= 70% of calendar years", share >= 0.7, f"{v['acct_years_pos']}"),
                  ("3. worst month better than -25% (10% risk)", v["acct_worst_month"] > -0.25, f"{v['acct_worst_month']:+.1%}"),
                  ("4. max drawdown better than -30% (10% risk)", v["acct_maxdd"] > -0.30, f"{v['acct_maxdd']:+.1%}")]
        print(f"\nFINAL PRE-REGISTERED TEST {PERIODS['final'][0]} .. {PERIODS['final'][1]}: "
              f"{frozen['entry']} | {frozen['structure']} | {frozen['exit']}")
        for name, ok, det in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: {det}")
        print(f"  account: total {v['acct_total']:+.1%}, yearly {yearly}")
        print(f"  VERDICT: {'PASS' if all(c[1] for c in checks) else 'FAIL'}")
        print(f"  rules file sha256: {hashlib.sha256(PREREG.read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
