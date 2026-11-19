"""
ETF branch (AG-33..36): the SAME tested index rule, run on 29 liquid ETFs at once to get more trades per year.

Rules are frozen in strategy/preregistered_etf_branch.md (JSON block; its sha256 prefix is printed on every run).
Nothing is chosen from results: entry, strikes, exit and costs are the walk-forward pick of the index branch
(strategy/preregistered_index_branch.md), with three written-down generalizations for non-SPY ETFs:
vol-scaled shock size, wider assumed bid/ask for the less liquid ETFs, and a portfolio risk cap.

    py -3.12 -u scripts/etf_branch.py --validate-spy          # SPY only: must match index_branch (no ETF data needed)
    py -3.12 -u scripts/etf_branch.py --prepare               # split data/processed/options/etf/*.csv.gz per ETF (once)
    py -3.12 -u scripts/etf_branch.py --run                   # all ETFs, capital $2,500 / $5,000 / $10,000

Data: data/processed/options/etf/puts_YYYY-MM.csv.gz (scripts/extract_etf_options.py, OPRA day closes 2024-04..),
data/raw/prices_bulk/{adj,unadj}/<RIC>.parquet + etf_rics.csv (scripts/download_etf_prices.py, LSEG).
Only ~2.5 years of option data: the result is evidence, not proof.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import ndtr  # noqa: E402

import scripts.index_branch as IB  # noqa: E402
from scripts.extract_etf_options import ROOTS  # noqa: E402
from strategy.strategy_logic import pick_expiry, should_take_profit  # noqa: E402

PREREG = Path("strategy/preregistered_etf_branch.md")
OUT = Path("data/results/etf_branch")
ETF_DIR = Path("data/processed/options/etf")
PX_DIR = Path("data/raw/prices_bulk")
SPY_CSV = Path("data/processed/options/spy_puts_2024_2026.csv")
TICK = re.compile(r"^O:([A-Z]+)(\d{6})P(\d{8})$")


def prereg():
    """read the frozen spec (JSON block) and the file's sha256 prefix"""
    txt = PREREG.read_text(encoding="utf-8")
    spec = json.loads(txt.split("```json", 1)[1].split("```", 1)[0])
    return spec, hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- [AG-33] data: option chains per ETF
def parse_tickers(tickers, root):
    """[AG-33] 'O:QQQ250321P00450000' -> expiry 2025-03-21, strike 450.0; other roots, calls, adjusted roots -> NaN"""
    t = pd.Series(tickers).str.extract(r"^O:" + re.escape(root) + r"(\d{6})P(\d{8})$")
    exp = pd.to_datetime(t[0], format="%y%m%d", errors="coerce")
    strike = pd.to_numeric(t[1], errors="coerce") / 1000
    return pd.DataFrame({"expiry": exp.values, "strike": strike.values}, index=pd.Series(tickers).index)


def split_months(force=False):
    """one pass over the monthly ETF files -> OUT/cache/<ROOT>.parquet (date, expiry, strike, volume, close)"""
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    files = sorted(ETF_DIR.glob("puts_*.csv.gz"))
    if not files:
        sys.exit(f"no files in {ETF_DIR}: run scripts/extract_etf_options.py first")
    if not force and all((cache / f"{r}.parquet").exists() for r in ROOTS):
        return
    parts = {r: [] for r in ROOTS}
    for f in files:
        p = pd.read_csv(f, header=None, names=["date", "ticker", "volume", "open", "close", "high", "low", "ws", "tx"],
                        usecols=["date", "ticker", "volume", "close"])
        p = p[(p["volume"] > 0) & (p["close"] > 0)]
        m = p["ticker"].str.extract(TICK.pattern)
        p = p.assign(root=m[0], expiry=pd.to_datetime(m[1], format="%y%m%d", errors="coerce"),
                     strike=pd.to_numeric(m[2], errors="coerce") / 1000).dropna(subset=["root", "expiry"])
        for r, g in p.groupby("root"):
            if r in parts:
                parts[r].append(g[["date", "expiry", "strike", "volume", "close"]])
        print(f"  {f.name}: {len(p):,} rows", flush=True)
    for r, lst in parts.items():
        if lst:
            d = pd.concat(lst, ignore_index=True)
            d["date"] = pd.to_datetime(d["date"])
            d.to_parquet(cache / f"{r}.parquet", index=False)
    print(f"  cache written: {cache}", flush=True)


def half_spread(root, close, spec):
    """assumed half bid/ask per option (closes only in OPRA day files): tight list vs the rest (pre-registered)"""
    c = spec["costs"]["tight"] if root in spec["costs"]["tight_roots"] else spec["costs"]["other"]
    return np.maximum(c["min"], c["pct"] * close)


def chain(raw, spot, root, spec):
    """raw rows (date, expiry, strike, close) -> chain like index_branch.load_opra, with the per-root cost model"""
    p = raw[(raw["expiry"] - raw["date"]).dt.days.between(0, 100)].copy()
    p["S"] = p["date"].map(spot)
    p = p.dropna(subset=["S"])
    p = p[(p["strike"] <= p["S"] * 1.03) & (p["strike"] >= p["S"] * 0.6)].copy()
    T = ((p["expiry"] - p["date"]).dt.days.clip(lower=1)) / 365
    p["iv"] = IB.implied(p["S"].values, p["strike"].values, T.values, p["close"].values)
    _, d1 = IB.bs_put(p["S"].values, p["strike"].values, T.values, np.nan_to_num(p["iv"].values, nan=0.2))
    p["delta"] = -np.exp(-0.013 * T.values) * ndtr(-d1)
    h = half_spread(root, p["close"].values, spec)
    p["bid"], p["ask"], p["mid"] = np.clip(p["close"].values - h, 0, None), p["close"].values + h, p["close"].values
    p["src"] = "opra"
    return p[["date", "expiry", "strike", "bid", "ask", "mid", "delta", "iv", "S", "src"]].reset_index(drop=True)


def load_prices(ticker):
    """(adjusted close, unadjusted close) from LSEG files written by scripts/download_etf_prices.py"""
    m = pd.read_csv(PX_DIR / "etf_rics.csv")
    ric = m.loc[m["ticker"] == ticker, "ric"]
    if ric.empty:
        return None, None
    name = ric.iloc[0].replace(".", "_").replace("^", "-")
    out = []
    for kind in ("adj", "unadj"):
        f = PX_DIR / kind / f"{name}.parquet"
        if not f.exists():
            return None, None
        d = pd.read_parquet(f)
        d = d.set_index("date") if "date" in d.columns else d
        d.index = pd.to_datetime(d.index).normalize()
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
        s = pd.to_numeric(d["TRDPRC_1"], errors="coerce")
        out.append(s.dropna().sort_index().groupby(level=0).last())
    return out[0], out[1]


# ---------------------------------------------------------------- [AG-34] signals: vol-scaled shock + crash guard
def trailing_sigma(px, n=252, min_obs=200):
    """std of daily returns over the n trading days ending the PRIOR day (known before today's close)"""
    return px.pct_change().shift(1).rolling(n, min_periods=min_obs).std()


def shock_flags(px, spy_px, spec, is_spy=False):
    """[AG-34] 1-day <= -1.5% x s or 2-day <= -2.5% x s, s = sigma_ETF(t) / sigma_SPY(t); SPY: s = 1 (the tested rule)"""
    sh = spec["shock"]
    if is_spy:
        s = pd.Series(1.0, index=px.index)
    else:
        s = trailing_sigma(px) / trailing_sigma(spy_px).reindex(px.index).ffill()
    r1, r2 = px.pct_change(), px.pct_change(2)
    return ((r1 <= sh["one_day"] * s) | (r2 <= sh["two_day"] * s)).fillna(False), r1, s


def features_etf(ch, px, spy_px, spec, is_spy=False):
    """per chain date: shock (vol-scaled), 200-day trend and 9d-vs-30d IV guard, on ADJUSTED closes"""
    rows = [{"date": d, "iv9": IB.chain_iv(g, 9), "iv30": IB.chain_iv(g, 30)} for d, g in ch.groupby("date")]
    f = pd.DataFrame(rows).set_index("date").sort_index()
    full = px.reindex(px.index.union(f.index)).ffill()
    spy_full = spy_px.reindex(spy_px.index.union(f.index)).ffill()
    shock, r1, scale = shock_flags(full, spy_full, spec, is_spy)
    ma = full.rolling(spec["guard"]["ma_days"]).mean()
    f["S"] = full.reindex(f.index)                    # adjusted close: used only for the "no new low" check
    f["ret1"] = r1.reindex(f.index)
    f["scale"] = scale.reindex(f.index)
    f["shock"] = shock.reindex(f.index).fillna(False)
    f["rv20"] = full.pct_change().rolling(20).std().reindex(f.index) * math.sqrt(252)
    f["vrp"] = f["iv30"] - f["rv20"]
    f["guard_ok"] = (f["S"] > ma.reindex(f.index)) & ~(f["iv9"] > f["iv30"])
    return f


def entries(f):
    """[AG-34] shock+guard entry dates (day +1..+3, first close above the post-shock low) = index_branch rule"""
    return IB.entry_dates(f, "shock+guard")


# ---------------------------------------------------------------- [AG-35] one trade, capital-rule long leg
def trade(d, by, ix, spot, W, spec, last_day):
    """[AG-35] short ~0.10 delta; long = the NARROWER of (strike nearest short - W) and the ~0.03-delta put
    (W=None: 0.03-delta put only); 14-35 DTE nearest 21; dynamic take-profit, else expiry; no stop"""
    g = by.get(d)
    if g is None:
        return None
    g = g[((g["expiry"] - d).dt.days.between(14, 35)) & (g["bid"] > 0) & (g["ask"] >= g["bid"]) & g["delta"].notna()]
    if g.empty:
        return None
    exp = pick_expiry(sorted(g["expiry"].unique()), d, 21, 14, 35)
    if exp is None or exp > last_day:
        return None
    ch = g[g["expiry"] == exp]
    S = float(spot.get(d, ch["S"].median()))
    sh = ch[(ch["strike"] < S) & ch["delta"].between(-0.45, -0.03)]
    if sh.empty:
        return None
    s = sh.loc[(sh["delta"] + spec["short_delta"]).abs().idxmin()]
    lg = ch[ch["strike"] < s.strike]
    if lg.empty:
        return None
    l_delta = lg.loc[(lg["delta"] + spec["long_delta"]).abs().idxmin()]
    if W is None:
        l = l_delta
    else:
        l_abs = lg.loc[(lg["strike"] - (s.strike - W)).abs().idxmin()]
        l = l_abs if l_abs.strike >= l_delta.strike else l_delta
    width = float(s.strike - l.strike)
    credit_mid = float(s.mid - l.mid)
    credit = credit_mid - IB.FILL * float((s.ask - s.bid) + (l.ask - l.bid)) / 2 - 2 * IB.FEE
    if not (0 < credit < width) or credit_mid <= 0:
        return None
    settle = spot.loc[:exp]
    if settle.empty or settle.index[-1] < exp - pd.Timedelta(days=4):
        return None
    out, how = exp, "expiry"
    cost = min(width, max(0.0, s.strike - float(settle.iloc[-1])))
    if 0 < cost < width:
        cost += 2 * IB.FEE
    for day in spot.loc[d:exp].index[1:]:
        if day >= exp:
            break
        try:
            a, b = ix.loc[(day, exp, s.strike)], ix.loc[(day, exp, l.strike)]
        except KeyError:
            continue
        mc = max(float(a["mid"] - b["mid"]), 0.0)
        pay = max(float(a["mid"] - b["mid"]) + IB.FILL * float((a["ask"] - a["bid"]) + (b["ask"] - b["bid"])) / 2,
                  0.0) + 2 * IB.FEE
        if should_take_profit(credit_mid, mc, (day - d).days, (exp - d).days, "dynamic", 0.50, 0.20, 0.75):
            out, cost, how = day, pay, "take-profit"
            break
    return {"date": d, "expiry": exp, "exit": out, "how": how, "S": S, "short": float(s.strike),
            "long": float(l.strike), "width": width, "credit": credit, "cost": cost,
            "ror_net": (credit - cost) / (width - credit)}


# ---------------------------------------------------------------- [AG-36] portfolio + checks
def portfolio(cands, capital, spec):
    """[AG-36] walk entries by (date, ROOTS order): one open position per ETF; contracts = floor(budget / max loss)
    (1 if max loss <= 1.15 x budget, else skip); skip if total open risk would pass the cap"""
    budget, cap = capital * spec["risk_per_trade"], capital * spec["open_risk_cap"]
    order = {r: i for i, r in enumerate(ROOTS)}
    c = cands.assign(_o=cands["root"].map(order)).sort_values(["date", "_o"])
    busy, open_pos, taken, skipped_cap = {}, [], [], 0
    for _, r in c.iterrows():
        if r["date"] <= busy.get(r["root"], pd.Timestamp("1900-01-01")):
            continue
        ml = (r["width"] - r["credit"]) * 100
        n = int(budget // ml) if ml > 0 else 0
        if n == 0:
            if ml > budget * 1.15:
                continue
            n = 1
        open_pos = [(e, k) for e, k in open_pos if e >= r["date"]]
        if sum(k for _, k in open_pos) + n * ml > cap + 1e-9:
            skipped_cap += 1
            continue
        open_pos.append((r["exit"], n * ml))
        busy[r["root"]] = r["exit"]
        taken.append({**r.drop("_o").to_dict(), "contracts": n, "risk_usd": n * ml,
                      "pnl_usd": n * (r["credit"] - r["cost"]) * 100})
    return pd.DataFrame(taken), skipped_cap


def cluster_boot(x, groups, reps=3000, seed=0):
    """95% interval of the mean, resampling whole groups (entry weeks) so same-week trades move together"""
    x, groups = np.asarray(x, float), np.asarray(groups)
    if len(x) < 5:
        return np.nan, np.nan
    keys = np.unique(groups)
    by = {k: x[groups == k] for k in keys}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(reps):
        pick = rng.choice(keys, len(keys))
        v = np.concatenate([by[k] for k in pick])
        means.append(v.mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score(t, capital, start, end):
    """the 4 pre-registered checks + what the account would have made"""
    if t.empty:
        return {"n": 0, "verdict": "FAIL (no trades)"}
    wk = t["date"].dt.to_period("W-FRI").astype(str)
    lo, hi = cluster_boot(t["ror_net"], wk)
    pnl = t.groupby("exit")["pnl_usd"].sum().sort_index()
    eq = capital + pnl.cumsum()
    dd = float((eq / eq.cummax().clip(lower=capital) - 1).min())
    month = pnl.groupby([pnl.index.year, pnl.index.month]).sum() / capital
    yrs = (end - start).days / 365.25
    per_wk = wk.map(wk.value_counts())
    res = {"n": len(t), "trades_per_year": len(t) / yrs, "mean_ror": float(t["ror_net"].mean()), "ci_lo": lo,
           "ci_hi": hi, "win": float((t["ror_net"] > 0).mean()), "total_usd": float(pnl.sum()),
           "total_pct": float(pnl.sum() / capital), "per_year_usd": float(pnl.sum() / yrs),
           "per_year_pct": float(pnl.sum() / capital / yrs), "maxdd": dd, "worst_month": float(month.min()),
           "same_week_share": float((per_wk >= 2).mean()),
           "yearly_usd": {int(k): round(float(v), 2) for k, v in pnl.groupby(pnl.index.year).sum().items()}}
    res["checks"] = {"1 mean ror > 0, week-clustered 95% interval above 0": bool(lo > 0),
                     "2 account positive over the period": bool(pnl.sum() > 0),
                     "3 max drawdown better than -30%": bool(dd > -0.30),
                     "4 worst month better than -25%": bool(res["worst_month"] > -0.25)}
    res["verdict"] = "PASS" if all(res["checks"].values()) else "FAIL"
    return res


def width_for(capital, spec):
    return int(capital * spec["risk_per_trade"] // 100)


# ---------------------------------------------------------------- stages
def validate_spy():
    """SPY only, no ETF data needed: (A) index_branch's own chain/features + this file's trade() must reproduce the
    capital-stage trades for 2024-2026; (B) this file's whole pipeline on SPY must give the same trades"""
    spec, sha = prereg()
    print(f"pre-registration {PREREG} sha256 {sha}\n")
    IB.FILL = 1.0
    un = pd.read_parquet("data/raw/prices/daily/SPY_unadj.parquet")
    un = un.set_index("date") if "date" in un.columns else un
    un.index = pd.to_datetime(un.index).normalize()
    un = un["TRDPRC_1"].astype(float)
    hist = IB.lseg_series("spy")
    t0 = time.time()
    puts = IB.load_opra(str(SPY_CSV), un)
    chain_spot = puts.groupby("date")["S"].median()
    spot = chain_spot.combine_first(hist).sort_index()
    f = IB.features(puts, hist.combine_first(chain_spot).sort_index())
    by = dict(tuple(puts.groupby("date")))
    ix = puts.drop_duplicates(["date", "expiry", "strike"]).set_index(["date", "expiry", "strike"]).sort_index()
    dates = IB.entry_dates(f, "shock+guard")
    print(f"index_branch inputs: {len(puts):,} chain rows, {len(dates)} shock+guard entry days ({time.time()-t0:.0f}s)")
    last = pd.Timestamp("2026-11-02")
    ok_all = True
    cols = ["date", "short", "long", "credit", "cost", "ror_net"]
    for cap in (2500, 5000, 10000, 25000):
        W = width_for(cap, spec)
        ref = pd.read_csv(IB.OUT / f"capital_{cap}_trades.csv", parse_dates=["date", "expiry", "exit"])
        ref = ref[ref["date"] >= "2024-01-01"].reset_index(drop=True)
        # capital stage: W >= 15 -> delta leg only, W < 15 -> $W leg, filter max loss <= 1.15 budget
        Wc = None if W >= 15 else W
        rows = [trade(d, by, ix, spot, Wc, spec, last) for d in dates]
        t = pd.DataFrame([r for r in rows if r])
        if W < 15 and len(t):
            t = t[(t["width"] - t["credit"]) * 100 <= cap * 0.10 * 1.15]
        t = t.reset_index(drop=True)
        same = len(t) == len(ref) and np.allclose(t[cols[1:]].values, ref[cols[1:]].values) and \
            (t["date"].values == ref["date"].values).all()
        ok_all &= same
        print(f"  A ${cap:>6,}: {len(t)} trades vs index_branch {len(ref)} -> {'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            print(t[cols].to_string(), "\n", ref[cols].to_string())
    # (B) this file's loader + features + entries + portfolio on SPY; unadjusted LSEG SPY stands in for adjusted
    raw = pd.read_csv(SPY_CSV, header=None, names=["date", "ticker", "volume", "open", "close", "high", "low", "ws", "tx"],
                      usecols=["date", "ticker", "volume", "close"])
    raw = raw[(raw["volume"] > 0) & (raw["close"] > 0)]
    pt = parse_tickers(raw["ticker"], "SPY")
    raw = raw.assign(expiry=pt["expiry"], strike=pt["strike"]).dropna(subset=["expiry"])
    raw["date"] = pd.to_datetime(raw["date"])
    ch = chain(raw[["date", "expiry", "strike", "close"]], un, "SPY", spec)
    a = puts.sort_values(["date", "expiry", "strike"]).reset_index(drop=True)
    b = ch.sort_values(["date", "expiry", "strike"]).reset_index(drop=True)
    same_chain = len(a) == len(b) and np.allclose(a[["bid", "ask", "delta"]].values, b[["bid", "ask", "delta"]].values,
                                                  equal_nan=True)
    print(f"  B chain: {len(b):,} rows vs {len(a):,} -> {'IDENTICAL' if same_chain else 'DIFFERENT'}")
    fb = features_etf(ch, hist, hist, spec, is_spy=True)
    db = entries(fb)
    same_dates = list(db) == list(dates)
    print(f"  B entry days: {len(db)} vs {len(dates)} -> {'IDENTICAL' if same_dates else 'DIFFERENT'}")
    if not same_dates:
        print("   only here:", sorted(set(db) - set(dates)), " only index_branch:", sorted(set(dates) - set(db)))
    byb = dict(tuple(ch.groupby("date")))
    ixb = ch.drop_duplicates(["date", "expiry", "strike"]).set_index(["date", "expiry", "strike"]).sort_index()
    spotb = un.combine_first(hist).sort_index()
    for cap in (2500, 5000, 10000):
        W = width_for(cap, spec)
        rows = [trade(d, byb, ixb, spotb, W, spec, last) for d in db]
        c = pd.DataFrame([r for r in rows if r]).assign(root="SPY")
        tk, _ = portfolio(c, cap, spec) if len(c) else (pd.DataFrame(), 0)
        ref = pd.read_csv(IB.OUT / f"capital_{cap}_trades.csv", parse_dates=["date"])
        ref = ref[ref["date"] >= "2024-01-01"]
        print(f"  B ${cap:>6,}: portfolio took {len(tk)} trades {list(tk['date'].dt.date.astype(str)) if len(tk) else []}"
              f" | index_branch {len(ref)} {list(ref['date'].dt.date.astype(str))}")
    # how the vol-scaled shock compares on SPY's long history (the scaled rule is identical for SPY by construction;
    # this shows what an ABSOLUTE scaling by 1% would have changed, for the record)
    px = hist.loc["2009-01-01":]
    fixed = ((px.pct_change() <= -0.015) | (px.pct_change(2) <= -0.025)).loc["2010-01-01":]
    s_abs = trailing_sigma(px) / 0.01
    absd = ((px.pct_change() <= -0.015 * s_abs) | (px.pct_change(2) <= -0.025 * s_abs)).loc["2010-01-01":]
    print(f"\n  SPY shock days 2010-2026: fixed rule {int(fixed.sum())}, relative-to-SPY scaling {int(fixed.sum())} "
          f"(same by construction), absolute scaling by 1% would give {int(absd.sum())}")
    print(f"\nVALIDATION: {'PASS' if ok_all and same_chain and same_dates else 'CHECK DIFFERENCES ABOVE'}")


def run(capitals):
    spec, sha = prereg()
    print(f"pre-registration {PREREG} sha256 {sha}  (evidence, not proof: ~2.5 years of option data)\n", flush=True)
    IB.FILL = 1.0
    split_months()
    spy_adj, _ = load_prices("SPY")
    if spy_adj is None:
        sys.exit("SPY prices missing in data/raw/prices_bulk: run scripts/download_etf_prices.py")
    cands, per = [], []
    for root in ROOTS:
        t0 = time.time()
        f = OUT / "cache" / f"{root}.parquet"
        adj, unadj = load_prices(root)
        if not f.exists() or adj is None:
            print(f"  {root:5s} skipped (no {'options' if not f.exists() else 'prices'})", flush=True)
            continue
        raw = pd.read_parquet(f)
        ch = chain(raw, unadj, root, spec)
        if ch.empty:
            continue
        fe = features_etf(ch, adj, spy_adj, spec, is_spy=(root == "SPY"))
        days = entries(fe)
        by = dict(tuple(ch.groupby("date")))
        ix = ch.drop_duplicates(["date", "expiry", "strike"]).set_index(["date", "expiry", "strike"]).sort_index()
        last = min(ch["date"].max(), unadj.index.max())
        for cap in capitals:
            W = width_for(cap, spec)
            for d in days:
                r = trade(d, by, ix, unadj, W, spec, last)
                if r:
                    cands.append({**r, "root": root, "capital": cap})
        per.append({"root": root, "chain_days": int(ch["date"].nunique()), "shock_days": int(fe["shock"].sum()),
                     "entry_days": len(days), "median_scale": float(fe["scale"].median())})
        print(f"  {root:5s} {ch['date'].nunique():4d} days, {int(fe['shock'].sum()):3d} shocks, {len(days):3d} entries "
              f"({time.time()-t0:.0f}s)", flush=True)
        del ch, by, ix, raw
    OUT.mkdir(parents=True, exist_ok=True)
    C = pd.DataFrame(cands)
    if C.empty:
        sys.exit("no candidate trades: check the option and price files")
    C.to_csv(OUT / "candidates.csv", index=False)
    pd.DataFrame(per).to_csv(OUT / "per_etf_signals.csv", index=False)
    start = pd.Timestamp(spec["period"][0])
    end = min(C["exit"].max(), pd.Timestamp(spec["period"][1])) if len(C) else pd.Timestamp(spec["period"][1])
    summary = {"prereg_sha": sha}
    for cap in capitals:
        c = C[C["capital"] == cap].drop(columns="capital")
        t, skipped = portfolio(c, cap, spec)
        t.to_csv(OUT / f"run_{cap}_trades.csv", index=False)
        s = score(t, cap, start, end)
        s["skipped_by_cap"] = skipped
        s["candidates"] = len(c)
        summary[str(cap)] = s
        print(f"\nETF BRANCH: ${cap:,}, width ${width_for(cap, spec)} (or the 0.03-delta put if narrower), "
              f"{start.date()}..{end.date()}")
        if not s["n"]:
            print("  no trades")
            continue
        for k, v in s["checks"].items():
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
        print(f"  {s['n']} trades ({s['trades_per_year']:.1f}/yr, {skipped} skipped by the 30% cap, "
              f"{s['same_week_share']:.0%} in weeks with 2+ trades) | mean ror {s['mean_ror']:+.2%} "
              f"[{s['ci_lo']:+.2%}, {s['ci_hi']:+.2%}] win {s['win']:.0%}")
        print(f"  account: ${s['total_usd']:+,.0f} ({s['total_pct']:+.1%}) = ${s['per_year_usd']:+,.0f}/yr "
              f"({s['per_year_pct']:+.2%}/yr), max dd {s['maxdd']:+.1%}, worst month {s['worst_month']:+.1%}")
        print(f"  yearly $ {s['yearly_usd']}")
        by_root = t.groupby("root").agg(n=("ror_net", "size"), mean=("ror_net", "mean"), pnl=("pnl_usd", "sum"))
        print("  per ETF: " + ", ".join(f"{r} {int(x.n)} ({x['mean']:+.1%}, ${x.pnl:+.0f})" for r, x in by_root.iterrows()))
        print(f"  VERDICT: {s['verdict']} (evidence, not proof)")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nsaved {OUT}/run_<capital>_trades.csv, candidates.csv, per_etf_signals.csv, summary.json")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate-spy", action="store_true")
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--capital", type=float, nargs="+")
    a = ap.parse_args()
    if a.validate_spy:
        validate_spy()
    elif a.prepare:
        split_months(force=True)
    else:
        spec, _ = prereg()
        run([int(c) for c in (a.capital or spec["capitals"])])


if __name__ == "__main__":
    main()
