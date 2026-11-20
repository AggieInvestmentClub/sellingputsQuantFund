"""
Simulated 1990-2009 test of the index-branch rule (strategy/preregistered_simulated_1990_2009.md).

  AG-64 calibrate(): VIX -> ATM IV ratio, volatility smile and half-spread tables from REAL Kaggle chains 2010-2023
  AG-65 SimMarket / sim_trade(): Black-Scholes pricing on modelled IV, 5-pt SPX strike grid, monthly expiries,
        modelled bid/ask (wider before 2005), dynamic take-profit, expiry settlement
  AG-66 step 1 (calibration check vs real chains 2010-2023) and step 2 (1990-2009: 4 checks + permutation test)

    py -3.12 -u scripts/sim_backtest.py
Needs data/raw/market_context/spx_1988_2010.parquet (scripts/download_spx_history.py) and vix_daily_cboe.csv.
Output: data/results/sim_1990_2009/
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm, spearmanr  # noqa: E402

import scripts.index_branch as IB  # noqa: E402
from strategy.strategy_logic import should_take_profit  # noqa: E402

PREREG = Path("strategy/preregistered_simulated_1990_2009.md")
OUT = Path("data/results/sim_1990_2009")
MC = Path("data/raw/market_context")


def load_spec():
    text = PREREG.read_text(encoding="utf-8")
    return (json.loads(re.search(r"```json\s*(\{.*?\})\s*```", text, re.S).group(1)),
            hashlib.sha256(PREREG.read_bytes()).hexdigest()[:16])


def load_vix():
    v = pd.read_csv(MC / "vix_daily_cboe.csv", parse_dates=["DATE"]).set_index("DATE")["CLOSE"].astype(float)
    return v[~v.index.duplicated()].sort_index()


def third_fridays(start, end):
    out = []
    for p in pd.period_range(start, end, freq="M"):
        d = pd.Timestamp(p.year, p.month, 15)
        out.append(d + pd.Timedelta(days=(4 - d.weekday()) % 7))
    return pd.DatetimeIndex(out)


# ---------------------------------------------------------------- Black-Scholes (r = 0)
def bs_put(S, K, T, s):
    if T <= 0 or s <= 0:
        return max(K - S, 0.0)
    v = s * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v) / v
    return K * norm.cdf(-(d1 - v)) - S * norm.cdf(-d1)


def bs_put_delta(S, K, T, s):
    if T <= 0 or s <= 0:
        return -1.0 if K > S else 0.0
    v = s * math.sqrt(T)
    return norm.cdf((math.log(S / K) + 0.5 * v * v) / v) - 1.0


# ---------------------------------------------------------------- AG-64 calibration (real chains only)
def regime(vix, cuts):
    return 0 if vix < cuts[0] else (1 if vix < cuts[1] else 2)


def calibrate(spec, vix):
    lo, hi = spec["calib"]
    k = IB.load_kaggle("data/raw/kaggle_spy")
    k = k[(k["date"] >= lo) & (k["date"] <= hi) & (k["bid"] > 0) & (k["ask"] >= k["bid"]) & (k["iv"] > 0)]
    k["dte"] = (k["expiry"] - k["date"]).dt.days
    k = k[k["dte"].between(14, 35)]
    k["gap"] = (k["dte"] - 21).abs()
    best = k.groupby("date")["gap"].transform("min")
    k = k[k["gap"] == best]
    k = k[k["expiry"] == k.groupby("date")["expiry"].transform("min")]
    atm = k.loc[(k["delta"] + 0.5).abs().groupby(k["date"]).idxmin(), ["date", "iv"]].set_index("date")["iv"]
    k["atm"] = k["date"].map(atm)
    k["vix"] = k["date"].map(vix.reindex(vix.index.union(atm.index)).ffill())
    k = k.dropna(subset=["atm", "vix"])
    k["reg"] = [regime(v, spec["regimes"]) for v in k["vix"]]
    k["T"] = k["dte"] / 365.0
    k["z"] = np.log(k["strike"] / k["S"]) / (k["atm"] * np.sqrt(k["T"]))
    k["ratio"] = k["iv"] / k["atm"]
    k["hs"] = (k["ask"] - k["bid"]) / 2 / k["mid"].where(k["mid"] > 0)
    zlo, zhi, zstep = spec["z_bins"]
    edges = np.arange(zlo, zhi + 1e-9, zstep)
    k = k[(k["z"] >= zlo) & (k["z"] <= zhi)]
    k["zb"] = np.clip(np.digitize(k["z"], edges) - 1, 0, len(edges) - 2)
    centers = (edges[:-1] + edges[1:]) / 2
    day = k.drop_duplicates("date")
    atm_k = {r: float((day.loc[day["reg"] == r, "atm"] / (day.loc[day["reg"] == r, "vix"] / 100)).median()) for r in range(3)}
    smile, spread = {}, {}
    for r in range(3):
        g = k[k["reg"] == r]
        sm = g.groupby("zb")["ratio"].median().reindex(range(len(centers)))
        hs = g.groupby("zb")["hs"].median().reindex(range(len(centers)))
        smile[r] = sm.interpolate().bfill().ffill().values.tolist()
        spread[r] = hs.interpolate().bfill().ffill().values.tolist()
    model = {"centers": centers.tolist(), "atm_k": atm_k, "smile": smile, "spread": spread,
             "n_days": int(day.shape[0]), "n_quotes": int(len(k))}
    return model


# ---------------------------------------------------------------- AG-65 simulated market
class SimMarket:
    def __init__(self, model, spec, vix, grid):
        self.m, self.spec, self.vix, self.grid = model, spec, vix, grid
        self.centers = np.array(model["centers"])

    def atm_iv(self, d):
        v = float(self.vix.asof(d))
        r = regime(v, self.spec["regimes"])
        return v / 100 * self.m["atm_k"][r], r

    def iv(self, S, K, T, d):
        a, r = self.atm_iv(d)
        z = math.log(K / S) / (a * math.sqrt(max(T, 1e-6)))
        return a * float(np.interp(z, self.centers, self.m["smile"][r])), z, r

    def quote(self, S, K, T, d):
        s, z, r = self.iv(S, K, T, d)
        mid = bs_put(S, K, T, s)
        mult = 2.0 if d < pd.Timestamp("2001-04-09") else (1.5 if d < pd.Timestamp("2005-01-01") else 1.0)
        tick = (0.0625 if d < pd.Timestamp("2001-04-09") else 0.05) if self.grid >= 5 else 0.01
        hs = max(mult * float(np.interp(z, self.centers, self.m["spread"][r])) * mid, tick)
        return mid, hs, s

    def pick_strikes(self, S, T, d):
        ks = np.arange(math.floor(S * 0.5 / self.grid) * self.grid, math.floor(S / self.grid) * self.grid, self.grid)
        if len(ks) < 3:
            return None
        deltas = np.array([bs_put_delta(S, k, T, self.iv(S, k, T, d)[0]) for k in ks])
        ks_s = ks[np.argmin(np.abs(deltas + self.spec["short_delta"]))]
        lg = ks < ks_s
        if not lg.any():
            return None
        ks_l = ks[lg][np.argmin(np.abs(deltas[lg] + self.spec["long_delta"]))]
        return float(ks_s), float(ks_l)


def sim_trade(mkt, d, px, expiries, fee):
    lo, hi, tgt = mkt.spec["dte"]
    cand = [e for e in expiries if lo <= (e - d).days <= hi]
    if not cand:
        return None
    exp = min(cand, key=lambda e: abs((e - d).days - tgt))
    if exp > px.index[-1]:
        return None
    S = float(px.loc[d])
    T = (exp - d).days / 365.0
    st = mkt.pick_strikes(S, T, d)
    if st is None:
        return None
    ks, kl = st
    ms, hs_s, _ = mkt.quote(S, ks, T, d)
    ml, hs_l, _ = mkt.quote(S, kl, T, d)
    credit_mid = ms - ml
    credit = credit_mid - hs_s - hs_l - 2 * fee
    width = ks - kl
    if not (0 < credit < width):
        return None
    out, how = exp, "expiry"
    settle = float(px.asof(exp))
    cost = min(width, max(0.0, ks - settle))
    if 0 < cost < width:
        cost += 2 * fee
    for day in px.loc[d:exp].index[1:]:
        if day >= exp:
            break
        Sd, Td = float(px.loc[day]), (exp - day).days / 365.0
        a, ha, _ = mkt.quote(Sd, ks, Td, day)
        b, hb, _ = mkt.quote(Sd, kl, Td, day)
        mc = max(a - b, 0.0)
        if should_take_profit(credit_mid, mc, (day - d).days, (exp - d).days, "dynamic", 0.50, 0.20, 0.75):
            out, cost, how = day, max(a - b + ha + hb, 0.0) + 2 * fee, "take-profit"
            break
    return {"date": d, "expiry": exp, "exit": out, "how": how, "S": S, "short": ks, "long": kl, "width": width,
            "credit": credit, "cost": cost, "ror_net": (credit - cost) / (width - credit)}


def sim_features(px):
    f = pd.DataFrame({"S": px})
    f["ma200"] = px.rolling(200).mean()
    f["ret1"], f["ret2"] = px.pct_change(), px.pct_change(2)
    f["shock"] = (f["ret1"] <= IB.SHOCK_1D) | (f["ret2"] <= IB.SHOCK_2D)
    f["guard_ok"] = f["S"] > f["ma200"]                       # MA-only guard (9d IV unavailable before 2011)
    f["vrp"] = np.nan
    return f


def account(t, risk):
    """one at a time (already applied), risk fraction per trade, simple sum by exit month/year"""
    if t is None or t.empty:
        return {"total": 0.0, "maxdd": 0.0, "worst_month": 0.0, "years_pos": "0/0", "years_pos_share": 0.0, "yearly": {}}
    pnl = pd.Series(t["ror_net"].values * risk, index=pd.to_datetime(t["exit"].values)).sort_index()
    eq = 1 + pnl.cumsum()
    dd = float((eq / eq.cummax().clip(lower=1) - 1).min())
    monthly = pnl.groupby(pnl.index.to_period("M")).sum()
    yearly = pnl.groupby(pnl.index.year).sum()
    return {"total": float(pnl.sum()), "maxdd": min(dd, 0.0), "worst_month": float(monthly.min()),
            "years_pos": f"{int((yearly > 0).sum())}/{len(yearly)}", "years_pos_share": float((yearly > 0).mean()),
            "yearly": {int(k): float(v) for k, v in yearly.items()}}


def boot_ci(x, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(np.mean(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ---------------------------------------------------------------- AG-66 steps
def step1(spec, model, vix):
    """calibration check: same MA-only rule, simulator vs REAL chains, 2010-2023, monthly expiries only"""
    lo, hi = spec["calib"]
    f, by, ix, spot = IB.build("validate") if False else IB.build("walkforward")
    f = f[(f.index >= lo) & (f.index <= hi)].copy()
    monthly = set(third_fridays(lo, "2024-03-31")) | set(third_fridays(lo, "2024-03-31") - pd.Timedelta(days=1))
    by_m = {d: g[g["expiry"].isin(monthly)] for d, g in by.items() if d >= pd.Timestamp(lo) and d <= pd.Timestamp(hi)}
    ix_m = ix[ix.index.get_level_values("expiry").isin(monthly)]
    f["guard_ok"] = f["S"] > f["ma200"]
    dates = IB.entry_dates(f, "shock+guard")
    real = [IB.trade(d, by_m, ix_m, spot, (0.10, ("delta", 0.03)), "tp") for d in dates]
    real = pd.DataFrame([r for r in real if r])
    px = spot[(spot.index >= lo)].sort_index()
    mkt = SimMarket(model, spec, vix, grid=1)
    sim = pd.DataFrame([r for r in (sim_trade(mkt, d, px, sorted(monthly), IB.FEE) for d in dates) if r])
    m = real.merge(sim, on="date", suffixes=("_real", "_sim"))
    rho = float(spearmanr(m["ror_net_real"], m["ror_net_sim"]).correlation) if len(m) > 3 else float("nan")
    gap = float(sim["ror_net"].mean() - real["ror_net"].mean())
    ok = abs(gap) <= spec["calib_check"]["max_mean_gap"] and rho >= spec["calib_check"]["min_spearman"]
    real.to_csv(OUT / "step1_real_trades.csv", index=False)
    sim.to_csv(OUT / "step1_sim_trades.csv", index=False)
    return {"n_real": len(real), "n_sim": len(sim), "n_common": len(m), "real_mean": float(real["ror_net"].mean()),
            "sim_mean": float(sim["ror_net"].mean()), "gap": gap, "spearman": rho,
            "credit_ratio_median": float((m["credit_sim"] / m["credit_real"]).median()) if len(m) else float("nan"),
            "ok": bool(ok)}


def step2(spec, model, vix, px_all):
    lo, hi = spec["period"]
    px = px_all[(px_all.index >= "1989-01-01") & (px_all.index <= "2010-03-31")]
    f = sim_features(px)
    f = f[(f.index >= lo) & (f.index <= hi)]
    exps = sorted(third_fridays("1990-01-01", "2010-03-31"))
    mkt = SimMarket(model, spec, vix, grid=spec["strike_grid"])
    fee = 0.0065
    # every tradable day once (pool), then the signal subset
    pool = []
    for d in f.index:
        if pd.isna(vix.asof(d)):
            continue
        r = sim_trade(mkt, d, px, exps, fee)
        if r:
            pool.append(r)
    pool = pd.DataFrame(pool).sort_values("date").reset_index(drop=True)
    sig = set(IB.entry_dates(f, "shock+guard"))
    pool["is_signal"] = pool["date"].isin(sig)
    pool["guard"] = f["guard_ok"].reindex(pool["date"]).fillna(False).values
    pool.to_csv(OUT / "pool_1990_2009.csv", index=False)
    t = IB.one_open(pool[pool["is_signal"]])
    t.to_csv(OUT / "trades_1990_2009.csv", index=False)
    risk = spec["risk"]
    mean, cl, ch = boot_ci(t["ror_net"].values)
    acct = account(t, risk)
    checks = [("1. mean ror > 0 with 95% interval above 0", cl > 0, f"{mean:+.2%} [{cl:+.2%}, {ch:+.2%}], n={len(t)}"),
              ("2. >= 70% of years with trades positive", acct["years_pos_share"] >= 0.7, acct["years_pos"]),
              ("3. worst month better than -25%", acct["worst_month"] > -0.25, f"{acct['worst_month']:+.1%}"),
              ("4. max drawdown better than -30%", acct["maxdd"] > -0.30, f"{acct['maxdd']:+.1%}")]
    # permutation (same machinery as scripts/permutation_test.py, risk-normalised P&L)
    from scripts.permutation_test import one_open_np
    years = (pool["exit"].max() - pool["date"].min()).days / 365.25
    dates, exits, ror = pool["date"].values, pool["exit"].values, pool["ror_net"].values
    sig_idx = np.flatnonzero(pool["is_signal"].values)
    kept = one_open_np(dates, exits, sig_idx)

    def cagr(idx):
        tot = 1 + ror[idx].sum() * risk
        return tot ** (1 / years) - 1 if tot > 0 else -1.0
    real = cagr(kept)
    rng = np.random.default_rng(spec["seed"])
    perm = {}
    for name, mask in (("all_days", None), ("guard_days", pool["guard"].values)):
        idx = np.flatnonzero(mask) if mask is not None else np.arange(len(pool))
        dist = np.empty(spec["n_perm"])
        for k in range(spec["n_perm"]):
            pick = np.sort(rng.choice(idx, size=min(len(sig_idx), len(idx)), replace=False))
            dist[k] = cagr(one_open_np(dates, exits, pick))
        kk = int((dist >= real).sum())
        perm[name] = {"k": kk, "p": (1 + kk) / (1 + len(dist)), "median": float(np.median(dist)),
                      "p5": float(np.percentile(dist, 5)), "p95": float(np.percentile(dist, 95))}
        pd.Series(dist).to_csv(OUT / f"perm_{name}.csv", index=False)
    crashes = {"1997-10": ("1997-09-15", "1997-11-30"), "1998-08..10": ("1998-07-15", "1998-10-31"),
               "2000-2002": ("2000-03-01", "2002-10-31"), "2007-08": ("2007-07-15", "2007-09-15"),
               "2008-09..2009-03": ("2008-08-15", "2009-03-31")}
    cw = {k: {"trades": int(((t["date"] >= a) & (t["date"] <= b)).sum()),
              "sum_ror": float(t.loc[(t["date"] >= a) & (t["date"] <= b), "ror_net"].sum())} for k, (a, b) in crashes.items()}
    return {"pool": len(pool), "signals": len(sig), "trades": len(t), "full_losses": int((t["ror_net"] <= -0.9).sum()),
            "worst": float(t["ror_net"].min()), "mean": mean, "ci": [cl, ch], "account": acct,
            "checks": [[n, bool(o), d] for n, o, d in checks], "real_cagr": real, "perm": perm, "crash_windows": cw,
            "four_checks": all(c[1] for c in checks)}


def spx_src(mult):
    return "SPY x 10" if mult == 10 else ".SPX"


def main():
    spec, sha = load_spec()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"pre-registration {PREREG} sha256 {sha} (SIMULATED prices: evidence, not proof)", flush=True)
    vix = load_vix()
    if (MC / "spx_1988_2010.parquet").exists():
        spx = pd.read_parquet(MC / "spx_1988_2010.parquet")
        mult = 10.0 if str(spx["ric"].iloc[0]) == "SPY" else 1.0     # LSEG gave SPY when .SPX was not permitted
        spx = (spx.set_index(pd.to_datetime(spx["date"]).dt.normalize())["close"].astype(float) * mult).sort_index()
        print(f"SPX: LSEG {spx_src(mult)} closes", flush=True)
    else:   # addendum: 10 x unadjusted SPY from 1993-01-29
        spy = pd.read_parquet(MC / "spy.parquet")
        spx = spy["TRDPRC_1"].astype(float) * 10
        spx.index = pd.to_datetime(spx.index).normalize()
        spx = spx.sort_index()
        print(f"SPX: 10 x unadjusted SPY (addendum), from {spx.index.min().date()}", flush=True)
    spx = spx[~spx.index.duplicated()].dropna()
    model = calibrate(spec, vix)
    (OUT / "model.json").write_text(json.dumps(model, indent=1))
    print(f"model fitted on {model['n_days']} days / {model['n_quotes']:,} real quotes: ATM IV / VIX by regime "
          f"{ {k: round(v, 3) for k, v in model['atm_k'].items()} }", flush=True)
    s1 = step1(spec, model, vix)
    print(f"\nSTEP 1 calibration check (2010-2023, MA-only guard, monthly expiries): real {s1['n_real']} trades mean "
          f"{s1['real_mean']:+.2%} | simulated {s1['n_sim']} mean {s1['sim_mean']:+.2%} | gap {s1['gap']:+.2%} | "
          f"Spearman {s1['spearman']:.2f} on {s1['n_common']} common dates | sim/real credit {s1['credit_ratio_median']:.2f}"
          f" -> {'USABLE' if s1['ok'] else 'MODEL NOT VALIDATED'}", flush=True)
    s2 = step2(spec, model, vix, spx)
    print(f"\nSTEP 2 simulated 1990-2009: {s2['pool']} tradable days, {s2['signals']} signals, {s2['trades']} trades, "
          f"{s2['full_losses']} full losses (worst {s2['worst']:+.0%})")
    for n, o, d in s2["checks"]:
        print(f"  {'PASS' if o else 'FAIL'}  {n}: {d}")
    a = s2["account"]
    print(f"  account at 10% risk: {a['total']:+.1%} over 20 years, max DD {a['maxdd']:+.1%}; yearly "
          + ", ".join(f"{y}: {v:+.1%}" for y, v in a["yearly"].items()))
    for k, v in s2["crash_windows"].items():
        print(f"  crash window {k}: {v['trades']} trades, sum ror {v['sum_ror']:+.0%}")
    for name, r in s2["perm"].items():
        print(f"  permutation null {name}: {r['k']}/{spec['n_perm']} >= real -> p = {r['p']:.4f} "
              f"(real CAGR {s2['real_cagr']:+.3%}, permuted median {r['median']:+.3%})")
    passed = s1["ok"] and s2["four_checks"] and s2["perm"]["all_days"]["p"] < spec["alpha"]
    print(f"\nVERDICT (simulated, evidence only): {'PASS' if passed else 'FAIL'}"
          + ("" if s1["ok"] else " (model not validated in step 1)"))
    (OUT / "summary.json").write_text(json.dumps({"sha": sha, "step1": s1, "step2": s2, "verdict": "PASS" if passed else "FAIL"},
                                                 indent=1, default=str))


if __name__ == "__main__":
    main()
