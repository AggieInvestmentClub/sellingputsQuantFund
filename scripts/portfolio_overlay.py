"""
Portfolio overlay and tail stress (AG-56..60), pre-registered in strategy/preregistered_portfolio_overlay.md.

  AG-56 capital-rule-v2 trades at several fills (index_branch.capital_structure / within_budget)
  AG-57 daily account: idle cash earns the 3-month T-bill (US3MT=RR, act/360), collateral = max loss earns nothing
  AG-58 fill model: 50% quarter-from-mid, 35% halfway, 15% unfilled (1,000 seeded runs)
  AG-59 checks vs T-bills only: bootstrap of yearly excess, years positive, drawdown, deflated Sharpe (n_trials 200)
  AG-60 tail stress: Jeffreys-posterior full-loss rate (0 losses seen) + guard-off historical crash trades

    py -3.12 -u scripts/portfolio_overlay.py
Output: data/results/portfolio_overlay/ (report.json, run.log, account_<capital>.csv, stress.csv)
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
from scipy.stats import kurtosis, skew  # noqa: E402

import scripts.index_branch as IB  # noqa: E402
from src.backtest.overfit import deflated_sharpe  # noqa: E402

PREREG = Path("strategy/preregistered_portfolio_overlay.md")
OUT = Path("data/results/portfolio_overlay")
START = pd.Timestamp("2014-01-02")
END = pd.Timestamp("2026-11-18")


def load_spec():
    text = PREREG.read_text(encoding="utf-8")
    spec = json.loads(re.search(r"```json\s*(\{.*?\})\s*```", text, re.S).group(1))
    return spec, hashlib.sha256(PREREG.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- AG-56 trades
def cap_trades(f, by, ix, spot, capital, fill, entry="shock+guard"):
    IB.FILL = fill
    try:
        budget, W, struct = IB.capital_structure(capital)
        rows = [IB.trade(d, by, ix, spot, struct, "tp") for d in IB.entry_dates(f, entry)]
        t = IB.within_budget(pd.DataFrame([r for r in rows if r]), budget)
    finally:
        IB.FILL = 1.0
    if len(t):
        t = t[t["date"] >= START].copy()
        t["pnl_usd"] = t["ror_net"] * (t["width"] - t["credit"]) * 100
        t["collateral"] = (t["width"] - t["credit"]) * 100
    return t.reset_index(drop=True)


# ---------------------------------------------------------------- AG-57 account
def tbill_rate(days):
    r = pd.read_parquet("data/raw/rates/treasury_yields.parquet")["US3MT=RR"]
    r.index = pd.to_datetime(r.index).normalize()
    return r.reindex(r.index.union(days)).sort_index().ffill().shift(1).reindex(days).fillna(0.0)


def account(t, capital, days, rate, with_spreads=True):
    """daily equity: idle cash earns T-bills; collateral earns nothing; P&L booked on exit"""
    coll = np.zeros(len(days))
    pnl = np.zeros(len(days))
    if with_spreads and t is not None and len(t):
        for _, x in t.iterrows():
            i = days.searchsorted(x["date"])
            j = min(days.searchsorted(x["exit"]), len(days) - 1)
            coll[i:j] += x["collateral"]
            pnl[j] += x["pnl_usd"]
    gap = np.r_[1, np.diff(days.values).astype("timedelta64[D]").astype(int)]
    eq = np.empty(len(days))
    e = float(capital)
    for k in range(len(days)):
        idle = max(e - coll[k], 0.0)
        e += idle * rate.iloc[k] * gap[k] / 360 + pnl[k]
        eq[k] = e
    return pd.Series(eq, index=days)


def stats(eq, capital):
    eq0 = pd.concat([pd.Series([capital], index=[eq.index[0] - pd.Timedelta(days=1)]), eq])
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    yearly = eq0.resample("YE").last().pct_change().dropna()
    yearly.index = yearly.index.year
    first = eq0.resample("YE").last()
    yearly.loc[eq.index[0].year] = first.iloc[1] / capital - 1 if len(first) > 1 else 0
    yearly = yearly.sort_index()
    monthly = eq0.resample("ME").last().pct_change().dropna()
    return {"cagr": (eq.iloc[-1] / capital) ** (1 / yrs) - 1, "total": eq.iloc[-1] / capital - 1,
            "maxdd": float((eq / eq.cummax() - 1).min()), "worst_month": float(monthly.min()),
            "yearly": yearly, "monthly": monthly}


def boot_mean(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n, len(x))).mean(axis=1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def sr_trials_var():
    """variance of per-month Sharpes across the return-branch family (144 candidates), the best estimate we have"""
    try:
        import scripts.return_branch as RB
        spec, _ = RB.load_spec()
        trades = {p.stem: pd.read_parquet(p) for p in sorted((RB.OUT / "trades").glob("*.parquet"))}
        R, _ = RB.monthly_matrix(trades, spec)
        sr = R.mean(0) / np.where(R.std(0, ddof=1) > 0, R.std(0, ddof=1), np.nan)
        return float(np.nanvar(sr)), R.shape[1]
    except Exception as e:  # noqa: BLE001
        print(f"  (sr variance from return branch unavailable: {e}; using 0.01)")
        return 0.01, 0


# ---------------------------------------------------------------- AG-58 fill model
def fill_runs(sets, capital, days, rate, spec, base_tbill):
    fm = spec["fill_model"]
    rng = np.random.default_rng(42)
    keys = {fl: s.set_index("date") for fl, s in sets.items()}
    dates = sorted(set().union(*[set(s["date"]) for s in sets.values()]))
    out = []
    for _ in range(fm["runs"]):
        u = rng.random(len(dates))
        rows = []
        for d, x in zip(dates, u):
            fl = 0.25 if x < fm["0.25"] else (0.5 if x < fm["0.25"] + fm["0.5"] else None)
            if fl is not None and d in keys[fl].index:
                rows.append(keys[fl].loc[[d]].reset_index())
        t = IB.one_open(pd.concat(rows, ignore_index=True)) if rows else pd.DataFrame()
        st = stats(account(t, capital, days, rate), capital)
        ex = (st["yearly"] - base_tbill["yearly"]).reindex(st["yearly"].index)
        out.append({"cagr": st["cagr"], "maxdd": st["maxdd"], "excess": float(ex.mean()),
                    "spread_usd": float(t["pnl_usd"].sum()) if len(t) else 0.0, "trades": len(t)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- AG-60 tail stress
def stress_param(ror, risk_frac, per_year, n_seen, years=20000, seed=7):
    rng = np.random.default_rng(seed)
    ror = np.asarray(ror, float)
    risk_frac = np.broadcast_to(np.asarray(risk_frac, float), ror.shape) if np.ndim(risk_frac) else np.full(ror.shape, risk_frac)
    p = rng.beta(0.5, n_seen + 0.5, years)
    n = rng.poisson(per_year, years)
    res = np.zeros(years)
    for y in range(years):
        if n[y] == 0:
            continue
        idx = rng.integers(0, len(ror), n[y])
        r = ror[idx].copy()
        r[rng.random(n[y]) < p[y]] = -1.0
        res[y] = float((r * risk_frac[idx]).sum())
    q5 = np.percentile(res, 5)
    worst1 = res[res <= np.percentile(res, 1)]
    return {"mean": float(res.mean()), "var5": float(q5), "cvar1": float(worst1.mean()),
            "p_loss_year": float((res < 0).mean()), "p_below_-10%": float((res < -0.10).mean()),
            "p_full_loss_mean": float(p.mean())}


def main():
    spec, sha = load_spec()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"pre-registration {PREREG} sha256 {sha}")
    f, by, ix, spot = IB.build("capital_v2")
    days = pd.bdate_range(START, END)
    rate = tbill_rate(days)
    var_sr, n_fam = sr_trials_var()
    print(f"Sharpe variance across trials (return-branch family of {n_fam}): {var_sr:.4f}")
    spy = pd.read_parquet("data/raw/prices_bulk/adj/SPY.parquet")
    spy = spy.set_index(pd.to_datetime(spy["date"]).dt.normalize())["TRDPRC_1"].astype(float)
    spy = spy[~spy.index.duplicated()].reindex(days).ffill()
    spy_st = stats(spy / spy.iloc[0] * 10000, 10000)
    report = {"sha": sha, "spy": {k: spy_st[k] for k in ("cagr", "maxdd", "worst_month")}, "sizes": {}}
    print(f"SPY buy and hold 2014-2026: CAGR {spy_st['cagr']:+.2%}, max DD {spy_st['maxdd']:+.1%}")
    stress_rows = []
    for cap in spec["capitals"]:
        sets = {fl: cap_trades(f, by, ix, spot, cap, fl) for fl in (1.0, 0.5, 0.25)}
        base = IB.one_open(sets[1.0])
        base.to_csv(OUT / f"trades_{cap}_fill1.csv", index=False)
        tb = stats(account(None, cap, days, rate, with_spreads=False), cap)
        ov = stats(account(base, cap, days, rate), cap)
        excess_y = (ov["yearly"] - tb["yearly"]).dropna()
        m, lo, hi = boot_mean(excess_y.values, spec["checks"]["boot"])
        ex_m = (ov["monthly"] - tb["monthly"]).dropna()
        sr = float(ex_m.mean() / ex_m.std(ddof=1)) if ex_m.std(ddof=1) > 0 else 0.0
        dsr = deflated_sharpe(sr, len(ex_m), spec["checks"]["n_trials"], var_sr,
                              float(skew(ex_m)), float(kurtosis(ex_m, fisher=False)))
        ch = spec["checks"]
        checks = [("1. mean yearly excess over T-bills > 0, 95% interval above 0", lo > 0, f"{m:+.2%} [{lo:+.2%}, {hi:+.2%}]"),
                  (f"2. excess positive in >= {ch['years_pos']:.0%} of years", (excess_y > 0).mean() >= ch["years_pos"],
                   f"{(excess_y > 0).sum()}/{len(excess_y)}"),
                  ("3. max DD better than -30% and worst month better than -25%",
                   ov["maxdd"] > ch["max_dd"] and ov["worst_month"] > ch["worst_month"],
                   f"{ov['maxdd']:+.2%} / {ov['worst_month']:+.2%}"),
                  (f"4. deflated Sharpe of monthly excess >= {ch['dsr_min']} (n_trials {ch['n_trials']})",
                   dsr >= ch["dsr_min"], f"{dsr:.3f} (monthly SR {sr:.3f}, annualized {sr * math.sqrt(12):.2f})")]
        runs = fill_runs(sets, cap, days, rate, spec, tb)
        q = runs.quantile([0.05, 0.5, 0.95])
        print(f"\n=== ${cap:,} account, 2014-01-02 .. {END.date()} (one spread at a time, capital rule v2)")
        print(f"  T-bills only:            CAGR {tb['cagr']:+.2%}  total {tb['total']:+.1%}")
        print(f"  overlay (bid/ask fills): CAGR {ov['cagr']:+.2%}  total {ov['total']:+.1%}  max DD {ov['maxdd']:+.2%}  "
              f"trades {len(base)}  spread P&L ${base['pnl_usd'].sum():+,.0f}")
        print(f"  overlay (fill model, median [5%, 95%] of {len(runs)} runs): CAGR {q.loc[0.5, 'cagr']:+.2%} "
              f"[{q.loc[0.05, 'cagr']:+.2%}, {q.loc[0.95, 'cagr']:+.2%}], spread P&L ${q.loc[0.5, 'spread_usd']:+,.0f} "
              f"[{q.loc[0.05, 'spread_usd']:+,.0f}, {q.loc[0.95, 'spread_usd']:+,.0f}]")
        for name, ok, det in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: {det}")
        verdict = all(c[1] for c in checks)
        print(f"  VERDICT (spread sleeve adds value over T-bills): {'PASS' if verdict else 'FAIL'}")
        # stress
        risk_frac = (base["collateral"] / cap).values
        per_year = len(base) / ((END - START).days / 365.25)
        sp = stress_param(base["ror_net"].values, risk_frac, per_year, len(sets[1.0]))
        stress_rows.append({"candidate": f"capital v2 ${cap:,}", **sp})
        report["sizes"][cap] = {"tbill_cagr": tb["cagr"], "overlay_cagr": ov["cagr"], "overlay_total": ov["total"],
                                "maxdd": ov["maxdd"], "worst_month": ov["worst_month"], "trades": len(base),
                                "spread_pnl": float(base["pnl_usd"].sum()), "excess_mean": m, "excess_ci": [lo, hi],
                                "years_pos": f"{(excess_y > 0).sum()}/{len(excess_y)}", "dsr": dsr, "sr_month": sr,
                                "fill_model": {c: {str(k): float(v) for k, v in q[c].items()} for c in q.columns},
                                "checks": [[n, bool(o), d] for n, o, d in checks], "verdict": "PASS" if verdict else "FAIL",
                                "yearly_overlay": {int(k): float(v) for k, v in ov["yearly"].items()},
                                "yearly_tbill": {int(k): float(v) for k, v in tb["yearly"].items()}}
        pd.DataFrame({"overlay": account(base, cap, days, rate),
                      "tbill_only": account(None, cap, days, rate, with_spreads=False)}).to_csv(OUT / f"account_{cap}.csv")

    # guard-off historical crash trades (same structure, $10,000 and $25,000 widths)
    print("\nGUARD OFF (shock entries only) inside crash windows: what the guard protects against")
    hist = []
    for cap in (10000, 25000):
        t = cap_trades(f, by, ix, spot, cap, 1.0, entry="shock")
        for a, b in spec["stress"]["windows"]:
            w = t[(t["date"] >= a) & (t["date"] <= b)]
            for r in w.itertuples():
                hist.append({"capital": cap, "window": f"{a[:7]}..{b[:7]}", "date": r.date.date(), "ror": r.ror_net,
                             "loss_usd": r.pnl_usd, "pct_of_account": r.pnl_usd / cap})
    h = pd.DataFrame(hist)
    if len(h):
        h.to_csv(OUT / "guard_off_crash_trades.csv", index=False)
        for cap in (10000, 25000):
            x = h[h["capital"] == cap]
            print(f"  ${cap:,}: {len(x)} trades, {int((x['ror'] <= -0.9).sum())} full losses, worst {x['ror'].min():+.0%} "
                  f"(= {x['pct_of_account'].min():+.1%} of the account), sum {x['pct_of_account'].sum():+.1%}")
        report["guard_off"] = {int(c): {"n": int((h["capital"] == c).sum()),
                                        "full_losses": int(((h["capital"] == c) & (h["ror"] <= -0.9)).sum()),
                                        "worst_pct": float(h.loc[h["capital"] == c, "pct_of_account"].min())}
                               for c in (10000, 25000)}

    # return-branch candidates (hindsight best and walk-forward OOS account)
    rb = Path("data/results/return_branch")
    hb = rb / "trades" / "T10_guard_s0.15_l0.05_dte21.parquet"
    if hb.exists():
        t = pd.read_parquet(hb)
        yrs = (t["exit"].max() - t["date"].min()).days / 365.25
        sp = stress_param(t["ror_net"].values, 0.20, len(t) / yrs, len(t))
        stress_rows.append({"candidate": "return branch hindsight best T10+guard 0.15/0.05 21d, 20% risk [hindsight]", **sp})
    oo = rb / "oos_trades.csv"
    if oo.exists():
        t = pd.read_csv(oo, parse_dates=["date", "exit"])
        yrs = (t["exit"].max() - t["date"].min()).days / 365.25
        sp = stress_param(t["ror_net"].values, t["risk"].values, len(t) / yrs, len(t))
        stress_rows.append({"candidate": "return branch walk-forward OOS pick (FAILED)", **sp})
    st = pd.DataFrame(stress_rows)
    st.to_csv(OUT / "stress.csv", index=False)
    print("\nTAIL STRESS (spread sleeve only; full-loss rate drawn from Beta(0.5, n+0.5); 20,000 simulated years)")
    for r in st.to_dict("records"):
        print(f"  {r['candidate']}: mean {r['mean']:+.2%}/yr, 5% VaR {r['var5']:+.2%}, 1% CVaR {r['cvar1']:+.2%}, "
              f"P(loss year) {r['p_loss_year']:.0%}, P(< -10%) {r['p_below_-10%']:.1%}, "
              f"mean full-loss rate {r['p_full_loss_mean']:.1%}")
    report["stress"] = st.to_dict("records")
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\nsaved {OUT}/report.json")


if __name__ == "__main__":
    main()
