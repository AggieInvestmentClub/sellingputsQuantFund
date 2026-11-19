"""
Return branch (AG-41..45): can the same elements earn more ON THE ACCOUNT, out of sample?

Pre-registered in strategy/preregistered_return_branch.md (read before running; the JSON block there is the family).
Levers: more trades (-1.0% shock threshold, overlapping positions), tenor (7-14 / 14-35 / 30-45 DTE), short delta
(0.10 / 0.15 / 0.20), and risk per trade (10% / 20% with an open-risk cap). Selection is walk-forward on the ACCOUNT
(CAGR under drawdown limits, prior years only), scored against the index branch's walk-forward (AG-28c).

    py -3.12 -u scripts/return_branch.py                 # everything: trades (cached), walk-forward, checks, report
    py -3.12 -u scripts/return_branch.py --fill 0.25     # sensitivity: better fills for the picks (not scored)

Output: data/results/return_branch/ (trades/ cache, candidates.csv, walkforward_picks.csv, oos_trades.csv, report.json)
"""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import scripts.index_branch as IB  # noqa: E402
from src.backtest.overfit import deflated_sharpe_from_matrix, pbo_cscv  # noqa: E402
from strategy.strategy_logic import pick_expiry, should_take_profit  # noqa: E402

PREREG = Path("strategy/preregistered_return_branch.md")
OUT = Path("data/results/return_branch")
START, END = pd.Timestamp("2010-01-01"), pd.Timestamp("2026-11-18")
OOS_START = pd.Timestamp("2014-01-01")
THRESH = {"T15": (-0.015, -0.025), "T10": (-0.010, -0.020)}


def load_spec(path=PREREG):
    text = Path(path).read_text(encoding="utf-8")
    spec = json.loads(re.search(r"```json\s*(\{.*?\})\s*```", text, re.S).group(1))
    return spec, hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- [AG-41] entries and trades
def entry_dates_x(f, entry):
    """shock entries at a given threshold, day +1..+3 first close above the post-shock low; '+guard' = crash guard"""
    name = entry.split("+")[0]
    t1, t2 = THRESH[name]
    shock = ((f["ret1"] <= t1) | (f["ret2"] <= t2)).fillna(False)
    idx = list(f.index)
    pos = {d: i for i, d in enumerate(idx)}
    base, last_used = [], None
    for s in f.index[shock.values]:
        if last_used is not None and s <= last_used:
            continue
        k = pos[s]
        low = f["S"].iloc[k]
        for j in range(k + 1, min(k + 4, len(idx))):
            if f["S"].iloc[j] > low:
                base.append(idx[j])
                last_used = idx[min(k + 3, len(idx) - 1)]
                break
            low = min(low, f["S"].iloc[j])
    base = pd.DatetimeIndex(base)
    if entry.endswith("+guard"):
        base = base[f["guard_ok"].fillna(False).reindex(base).fillna(False).values]
    return base


def trade_x(d, by, ix, spot, short_d, long_leg, tenor, fill=1.0):
    """AG-28 trade() with the tenor and the fill as parameters; exit = dynamic take-profit else settle"""
    lo, target, hi = tenor
    g = by.get(d)
    if g is None:
        return None
    g = g[((g["expiry"] - d).dt.days.between(lo, hi)) & (g["bid"] > 0) & (g["ask"] >= g["bid"]) & g["delta"].notna()]
    if g.empty:
        return None
    exp = pick_expiry(sorted(g["expiry"].unique()), d, target, lo, hi)
    ch = g[g["expiry"] == exp]
    S = float(spot.get(d, ch["S"].median()))
    sh = ch[(ch["strike"] < S) & ch["delta"].between(-0.45, -0.03)]
    if sh.empty:
        return None
    s = sh.loc[(sh["delta"] + short_d).abs().idxmin()]
    lg = ch[ch["strike"] < s.strike]
    if lg.empty:
        return None
    mode, val = long_leg
    l = (lg.loc[(lg["strike"] - (s.strike - val)).abs().idxmin()] if mode == "abs"
         else lg.loc[(lg["delta"] + val).abs().idxmin()])
    width = float(s.strike - l.strike)
    credit_mid = float(s.mid - l.mid)
    credit = credit_mid - fill * float((s.ask - s.bid) + (l.ask - l.bid)) / 2 - 2 * IB.FEE
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
        pay = max(float(a["mid"] - b["mid"]) + fill * float((a["ask"] - a["bid"]) + (b["ask"] - b["bid"])) / 2,
                  0.0) + 2 * IB.FEE
        if should_take_profit(credit_mid, mc, (day - d).days, (exp - d).days, "dynamic", 0.50, 0.20, 0.75):
            out, cost, how = day, pay, "take-profit"
            break
    return {"date": d, "expiry": exp, "exit": out, "how": how, "S": S, "short": float(s.strike),
            "long": float(l.strike), "width": width, "credit": credit, "cost": cost,
            "ror_net": (credit - cost) / (width - credit)}


def set_key(entry, structure, tenor):
    return f"{entry}|s{structure[0]:.2f}/l{structure[1]:.2f}|dte{tenor[1]}"


def candidate_trades(f, by, ix, spot, spec, fill=1.0, cache=True, only=None):
    """every (entry, structure, tenor) set's trades, cached to parquet; sizing is applied later by the account"""
    fam = spec["family"]
    d = OUT / ("trades" if fill == 1.0 else f"trades_fill{fill}")
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    entries = {e: entry_dates_x(f, e) for e in fam["entry"]}
    for e in fam["entry"]:
        for st in fam["structure"]:
            for tn in fam["tenor"]:
                key = set_key(e, st, tn)
                if only is not None and key not in only:
                    continue
                p = d / (re.sub(r"[^A-Za-z0-9.]+", "_", key) + ".parquet")
                if cache and p.exists():
                    t = pd.read_parquet(p)
                else:
                    rows = [trade_x(x, by, ix, spot, st[0], ("delta", st[1]), tn, fill) for x in entries[e]]
                    t = pd.DataFrame([r for r in rows if r])
                    if t.empty:
                        t = pd.DataFrame(columns=["date", "expiry", "exit", "how", "S", "short", "long", "width",
                                                  "credit", "cost", "ror_net"])
                    t.to_parquet(p, index=False)
                out[key] = t
                print(f"  {key:34s} n={len(t):4d} mean={t['ror_net'].mean() if len(t) else float('nan'):+.2%}",
                      flush=True)
    return out


# ---------------------------------------------------------------- [AG-42] the account
def simulate(tr, equity0=1.0):
    """
    Book trades on one account. tr: date, exit, ror_net and per-trade k (max open), risk (fraction of equity),
    cap (max total open risk). Risk is taken on equity at entry (booked P&L only); P&L is booked on the exit day.
    A position frees its slot only for entries AFTER its exit day (same convention as index_branch.one_open).
    returns (taken trades with risk_amt / pnl_amt, booked P&L per exit day, max open risk)
    """
    if tr is None or len(tr) == 0:
        return tr.iloc[0:0].assign(risk_amt=[], pnl_amt=[]) if tr is not None else None, pd.Series(dtype=float), 0.0
    tr = tr.sort_values(["date", "exit"]).reset_index(drop=True)
    eq, open_pos, taken, booked, max_open = equity0, [], [], [], 0.0
    for i, r in tr.iterrows():
        d = r["date"]
        for p in sorted([p for p in open_pos if p["exit"] < d], key=lambda p: p["exit"]):
            eq += p["pnl"]
            booked.append((p["exit"], p["pnl"]))
            open_pos.remove(p)
        if len(open_pos) >= int(r["k"]) or sum(p["frac"] for p in open_pos) + r["risk"] > r["cap"] + 1e-9:
            continue
        risk_amt = r["risk"] * eq
        pnl = r["ror_net"] * risk_amt
        open_pos.append({"exit": r["exit"], "pnl": pnl, "frac": r["risk"]})
        max_open = max(max_open, sum(p["frac"] for p in open_pos))
        taken.append({**r.to_dict(), "risk_amt": risk_amt, "pnl_amt": pnl})
    for p in sorted(open_pos, key=lambda p: p["exit"]):
        eq += p["pnl"]
        booked.append((p["exit"], p["pnl"]))
    b = pd.Series([x[1] for x in booked], index=pd.DatetimeIndex([x[0] for x in booked]), dtype=float)
    return pd.DataFrame(taken), b.groupby(level=0).sum().sort_index(), max_open


def account_stats(booked, start, end, equity0=1.0, taken=None, max_open=np.nan):
    """CAGR, drawdown, worst month and yearly returns from booked P&L (amounts) on [start, end]"""
    years = max((end - start).days / 365.25, 1e-9)
    b = booked[(booked.index >= start) & (booked.index <= end)] if len(booked) else booked
    eq = equity0 + b.cumsum() if len(b) else pd.Series([equity0], index=[start])
    curve = pd.concat([pd.Series([equity0], index=[start - pd.Timedelta(days=1)]), eq])
    final = float(curve.iloc[-1])
    dd = float((curve / curve.cummax() - 1).min())
    # monthly / yearly returns on the equity at the start of each period
    me = curve.groupby(curve.index.to_period("M")).last()
    months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    me = me.reindex(months).ffill().fillna(equity0)
    mret = me / me.shift(1).fillna(equity0) - 1
    ye = curve.groupby(curve.index.year).last().reindex(range(start.year, end.year + 1)).ffill().fillna(equity0)
    yret = ye / ye.shift(1).fillna(equity0) - 1
    active = sorted(set(b.index.year)) if len(b) else []
    n = len(taken) if taken is not None else np.nan
    worst_trade = float(taken["ror_net"].min()) if taken is not None and len(taken) else np.nan
    max_losses = int((taken["ror_net"] <= -0.9).sum()) if taken is not None and len(taken) else 0
    cagr = (final / equity0) ** (1 / years) - 1 if final > 0 else -1.0
    return {"cagr": cagr, "total": final / equity0 - 1, "maxdd": dd, "worst_month": float(mret.min()),
            "monthly": mret, "yearly": yret, "years_active": len(active),
            "years_pos_share": float((yret.loc[active] > 0).mean()) if active else np.nan,
            "n": n, "trades_per_year": n / years if n == n else np.nan, "worst_trade": worst_trade,
            "max_losses": max_losses, "max_open": max_open, "ret_per_dd": cagr / abs(dd) if dd < 0 else np.inf}


def with_sizing(t, sizing):
    k, risk, cap = sizing
    return t.assign(k=k, risk=risk, cap=cap)


def candidates(spec, trades):
    fam = spec["family"]
    return [(key, sz) for key in trades for sz in fam["sizing"]]


def run_candidate(trades, key, sz, spec, end, start=START):
    t = trades[key]
    t = t[(t["date"] >= start) & (t["exit"] < end)]
    taken, booked, mo = simulate(with_sizing(t, spec["family"]["sizing"][sz]))
    return account_stats(booked, start, end - pd.Timedelta(days=1), taken=taken, max_open=mo), taken, booked


def eligible(st, wf):
    return st["n"] >= wf["min_trades"] and st["maxdd"] > wf["max_dd"] and st["worst_month"] > wf["worst_month"]


# ---------------------------------------------------------------- [AG-43] walk-forward on the account
def walk_forward(trades, spec):
    wf = spec["walk_forward"]
    order = {s: i for i, s in enumerate(spec["family"]["sizing"])}
    picks, stitched = [], []
    for y in range(wf["first_year"], wf["last_year"] + 1):
        y0 = pd.Timestamp(f"{y}-01-01")
        best = None
        for key, sz in candidates(spec, trades):
            st, _, _ = run_candidate(trades, key, sz, spec, y0)
            if not eligible(st, wf):
                continue
            score = (round(st["cagr"], 10), -order[sz])
            if best is None or score > best[0]:
                best = (score, key, sz, st)
        if best is None:
            picks.append({"year": y, "pick": None})
            continue
        _, key, sz, st = best
        t = trades[key]
        yr = with_sizing(t[t["date"].dt.year == y], spec["family"]["sizing"][sz]).assign(cand=f"{key}|{sz}")
        stitched.append(yr)
        picks.append({"year": y, "pick": f"{key}|{sz}", "train_cagr": st["cagr"], "train_maxdd": st["maxdd"],
                      "train_worst_month": st["worst_month"], "train_n": st["n"], "entries_in_year": len(yr)})
        print(f"  {y}: {key}|{sz}  train CAGR {st['cagr']:+.2%} dd {st['maxdd']:+.1%} n={st['n']}", flush=True)
    return pd.DataFrame(picks), (pd.concat(stitched, ignore_index=True) if stitched else pd.DataFrame())


def boot_diff(a, b, n=5000, seed=0):
    """95% interval of the mean paired yearly difference a - b, resampling calendar years"""
    d = (a - b).values
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def current_rule_oos():
    t = pd.read_csv(IB.OUT / "walkforward_oos_trades.csv", parse_dates=["date", "expiry", "exit"])
    return with_sizing(t, (1, 0.10, 0.10))


def oos(tr, end=END):
    taken, booked, mo = simulate(tr)
    return account_stats(booked, OOS_START, end, taken=taken, max_open=mo), taken, booked


def fmt(name, st):
    return (f"  {name:44s} CAGR {st['cagr']:+6.2%}  maxDD {st['maxdd']:+6.1%}  worst month {st['worst_month']:+6.1%}"
            f"  years+ {st['years_pos_share']:.0%}  trades/yr {st['trades_per_year']:.1f}  max losses {st['max_losses']}"
            f"  worst trade {st['worst_trade']:+.0%}  max open risk {st['max_open']:.0%}")


# ---------------------------------------------------------------- [AG-44] capital translation, [AG-45] report
def capital_dollars(f, by, ix, spot, pick, spec, capitals=(2500, 5000, 10000), fill=1.0):
    """the latest pick with the capital width rule; one contract per position; $ per year over 2014-2026"""
    entry, struct, tenor_s, sz = pick.split("|")
    sd, ld = [float(x) for x in re.findall(r"[0-9.]+", struct)]
    tn = next(t for t in spec["family"]["tenor"] if f"dte{t[1]}" == tenor_s)
    k, risk, cap = spec["family"]["sizing"][sz]
    dates = entry_dates_x(f, entry)
    dates = dates[dates >= OOS_START]
    years = (END - OOS_START).days / 365.25
    res = []
    for c in capitals:
        budget = c * risk
        W = int(budget // 100)
        if W < 1:
            res.append({"capital": c, "width": "none", "note": "less than one $1 spread"})
            continue
        leg = ("delta", ld) if W >= 15 else ("abs", float(W))
        rows = [trade_x(d, by, ix, spot, sd, leg, tn, fill) for d in dates]
        t = pd.DataFrame([r for r in rows if r])
        if W < 15 and len(t):
            t = t[(t["width"] - t["credit"]) * 100 <= budget * 1.15]
        taken, _, _ = simulate(t.assign(k=k, risk=risk, cap=cap)) if len(t) else (t, None, 0)
        dollars = float(((taken["credit"] - taken["cost"]) * 100).sum()) if len(taken) else 0.0
        worst = float(((taken["credit"] - taken["cost"]) * 100).min()) if len(taken) else 0.0
        res.append({"capital": c, "width": f"${W}" if W < 15 else f"{ld}d", "trades": len(taken),
                    "total_$": dollars, "per_year_$": dollars / years, "per_year_pct": dollars / years / c,
                    "worst_trade_$": worst, "max_loss_one_spread_$": float(((taken["width"] - taken["credit"]) * 100).max())
                    if len(taken) else 0.0})
    return pd.DataFrame(res)


def monthly_matrix(trades, spec):
    cols, names = [], []
    for key, sz in candidates(spec, trades):
        st, _, _ = run_candidate(trades, key, sz, spec, END + pd.Timedelta(days=1))
        cols.append(st["monthly"].values)
        names.append(f"{key}|{sz}")
    return np.column_stack(cols), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", type=float, default=1.0, help="sensitivity only; the scored run is FILL = 1")
    a = ap.parse_args()
    spec, sha = load_spec()
    print(f"pre-registration {PREREG} sha256 {sha}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    f, by, ix, spot = IB.build("walkforward")
    trades = candidate_trades(f, by, ix, spot, spec, fill=1.0)
    n_c = len(candidates(spec, trades))
    print(f"\n{n_c} candidates", flush=True)

    # walk-forward (scored)
    print("\nWALK-FORWARD (account CAGR, prior years only)", flush=True)
    picks, st_tr = walk_forward(trades, spec)
    picks.to_csv(OUT / "walkforward_picks.csv", index=False)
    new, new_taken, _ = oos(st_tr)
    new_taken.to_csv(OUT / "oos_trades.csv", index=False)
    cur, cur_taken, _ = oos(current_rule_oos())

    # best full-period candidate (reported, not scored)
    full = []
    for key, sz in candidates(spec, trades):
        st, _, _ = run_candidate(trades, key, sz, spec, END + pd.Timedelta(days=1))
        full.append({"cand": f"{key}|{sz}", "eligible": eligible(st, spec["walk_forward"]),
                     **{k: st[k] for k in ("cagr", "maxdd", "worst_month", "years_pos_share", "n", "trades_per_year",
                                           "max_losses", "worst_trade", "max_open", "ret_per_dd")}})
    full = pd.DataFrame(full).sort_values("cagr", ascending=False)
    full.to_csv(OUT / "candidates.csv", index=False)
    best_is = full[full["eligible"]].iloc[0]
    key, sz = best_is["cand"].rsplit("|", 1)
    t = trades[key]
    best_oos, _, _ = oos(with_sizing(t[t["date"] >= OOS_START], spec["family"]["sizing"][sz]))
    best_full, _, _ = run_candidate(trades, key, sz, spec, END + pd.Timedelta(days=1))
    frozen_key = set_key("T15+guard", [0.10, 0.03], [14, 21, 35])
    tf = trades[frozen_key]
    frozen, _, _ = oos(with_sizing(tf[tf["date"] >= OOS_START], (1, 0.10, 0.10)))

    # checks
    ch = spec["checks"]
    yrs = range(2014, 2027)
    ya, yb = new["yearly"].reindex(yrs).fillna(0), cur["yearly"].reindex(yrs).fillna(0)
    mdiff, lo, hi = boot_diff(ya, yb, ch["boot"])
    checks = [("1. CAGR above the current rule, yearly-difference interval above 0",
               new["cagr"] > cur["cagr"] and lo > 0,
               f"{new['cagr']:+.2%} vs {cur['cagr']:+.2%}; mean yearly diff {mdiff:+.2%} [{lo:+.2%}, {hi:+.2%}]"),
              (f"2. positive in >= {ch['years_pos']:.0%} of years with trades", new["years_pos_share"] >= ch["years_pos"],
               f"{new['years_pos_share']:.0%}"),
              (f"3. max drawdown better than {ch['max_dd']:.0%}", new["maxdd"] > ch["max_dd"], f"{new['maxdd']:+.1%}")]

    # overfitting across the family
    R, names = monthly_matrix(trades, spec)
    dsr = deflated_sharpe_from_matrix(R)
    pbo = pbo_cscv(R, n_splits=8)

    print(f"\nOUT-OF-SAMPLE ACCOUNT 2014-01 .. {END.date()} (booked P&L, bid/ask fills)")
    print(fmt("current rule (AG-28c walk-forward, sizing A)", cur))
    print(fmt("frozen index rule T15+guard 0.10/0.03 21d A", frozen) + "   [in-sample pick]")
    print(fmt("NEW walk-forward pick (this test)", new))
    print(fmt(f"best full-period: {best_is['cand']}", best_oos) + "   [hindsight]")
    print(f"  best full-period candidate 2010-2026: CAGR {best_full['cagr']:+.2%}, maxDD {best_full['maxdd']:+.1%}")
    print("\nCHECKS")
    for name, ok, det in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {det}")
    verdict = "PASS" if all(c[1] for c in checks) else "FAIL"
    print(f"  VERDICT: {verdict}")
    print(f"\nfamily: {len(names)} candidates; deflated Sharpe of the best (monthly) {dsr['dsr']:.3f} "
          f"(best {names[dsr['best']]}); PBO {pbo['pbo']:.2f} over {pbo['n_combos']} splits")
    print("\nyearly (new vs current):")
    print(pd.DataFrame({"new": ya, "current": yb}).map(lambda x: f"{x:+.2%}").T.to_string())

    latest = picks.dropna(subset=["pick"]).iloc[-1]["pick"] if picks["pick"].notna().any() else None
    cap = capital_dollars(f, by, ix, spot, latest, spec) if latest else pd.DataFrame()
    if len(cap):
        print(f"\nlatest pick {latest} with the capital width rule, one contract per position, 2014-2026 "
              f"(latest pick applied to every year: not walk-forward):")
        print(cap.to_string(index=False))

    sens = {}
    if a.fill != 1.0:
        print(f"\nSENSITIVITY FILL = {a.fill} (not scored)", flush=True)
        IB.FILL = a.fill
        need = {p.rsplit('|', 1)[0] for p in picks['pick'].dropna()}
        tr2 = candidate_trades(f, by, ix, spot, spec, fill=a.fill, only=need)
        rows = []
        for _, p in picks.dropna(subset=["pick"]).iterrows():
            key, sz = p["pick"].rsplit("|", 1)
            t2 = tr2[key]
            rows.append(with_sizing(t2[t2["date"].dt.year == p["year"]], spec["family"]["sizing"][sz]))
        new2, _, _ = oos(pd.concat(rows, ignore_index=True))
        cur_rows = []
        wp = pd.read_csv(IB.OUT / "walkforward_picks.csv")
        cache = {}
        for _, p in wp.iterrows():
            e, sname, x = [s.strip() for s in p["pick"].split("|")]
            cache.setdefault(e, IB.entry_dates(f, e))
            ds = [d for d in cache[e] if d.year == p["year"]]
            rr = [IB.trade(d, by, ix, spot, IB.STRUCTURES[sname], x) for d in ds]
            cur_rows += [r for r in rr if r]
        cur2, _, _ = oos(with_sizing(pd.DataFrame(cur_rows), (1, 0.10, 0.10)))
        IB.FILL = 1.0
        print(fmt("current rule, FILL 0.25", cur2))
        print(fmt("NEW walk-forward pick, FILL 0.25", new2))
        cap2 = capital_dollars(f, by, ix, spot, latest, spec, fill=a.fill) if latest else pd.DataFrame()
        if len(cap2):
            print(cap2.to_string(index=False))
        sens = {"fill": a.fill, "current_cagr": cur2["cagr"], "new_cagr": new2["cagr"], "new_maxdd": new2["maxdd"],
                "capital": cap2.to_dict("records") if len(cap2) else []}

    def slim(st):
        return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in st.items()
                if k not in ("monthly", "yearly")}
    rep = {"prereg_sha": sha, "n_candidates": len(names), "verdict": verdict,
           "checks": [{"name": c[0], "pass": bool(c[1]), "detail": c[2]} for c in checks],
           "current": slim(cur), "frozen_index_rule": slim(frozen), "new": slim(new),
           "best_full_period": {"cand": best_is["cand"], "full": slim(best_full), "oos_2014_2026": slim(best_oos)},
           "dsr": dsr["dsr"], "dsr_best": names[dsr["best"]], "pbo": pbo["pbo"], "latest_pick": latest,
           "capital": cap.to_dict("records") if len(cap) else [], "fill_sensitivity": sens,
           "yearly_new": {int(k): float(v) for k, v in ya.items()},
           "yearly_current": {int(k): float(v) for k, v in yb.items()}}
    name = "report.json" if a.fill == 1.0 else f"report_fill{a.fill}.json"
    (OUT / name).write_text(json.dumps(rep, indent=2, default=str))
    print(f"\nsaved {OUT / name}")


if __name__ == "__main__":
    main()
