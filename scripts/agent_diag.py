"""
Agent diagnostics.

    py -3.12 -u scripts/agent_diag.py                 # check the LLM API the agent uses (.env AGENT_* / LLM_*);
                                                      # prints what the server returns, never the key
    py -3.12 -u scripts/agent_diag.py --ablation      # [AG-53] pre-registered offline ablation on post-cutoff shocks
                                                      # (strategy/preregistered_agent_ablation.md), no LLM calls
    py -3.12 -u scripts/agent_diag.py --macro         # [AG-54] macro classifier on SPY shock days (descriptive)
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

PREREG = Path("strategy/preregistered_agent_ablation.md")
OUT = Path("data/results/agent_diag")


def check_api():
    import requests
    from scripts.run_agent import model_settings
    key, base, model = model_settings()
    print(f"base {base} | model {model} | key set: {bool(key)}")
    url = base.rstrip("/") + "/chat/completions"
    msgs = [{"role": "user", "content": 'Return JSON: {"ok": true}'}]
    for label, extra in (("json mode", {"response_format": {"type": "json_object"}}), ("plain", {}),
                         ("stream off", {"stream": False})):
        payload = {"model": model, "messages": msgs, "temperature": 0, "max_tokens": 50, **extra}
        try:
            r = requests.post(url, json=payload, headers={"Authorization": f"Bearer {key}"}, timeout=60)
            print(f"\n[{label}] HTTP {r.status_code} | content-type {r.headers.get('content-type')} | {len(r.text)} chars")
            print(r.text[:600])
        except Exception as e:                                  # noqa: BLE001
            print(f"\n[{label}] {type(e).__name__}: {e}")


# ---------------------------------------------------------------- [AG-53] ablation
def load_spec(path=PREREG):
    text = Path(path).read_text(encoding="utf-8")
    spec = json.loads(re.search(r"```json\s*(\{.*?\})\s*```", text, re.S).group(1))
    return spec, hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def ablation_data(trades="data/results/agent/signal_trades.csv", results="data/results/agent/agent_results.csv"):
    t = pd.read_csv(trades, parse_dates=["date"])
    a = pd.read_csv(results)[["event_id", "confidence", "post_cutoff", "status"]]
    d = t.merge(a, on="event_id", how="left")
    d = d[(d["post_cutoff"].astype(str) == "True") & (d["status"] == "ok")].copy()
    for c in ("agent_take", "fv_pass", "fv_has"):
        d[c] = d[c].astype(str).str.lower().eq("true")
    d["week"] = d["date"].dt.strftime("%G-%V")
    return d.reset_index(drop=True)


def cluster_boot(x, weeks, n=10000, seed=0):
    """mean and week-clustered percentile 95% interval"""
    x, weeks = np.asarray(x, float), np.asarray(weeks)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    u = np.unique(weeks)
    groups = [x[weeks == w] for w in u]
    sums, cnts = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n, len(u)))
    means = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cluster_boot_diff(xf, wf, xc, wc, n=10000, seed=0):
    """difference of means (filter - complement), weeks resampled jointly"""
    weeks = np.unique(np.concatenate([wf, wc]))
    sf = np.array([xf[wf == w].sum() for w in weeks]); nf = np.array([(wf == w).sum() for w in weeks])
    sc = np.array([xc[wc == w].sum() for w in weeks]); nc = np.array([(wc == w).sum() for w in weeks])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(weeks), size=(n, len(weeks)))
    mf = sf[idx].sum(1) / np.maximum(nf[idx].sum(1), 1)
    mc = sc[idx].sum(1) / np.maximum(nc[idx].sum(1), 1)
    ok = (nf[idx].sum(1) > 0) & (nc[idx].sum(1) > 0)
    dd = (mf - mc)[ok]
    if len(xf) == 0 or len(xc) == 0 or len(dd) == 0:
        return np.nan, np.nan, np.nan
    return float(xf.mean() - xc.mean()), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))


def masks(d):
    take, fv, fvh = d["agent_take"], d["fv_pass"], d["fv_has"]
    hi = d["confidence"].astype(float) >= 0.7
    return {"H1": (take & fv & fvh, take & ~fv & fvh), "H2": (take & hi, ~(take & hi)),
            "H3": (take & fv & fvh, ~(take & fv & fvh))}


def run_ablation(d, spec, sha):
    from src.backtest.overfit import deflated_sharpe
    from scipy.stats import kurtosis, skew
    ho = d["date"] >= pd.Timestamp(spec["holdout_from"])
    b = spec["bootstrap"]
    rows, sharpes = [], {}
    for h, (f, c) in masks(d).items():
        x = d.loc[f, "ror"].astype(float).values
        sharpes[h] = (x.mean() / x.std(ddof=1)) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan
    var = float(np.nanvar(list(sharpes.values()), ddof=1)) if np.isfinite(list(sharpes.values())).sum() > 1 else 0.0
    for h, (f, c) in masks(d).items():
        r = {"hypothesis": h, "filter": spec["hypotheses"][h]["filter"], "complement": spec["hypotheses"][h]["complement"]}
        for part, m in (("ins", ~ho), ("ho", ho)):
            ff, cc = f & m, c & m
            xf, xc = d.loc[ff, "ror"].astype(float).values, d.loc[cc, "ror"].astype(float).values
            wf, wc = d.loc[ff, "week"].values, d.loc[cc, "week"].values
            mean, lo, hi = cluster_boot(xf, wf, b["n"], b["seed"])
            cm, clo, chi = cluster_boot(xc, wc, b["n"], b["seed"])
            dm, dlo, dhi = cluster_boot_diff(xf, wf, xc, wc, b["n"], b["seed"])
            r.update({f"{part}_n": int(len(xf)), f"{part}_mean": mean, f"{part}_lo": lo, f"{part}_hi": hi,
                      f"{part}_comp_n": int(len(xc)), f"{part}_comp_mean": cm,
                      f"{part}_diff": dm, f"{part}_diff_lo": dlo, f"{part}_diff_hi": dhi})
        x = d.loc[f, "ror"].astype(float).values
        r["dsr"] = (deflated_sharpe(sharpes[h], len(x), spec["dsr"]["n_trials"], var, float(skew(x)),
                                    float(kurtosis(x, fisher=False))) if len(x) > 2 and np.isfinite(sharpes[h]) else np.nan)
        p = spec["pass"]
        r["pass"] = bool(r["ho_n"] >= p["min_n_holdout"] and np.isfinite(r["ho_lo"]) and r["ho_lo"] > p["ci_lo_gt"]
                         and np.isfinite(r["ho_comp_mean"]) and r["ho_mean"] > r["ho_comp_mean"])
        rows.append(r)
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "ablation.csv", index=False)
    fmt = lambda m, lo, hi: "n/a" if not np.isfinite(m) else f"{m:+.2%} [{lo:+.2%}, {hi:+.2%}]"   # noqa: E731
    lines = [f"AGENT ABLATION (pre-reg {PREREG} sha256 {sha}) - post-cutoff shocks only, cached ag-v1 verdicts",
             f"universe {len(d)} priced trades ({(~ho).sum()} in-sample, {ho.sum()} hold-out from {spec['holdout_from']}); "
             f"TAKEs {int(d['agent_take'].sum())}, with fundamentals {int(d['fv_has'].sum())}", ""]
    for r in rows:
        lines += [f"{r['hypothesis']}: {r['filter']}  vs  {r['complement']}",
                  f"  in-sample: n={r['ins_n']:3d} {fmt(r['ins_mean'], r['ins_lo'], r['ins_hi'])}  vs complement "
                  f"n={r['ins_comp_n']} {r['ins_comp_mean']:+.2%}; diff {fmt(r['ins_diff'], r['ins_diff_lo'], r['ins_diff_hi'])}",
                  f"  hold-out:  n={r['ho_n']:3d} {fmt(r['ho_mean'], r['ho_lo'], r['ho_hi'])}  vs complement "
                  f"n={r['ho_comp_n']} {r['ho_comp_mean']:+.2%}; diff {fmt(r['ho_diff'], r['ho_diff_lo'], r['ho_diff_hi'])}",
                  f"  deflated Sharpe (10 trials) {r['dsr']:.3f}   VERDICT: {'PASS' if r['pass'] else 'FAIL'}", ""]
    txt = "\n".join(lines)
    (OUT / "ablation.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return res


def run_macro(start="2024-04-01", end="2026-10-28"):
    """[AG-54] descriptive check of the macro classifier on SPY shock days (not a trading rule)"""
    from src.agent import macro as M
    px = M.load_closes()
    iv = None
    fp = Path("data/results/index_branch/features_capital.csv")
    if fp.exists():
        iv = pd.read_csv(fp, index_col=0, parse_dates=True)[["iv9", "iv30"]]
    days = M.shock_days(px, start, end)
    ev = M.evaluate(px, days, iv)
    tr = Path("data/results/index_branch/capital_v2_10000_trades.csv")
    if tr.exists():
        t = pd.read_csv(tr, parse_dates=["date"])
        t = t[t["date"] >= start]
        te = M.evaluate(px, list(t["date"]), iv).assign(ror=t["ror_net"].values)
    else:
        te = pd.DataFrame()
    OUT.mkdir(parents=True, exist_ok=True)
    ev.to_csv(OUT / "macro_shock_days.csv", index=False)
    lines = [f"MACRO CLASSIFIER (AG-54, prototype, descriptive): {len(ev)} SPY shock days {start}..{end}"]
    for lab, g in ev.groupby("label"):
        lines.append(f"  {lab:28s} n={len(g):3d}  SPY next 21d mean {g['fwd_ret'].mean():+.2%}  "
                     f"worst-close mean {g['fwd_worst'].mean():+.2%}  worst {g['fwd_worst'].min():+.2%}")
    if len(te):
        te.to_csv(OUT / "macro_index_trades.csv", index=False)
        lines.append(f"\nIndex-branch entries ($10k, capital v2) since {start}: {len(te)}")
        for r in te.itertuples():
            lines.append(f"  {r.date:%Y-%m-%d}  {r.label:28s} ror {r.ror:+.2%}  ({r.flags or 'no flags'})")
    txt = "\n".join(lines)
    (OUT / "macro_eval.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--macro", action="store_true", help="[AG-54] descriptive macro classifier check")
    a = ap.parse_args()
    if a.macro:
        run_macro()
    elif a.ablation:
        spec, sha = load_spec()
        print(f"pre-registration {PREREG} sha256 {sha}")
        run_ablation(ablation_data(), spec, sha)
    else:
        check_api()


if __name__ == "__main__":
    main()
