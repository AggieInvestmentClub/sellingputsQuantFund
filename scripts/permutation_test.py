"""
[AG-61] Entry-timing permutation test of the index branch (strategy/preregistered_permutation_test.md).

Prices the capital-rule-v2 spread on EVERY tradable day once, then compares the real signal's account CAGR with
10,000 runs that sell the same spread on random days (same number of entries, one spread open at a time).

    py -3.12 -u scripts/permutation_test.py
Output: data/results/permutation_test/ (summary.json, run.log, pool_<capital>.csv, perm_<capital>.csv, chart.png)
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import scripts.index_branch as IB  # noqa: E402

PREREG = Path("strategy/preregistered_permutation_test.md")
OUT = Path("data/results/permutation_test")


def load_spec():
    text = PREREG.read_text(encoding="utf-8")
    spec = json.loads(re.search(r"```json\s*(\{.*?\})\s*```", text, re.S).group(1))
    return spec, hashlib.sha256(PREREG.read_bytes()).hexdigest()[:16]


def one_open_np(dates, exits, order):
    """one spread at a time over trades taken in date order; returns the kept positions (indices into the pool)"""
    kept, busy_until = [], np.datetime64("1900-01-01")
    for i in order:
        if dates[i] >= busy_until:
            kept.append(i)
            busy_until = exits[i]
    return kept


def cagr(pnl_sum, capital, years):
    end = capital + pnl_sum
    return (end / capital) ** (1 / years) - 1 if end > 0 else -1.0


def run_null(pool, n_draw, capital, years, n_perm, rng, mask=None):
    idx = np.flatnonzero(mask) if mask is not None else np.arange(len(pool))
    dates, exits, pnl = pool["date"].values, pool["exit"].values, pool["pnl_usd"].values
    out = np.empty(n_perm)
    ror = np.empty(n_perm)
    for k in range(n_perm):
        pick = np.sort(rng.choice(idx, size=min(n_draw, len(idx)), replace=False))
        kept = one_open_np(dates, exits, pick)
        out[k] = cagr(pnl[kept].sum(), capital, years)
        ror[k] = pool["ror_net"].values[kept].mean() if kept else 0.0
    return out, ror


def main():
    spec, sha = load_spec()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"pre-registration {PREREG} sha256 {sha}", flush=True)
    f, by, ix, spot = IB.build("capital_v2")
    days = [d for d in f.index if d in by]
    sig_dates = set(IB.entry_dates(f, "shock+guard"))
    guard = f["guard_ok"].reindex(days).fillna(False).astype(bool)
    rng = np.random.default_rng(spec["seed"])
    summary = {"sha": sha, "sizes": {}}
    for cap in spec["capitals"]:
        budget, W, struct = IB.capital_structure(cap)
        rows = []
        for d in days:
            r = IB.trade(d, by, ix, spot, struct, "tp")
            if r:
                rows.append(r)
        pool = IB.within_budget(pd.DataFrame(rows), budget).sort_values("date").reset_index(drop=True)
        pool["pnl_usd"] = pool["ror_net"] * (pool["width"] - pool["credit"]) * 100
        pool["is_signal"] = pool["date"].isin(sig_dates)
        pool["guard"] = guard.reindex(pool["date"]).values
        pool.to_csv(OUT / f"pool_{cap}.csv", index=False)
        years = (pool["exit"].max() - pool["date"].min()).days / 365.25
        sig_idx = np.flatnonzero(pool["is_signal"].values)
        kept = one_open_np(pool["date"].values, pool["exit"].values, sig_idx)
        real = cagr(pool["pnl_usd"].values[kept].sum(), cap, years)
        real_ror = pool["ror_net"].values[kept].mean()
        res = {"real_cagr": real, "real_ror": float(real_ror), "n_signal": int(len(sig_idx)), "n_kept": len(kept),
               "pool_days": len(pool), "guard_days": int(pool["guard"].sum()), "years": years}
        perms = {}
        for name, mask in (("all_days", None), ("guard_days", pool["guard"].values)):
            dist, ror = run_null(pool, len(sig_idx), cap, years, spec["n_perm"], rng, mask)
            k = int((dist >= real).sum())
            p = (1 + k) / (1 + len(dist))
            perms[name] = dist
            res[name] = {"k": k, "p": p, "median": float(np.median(dist)), "p5": float(np.percentile(dist, 5)),
                         "p95": float(np.percentile(dist, 95)), "mean_ror_median": float(np.median(ror))}
        pd.DataFrame(perms).to_csv(OUT / f"perm_{cap}.csv", index=False)
        verdict = res["all_days"]["p"] < spec["alpha"]
        res["verdict"] = ("PASS" if verdict else "FAIL") if cap == spec["primary"] else "reported"
        summary["sizes"][cap] = res
        print(f"\n=== ${cap:,} (width ${W} or 0.03d): {len(pool)} tradable days, {len(sig_idx)} signal entries, "
              f"{len(kept)} taken | real CAGR {real:+.3%}, mean ror {real_ror:+.2%}")
        for name in ("all_days", "guard_days"):
            r = res[name]
            print(f"  null {name:10s}: {r['k']:5d}/{spec['n_perm']} matched or beat it -> p = {r['p']:.4f} | "
                  f"permuted median {r['median']:+.3%} [5%-95%: {r['p5']:+.3%}, {r['p95']:+.3%}]")
        if cap == spec["primary"]:
            print(f"  VERDICT (null all_days, alpha {spec['alpha']}): {res['verdict']}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    chart(spec, summary)
    print(f"\nsaved {OUT}")


def chart(spec, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    caps = spec["capitals"]
    fig, axes = plt.subplots(len(caps), 2, figsize=(11, 3.1 * len(caps)), dpi=150)
    for row, cap in enumerate(caps):
        d = pd.read_csv(OUT / f"perm_{cap}.csv")
        s = summary["sizes"][cap]
        for col, name in enumerate(("all_days", "guard_days")):
            ax = axes[row, col]
            ax.hist(d[name] * 100, bins=40, color="#2a78d6", alpha=0.55, edgecolor="white")
            ax.axvline(s["real_cagr"] * 100, color="#0b0b0b", lw=2, label="real signal")
            ax.set_title(f"${cap:,} · random days from {'all days' if name == 'all_days' else 'guard-on days'}: "
                         f"{s[name]['k']}/{spec['n_perm']} ≥ real (p={s[name]['p']:.3f})", fontsize=8.5, loc="left")
            ax.set_xlabel("Spread-only CAGR (%)", fontsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=7.5)
            ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Index branch entry-timing permutation test · 10,000 random-day runs per panel · 2010-2026", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "chart.png")


if __name__ == "__main__":
    main()
