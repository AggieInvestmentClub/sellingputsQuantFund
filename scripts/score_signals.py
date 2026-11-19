"""
Signal scoreboard (AG-23): which shock filters improve a bull put spread priced with REAL option closes?

One spread per shock from the option panel (data/processed/options/event_puts.parquet, EVT-12):
  entry day +1 close, expiry 14-35 DTE closest to 21, short put closest to delta -0.15 below the post-news low,
  long put ~5% of the price lower, held to expiry, ror = P&L / max loss. No bid/ask (closes only) and no
  take-profit, so the LEVEL is rough; the point is the DIFFERENCE between filters on the same trades.
Each filter is scored on the in-sample period and, once, on the hold-out (default: shocks from 2026-06-01).

    py -3.12 -u scripts/score_signals.py                     # writes data/results/agent/signals.csv + prints the table
    py -3.12 -u scripts/score_signals.py --holdout 2026-06-01

Filters scored: agent TAKE, has news / no news, IV/RV and credit/width terciles, drop size, and the
fair-value gate when data/raw/fundamentals exists (scripts/download_fundamentals.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.agent.evidence import load_close
from src.agent.valuation import gate, valuation_context

RAW = Path("data/raw")


def build_trades():
    p = pd.read_parquet("data/processed/options/event_puts.parquet",
                        columns=["event_id", "ric", "date", "expiry", "strike", "close", "volume", "S", "iv", "delta"])
    p["dte"] = (p["expiry"] - p["date"]).dt.days
    ev = pd.read_csv("data/results/event_study/events.csv", parse_dates=["date"])
    ev["event_id"] = ev["ric"] + "|" + ev["date"].dt.strftime("%Y-%m-%d")
    ev = ev.set_index("event_id")
    rows = []
    for eid, g in p.groupby("event_id", sort=False):
        if eid not in ev.index:
            continue
        e = ev.loc[eid]
        px = load_close(RAW, e.ric)
        if px is None or e.date not in px.index:
            continue
        i = px.index.get_loc(e.date)
        if i + 1 >= len(px) or i < 21:
            continue
        d1 = px.index[i + 1]
        low = float(px["close"].iloc[i:i + 2].min())
        rv = float(px["close"].pct_change().iloc[i - 20:i].std() * np.sqrt(252))
        day = g[(g["date"] == d1) & g["dte"].between(14, 35) & g["iv"].notna()]
        if day.empty:
            continue
        exp = day.loc[(day["dte"] - 21).abs().idxmin(), "expiry"]
        ch = day[day["expiry"] == exp]
        atm = g[(g["date"] == d1) & g["dte"].between(14, 60) & g["delta"].abs().between(.35, .65)]["iv"].median()
        sh = ch[(ch["strike"] < low) & ch["delta"].between(-.30, -.05) & (ch["volume"] > 0)]
        if sh.empty:
            continue
        s = sh.loc[(sh["delta"] + 0.15).abs().idxmin()]
        lg = ch[ch["strike"] < s.strike]
        if lg.empty:
            continue
        l = lg.loc[(lg["strike"] - (s.strike - 0.05 * float(s.S))).abs().idxmin()]
        credit, width = float(s.close - l.close), float(s.strike - l.strike)
        if not (0 < credit < width):
            continue
        pe = px.loc[:exp]
        if pe.index[-1] < exp - pd.Timedelta(days=4):
            continue                                   # expiry not in the price data yet
        loss = min(width, max(0.0, s.strike - float(pe["close"].iloc[-1])))
        rows.append({"event_id": eid, "ric": e.ric, "date": e.date, "abnormal": float(e.abnormal),
                     "ror": (credit - loss) / (width - credit), "credit_pct": credit / width,
                     "ivrv": atm / rv if rv > 0 else np.nan})
    return pd.DataFrame(rows)


def boot_diff(a, b, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan, np.nan
    d = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)])
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="2026-06-01", help="shocks on/after this date are the hold-out")
    ap.add_argument("--reuse-trades", action="store_true", help="reuse data/results/agent/signal_trades.csv (skip pricing)")
    a = ap.parse_args()
    st = Path("data/results/agent/signal_trades.csv")
    if a.reuse_trades and st.exists():
        t = pd.read_csv(st, parse_dates=["date"])[["event_id", "ric", "date", "abnormal", "ror", "credit_pct", "ivrv"]]
    else:
        t = build_trades()
    ag = Path("data/results/agent/agent_results.csv")
    if ag.exists():
        g = pd.read_csv(ag)[["event_id", "agent_take", "has_news"]]
        t = t.merge(g, on="event_id", how="left")
    for c in ("agent_take", "has_news"):
        t[c] = t[c].eq(True) if c in t else False
    if (RAW / "fundamentals").exists():
        gics = pd.read_parquet(RAW / "reference" / "gics.parquet")
        vs = [valuation_context(RAW, r.ric, r.date, gics) for r in t.itertuples()]
        t["fv_pass"] = [gate(v)[0] if v else np.nan for v in vs]
        t["fv_has"] = [v is not None and np.isfinite(v["fwd_pe"]) for v in vs]
    ins = t["date"] < pd.Timestamp(a.holdout)
    cut = {c: t.loc[ins, c].quantile([1 / 3, 2 / 3]).values for c in ("credit_pct", "ivrv", "abnormal")}
    filters = {
        "agent TAKE": t["agent_take"], "has news": t["has_news"], "no news": ~t["has_news"],
        "credit/width top third": t["credit_pct"] >= cut["credit_pct"][1],
        "credit/width bottom third": t["credit_pct"] < cut["credit_pct"][0],
        "IV/RV top third": t["ivrv"] >= cut["ivrv"][1],
        "biggest drops (bottom third)": t["abnormal"] <= cut["abnormal"][0],
        "agent TAKE + credit top third": t["agent_take"] & (t["credit_pct"] >= cut["credit_pct"][1]),
    }
    if "fv_pass" in t:
        filters["fair-value gate pass"] = (t["fv_pass"] == True) & t["fv_has"]
        filters["fair-value gate fail"] = (t["fv_pass"] == False) & t["fv_has"]
    rows = []
    for name, m in filters.items():
        m = m.fillna(False).astype(bool)
        for part, sel in (("in-sample", ins), ("hold-out", ~ins)):
            x, rest = t.loc[sel & m, "ror"].values, t.loc[sel & ~m, "ror"].values
            d, lo, hi = boot_diff(x, rest)
            rows.append({"filter": name, "period": part, "n": len(x), "mean_ror": x.mean() if len(x) else np.nan,
                         "win": (x > 0).mean() if len(x) else np.nan, "vs_rest": d, "ci_low": lo, "ci_high": hi})
    out = pd.DataFrame(rows)
    Path("data/results/agent").mkdir(parents=True, exist_ok=True)
    out.to_csv("data/results/agent/signals.csv", index=False)
    t.to_csv("data/results/agent/signal_trades.csv", index=False)
    print(f"{len(t)} spreads ({ins.sum()} in-sample, {(~ins).sum()} hold-out from {a.holdout}); "
          f"all: mean ror {t['ror'].mean():+.1%}, win {(t['ror'] > 0).mean():.0%}")
    print(f"tercile cutoffs (in-sample): credit/width {np.round(cut['credit_pct'], 3)}, IV/RV {np.round(cut['ivrv'], 2)}")
    pd.set_option("display.width", 200)
    fmt = out.copy()
    for c in ("mean_ror", "win", "vs_rest", "ci_low", "ci_high"):
        fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}" if c != "win" else f"{v:.0%}")
    print(fmt.to_string(index=False))
    print("\nA filter helps only if vs_rest > 0 with ci_low > 0 in-sample AND the hold-out points the same way.")


if __name__ == "__main__":
    main()
