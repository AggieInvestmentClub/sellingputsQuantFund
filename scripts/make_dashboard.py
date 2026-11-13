"""
Results dashboard: the key charts from every backtest run on one page, plus links to the full reports.
No API calls; reads data/results/backtest/*. Run after any new run:
    python scripts/make_dashboard.py
Writes data/results/backtest/index.html (serve with: python -m http.server 8765 --bind 127.0.0.1
--directory data/results/backtest).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.backtest.grid import portfolio_curve, trade_metrics

B = Path("data/results/backtest")
CAND = "short_delta=0.15 width_pct=0.05 target_dte=35 max_wait=3"
gics = pd.read_parquet("data/raw/reference/gics.parquet").set_index("ric")["sector"].to_dict()


def summary(run, event_set="all", max_shock=None):
    s = pd.read_csv(B / run / "summary.csv")
    s = s[s["filter"].str.startswith(f"event_set={event_set} ")]
    if max_shock is not None:
        s = s[s["filter"].str.contains(f"max_shock={max_shock} ", regex=False)]
    return s.set_index("label")


def parse(labels, key):
    return labels.str.extract(rf"{key}=([\d.]+)")[0].astype(float)


real, rnd = summary("grid_nostop"), summary("grid_random")
data = {}

# 1. book equity: candidate on shocks, same on random days, no-stop baseline, SPY
def trades(run, label, shocks):
    t = pd.read_parquet(B / run / "trades.parquet")
    t = t[(t.label == label) & (t.status == "traded") & t.ror.notna()].copy()
    if shocks:
        ev = pd.read_csv("data/results/event_study/events.csv")
        ev = ev[ev.abnormal <= -0.05]
        t = t[t.event_id.isin(set(ev.ric + "|" + ev.date))]
    for c in ["entry_date", "exit_date"]:
        t[c] = pd.to_datetime(t[c])
    return t


spy = pd.read_parquet("data/raw/prices/daily/SPY_adj.parquet")["TRDPRC_1"].dropna()
spy.index = pd.to_datetime(spy.index)
curves = {}
for name, run, lab, sh in [("Candidate on shocks", "grid_nostop", CAND, True),
                           ("Candidate on random days", "grid_random", CAND, False),
                           ("Baseline on shocks (0.20 delta)", "grid_nostop", "baseline", True)]:
    _, eq, st = portfolio_curve(trades(run, lab, sh), gics)
    curves[name] = {"x": [d.strftime("%Y-%m-%d") for d in eq.index[::2]], "y": (eq / 100_000 - 1).iloc[::2].tolist(), **st}
start = min(pd.Timestamp(c["x"][0]) for c in curves.values())
s = spy[spy.index >= start]
s = s / s.iloc[0] - 1
curves["SPY buy and hold"] = {"x": [d.strftime("%Y-%m-%d") for d in s.index[::2]], "y": s.iloc[::2].tolist(),
                              "total_return": float(s.iloc[-1]), "max_drawdown": float(((1 + s) / (1 + s).cummax() - 1).min())}
data["curves"] = curves

# 2. short delta: shocks vs random (median over grid settings)
for df in (real, rnd):
    df["delta"] = parse(df.index.to_series(), "short_delta")
    df["width"] = parse(df.index.to_series(), "width_pct")
    df["dte"] = parse(df.index.to_series(), "target_dte")
    df["wait"] = parse(df.index.to_series(), "max_wait")
g_r, g_n = real.dropna(subset=["delta"]), rnd.dropna(subset=["delta"])
data["delta"] = {"x": ["0.15", "0.20", "0.25"],
                 "shock": [g_r[g_r.delta == d].mean_ror.median() for d in (0.15, 0.2, 0.25)],
                 "random": [g_n[g_n.delta == d].mean_ror.median() for d in (0.15, 0.2, 0.25)]}

# 3. heatmaps: delta x width (wait 3, mean over DTE), shocks and random
def heat(df):
    h = df[df.wait == 3].groupby(["delta", "width"]).mean_ror.mean().unstack()
    return {"x": [f"{w:.1%}" for w in h.columns], "y": [f"{d:.2f}" for d in h.index], "z": h.values.tolist()}


data["heat"] = {"shock": heat(g_r), "random": heat(g_n)}

# 4. entry day (10-day-wait run, 2x stop): later entries lose
t = pd.read_parquet(B / "oat_panel" / "trades.parquet")
t = t[(t.label == "baseline") & (t.status == "traded")]
e = t.groupby("entry_lag").ror.agg(["mean", "count"])
data["entry"] = {"x": [f"+{int(k)}" for k in e.index], "y": e["mean"].tolist(), "n": e["count"].astype(int).tolist()}

# 5. exits / stops (wait <= 3 run): each stop rule vs none
w3 = summary("oat_panel_w3")
stops = {"2× credit (old baseline)": "baseline", "No stop": "stop_rules=[]",
         "Max loss 50%": "stop_rules=['max_loss_pct']", "Below short strike": "stop_rules=['below_short_strike']",
         "Short delta trigger": "stop_rules=['short_delta']", "Below post-news low": "stop_rules=['below_post_news_low']",
         "1.5× credit": "credit_multiple=1.5", "3× credit": "credit_multiple=3.0"}
data["stops"] = [{"name": k, "mean": float(w3.at[v, "mean_ror"]), "n": int(w3.at[v, "n"]), "win": float(w3.at[v, "win_rate"])}
                 for k, v in stops.items() if v in w3.index]

# 6. baseline evolution
steps = [("Wait ≤ 10, 2× stop", summary("oat_panel"), "baseline"),
         ("Wait ≤ 3, 2× stop", summary("oat_panel_w3"), "baseline"),
         ("Wait ≤ 3, no stop", summary("oat_nostop"), "baseline"),
         ("Same, random days", rnd, "baseline")]
data["steps"] = [{"name": n, "mean": float(df.at[l, "mean_ror"]), "n": int(df.at[l, "n"])} for n, df, l in steps]

# 7. event sets (LLM filters): 0.15-delta, wait-3 grid settings
sets = {"all": "All shocks", "has_news": "Has news", "nj_gpt4o": "GPT-4o not justified",
        "nj_opus": "Opus not justified", "nj_both": "Both not justified", "nj_both_samecat": "Both, same category"}
ev = []
for k, name in sets.items():
    s_ = summary("grid_nostop", k)
    s_ = s_[s_.index.str.contains("short_delta=0.15") & s_.index.str.contains("max_wait=3")]
    ev.append({"name": name, "mean": float(s_.mean_ror.median()), "n": float(s_.n.median())})
data["sets"] = ev

# 8. return vs drawdown per setting (from the vs-SPY page)
txt = (B / "vs_spy.html").read_text(encoding="utf-8")
vs = json.loads(re.search(r"const D = (\{.*?\});\n", txt, re.S).group(1))
data["scatter"] = {"settings": [{k: r[k] for k in ["setting", "total_return", "max_drawdown", "sharpe", "n_taken"]}
                                for r in vs["settings"] if r["n_taken"] >= 30],
                   "spy": {k: vs["spy"][k] for k in ["total_return", "max_drawdown", "sharpe"]}}

# tiles
b = data["steps"]
data["tiles"] = {"cand_shock": float(real.at[CAND, "mean_ror"]), "cand_random": float(rnd.at[CAND, "mean_ror"]),
                 "cand_n": int(real.at[CAND, "n"]), "beat": int((g_r.mean_ror - g_n.reindex(g_r.index).mean_ror > 0).sum()),
                 "n_settings": int(len(g_r)), "spy": curves["SPY buy and hold"]["total_return"],
                 "cand_book": curves["Candidate on shocks"]["total_return"], "cand_dd": curves["Candidate on shocks"]["max_drawdown"]}


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    return x


HTML = Path(__file__).with_name("dashboard_template.html").read_text(encoding="utf-8")
out = B / "index.html"
out.write_text(HTML.replace("__DATA__", json.dumps(clean(data))).replace("__CAND__", CAND), encoding="utf-8")
print(f"wrote {out}")
