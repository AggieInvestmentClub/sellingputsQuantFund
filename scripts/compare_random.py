"""
Shock entries vs random entries: the same trade settings run on real shocks and on random no-shock
dates in the same stocks (scripts/make_random_events.py). If the shock trades do no better than the
random ones, the edge is selling put premium in general, not the shock.

Run from the repo root:
    python scripts/compare_random.py --real data/results/backtest/grid_nostop --random data/results/backtest/grid_random
Writes <random>/vs_real.html and prints the table.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.backtest.grid import trade_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--real", default="data/results/backtest/grid_nostop")
ap.add_argument("--random", default="data/results/backtest/grid_random")
ap.add_argument("--min-mcap", type=float, default=5e9)
args = ap.parse_args()


def load(run, shock):
    t = pd.read_parquet(Path(run) / "trades.parquet")
    t = t[(t.status == "traded") & t.ror.notna()].copy()
    ev = pd.read_csv("data/results/event_study/events.csv") if shock else \
        pd.read_csv("data/results/event_study/random_events.csv")
    if shock:  # same event filter as the "all" set: shock <= -5%
        ev = ev[ev.abnormal <= -0.05]
        t = t[t.event_id.isin(set(ev.ric + "|" + ev.date))]
    return t


real, rnd = load(args.real, True), load(args.random, False)
# market cap floor, same for both: last market cap before the date
caps = {}
for t in (real, rnd):
    for ric in t.ric.unique():
        if ric in caps:
            continue
        p = Path("data/raw/market_cap") / (ric.replace(".", "_").replace("^", "~") + ".parquet")
        caps[ric] = pd.read_parquet(p)["market_cap"].dropna() if p.exists() else None


def cap_ok(t):
    keep = []
    for ric, d in zip(t.ric, pd.to_datetime(t.date)):
        m = caps.get(ric)
        m = None if m is None else m[pd.to_datetime(m.index) < d]
        keep.append(m is not None and len(m) > 0 and float(m.iloc[-1]) >= args.min_mcap)
    return t[np.array(keep, bool)]


real, rnd = cap_ok(real), cap_ok(rnd)


def welch(a, b):
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


rows = []
for lab in sorted(set(real.label) & set(rnd.label)):
    a, b = real[real.label == lab], rnd[rnd.label == lab]
    ma, mb = trade_metrics(a), trade_metrics(b)
    rows.append({"setting": lab, "n_real": int(ma["n"]), "n_random": int(mb["n"]),
                 "win_real": ma["win_rate"], "win_random": mb["win_rate"],
                 "ror_real": ma["mean_ror"], "ror_random": mb["mean_ror"],
                 "diff": ma["mean_ror"] - mb["mean_ror"], "t_diff": welch(a.ror, b.ror),
                 "worst5_real": ma["worst5_ror"], "worst5_random": mb["worst5_ror"],
                 "delta": (lab.split("short_delta=")[1].split()[0] if "short_delta=" in lab else "0.2")})
R = pd.DataFrame(rows).sort_values("ror_real", ascending=False)

cands = [s for s in R.setting if "short_delta=0.15" in s and "max_wait=3" in s
         and any(f"width_pct={w}" in s for w in ["0.05", "0.075", "0.1"])
         and any(f"target_dte={d}" in s for d in ["21", "28", "35"])]
exits = {name: t[t.label == "baseline"].exit_reason.value_counts(normalize=True).round(4).to_dict()
         for name, t in [("real", real), ("random", rnd)]}

pd.set_option("display.width", 220)
show = ["setting", "n_real", "n_random", "win_real", "win_random", "ror_real", "ror_random", "diff", "t_diff"]
print(f"real shock trades: {len(real)}, random-date trades: {len(rnd)} (all settings)")
print("\nsettings where shocks beat random:", int((R["diff"] > 0).sum()), "of", len(R),
      "| median diff", round(R["diff"].median(), 4))
print(R.groupby("delta")[["ror_real", "ror_random", "diff"]].median().round(4).to_string())
print("\ncandidate settings (0.15 delta, 5-10% width, 21-35 DTE, wait 3):")
print(R[R.setting.isin(cands)][show].round(3).to_string(index=False))
print("\nbaseline:"); print(R[R.setting == "baseline"][show].round(3).to_string(index=False))
print("\nbaseline exits:", exits)


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


data = {"rows": R.to_dict("records"), "cands": cands, "exits": exits,
        "n_real": int(len(real)), "n_random": int(len(rnd)), "real": args.real, "random": args.random}

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shock vs Random Entries</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--neg:#e34948;--pos:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--neg:#e66767;--pos:#3987e5}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--neg:#e66767;--pos:#3987e5}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h3{font-size:15px;margin:0 0 2px}
.sub{color:var(--ink2);margin:0 0 16px}.note{color:var(--ink2);font-size:13px;margin:4px 0 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:16px;margin-top:16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;min-width:0;margin-top:16px}.grid .card{margin-top:0}
.chart{width:100%;height:420px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:12px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile .k{color:var(--ink2);font-size:13px}.tile .v{font-size:24px;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--grid)}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}.scroll{overflow-x:auto;max-height:520px;overflow-y:auto}thead th{position:sticky;top:0;background:var(--surface)}
</style></head><body><main>
<h1>Shock entries vs random entries</h1>
<p class="sub" id="meta"></p>
<div id="tiles"></div>
<div class="grid">
<div class="card"><h3>Each setting: mean return on risk, shocks vs random days</h3>
<p class="note">Above the diagonal = the shock adds return beyond selling puts on any day. Color = short delta.</p><div id="c_sc" class="chart"></div></div>
<div class="card"><h3>Candidate settings</h3>
<p class="note">0.15 delta, 5-10% width, 21-35 DTE, wait up to 3 days, no stop-loss.</p><div id="c_bar" class="chart"></div></div>
</div>
<div class="card"><h3>All settings</h3><p class="note">t = Welch t-statistic of the difference in mean return on risk (trades treated as independent).</p><div id="tbl" class="scroll"></div></div>
</main><script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const pct = (x, d=1) => x==null ? "–" : (x*100).toFixed(d) + "%";
const num = (x, d=2) => x==null ? "–" : x.toFixed(d);
function layout(extra={}) {
  const ax = {gridcolor: css('--grid'), linecolor: css('--axis'), zerolinecolor: css('--axis'), tickfont:{color: css('--muted')},
              title:{font:{color: css('--ink2'), size:13}}, automargin:true};
  return Object.assign({paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{family:'system-ui,-apple-system,"Segoe UI",sans-serif', color: css('--ink'), size:12},
    margin:{l:10,r:10,t:10,b:10}, xaxis:Object.assign({}, ax, extra.xaxis||{}), yaxis:Object.assign({}, ax, extra.yaxis||{}),
    legend:{orientation:'h', y:1.1, x:0, font:{color: css('--ink2')}}, hoverlabel:{bgcolor: css('--surface'), bordercolor: css('--axis'), font:{color: css('--ink')}},
    bargap:0.25}, Object.fromEntries(Object.entries(extra).filter(([k])=>!['xaxis','yaxis'].includes(k))));
}
const CFG = {displayModeBar:false, responsive:true}, R = D.rows;
const short = s => s.replace('short_delta=', 'Δ ').replace(' width_pct=', ' / w ').replace(' target_dte=', ' / ').replace(' max_wait=', 'd / wait ');
document.getElementById('meta').textContent = `${D.n_real.toLocaleString()} shock trades vs ${D.n_random.toLocaleString()} random-date trades, all settings · ${D.real} vs ${D.random}`;
const beat = R.filter(r => r.diff > 0).length, med = R.map(r => r.diff).sort((a, b) => a - b)[Math.floor(R.length / 2)];
const C = R.filter(r => D.cands.includes(r.setting));
const cMed = C.map(r => r.diff).sort((a, b) => a - b)[Math.floor(C.length / 2)];
document.getElementById('tiles').innerHTML = '<div class="tiles">' + [
  ['Settings where shocks beat random', `${beat} of ${R.length}`], ['Median difference, all settings', pct(med)],
  ['Median difference, candidates', pct(cMed)], ['Shock exits: closed between strikes', pct((D.exits.real || {})['closed at expiry (between strikes)'] || 0)]]
  .map(([k, v]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('') + '</div>';
function draw() {
  const cols = {'0.15': css('--s1'), '0.2': css('--s2'), '0.25': css('--s3')};
  const all = R.flatMap(r => [r.ror_real, r.ror_random]).filter(v => v != null), lo = Math.min(...all), hi = Math.max(...all);
  const tr = Object.keys(cols).map(d => { const g = R.filter(r => r.delta === d); return {type:'scatter', mode:'markers', name:'short delta ' + d,
     x:g.map(r => r.ror_random), y:g.map(r => r.ror_real), customdata:g.map(r => [short(r.setting), r.n_real, r.n_random, r.t_diff]),
     marker:{size:9, color:cols[d], line:{color:css('--surface'), width:2}},
     hovertemplate:'%{customdata[0]}<br>shocks %{y:.1%} (n %{customdata[1]})<br>random %{x:.1%} (n %{customdata[2]})<br>t %{customdata[3]:.2f}<extra></extra>'}; });
  tr.push({type:'scatter', mode:'lines', x:[lo, hi], y:[lo, hi], name:'equal', line:{color:css('--axis'), width:1, dash:'dot'}, hoverinfo:'skip'});
  Plotly.react('c_sc', tr, layout({xaxis:{title:{text:'random days: mean return on risk'}, tickformat:'.0%', zeroline:true},
    yaxis:{title:{text:'shocks: mean return on risk'}, tickformat:'.0%', zeroline:true}}), CFG);
  Plotly.react('c_bar', [
    {type:'bar', name:'Shocks', x:C.map(r => short(r.setting)), y:C.map(r => r.ror_real), marker:{color:css('--s1')}, customdata:C.map(r => r.n_real),
     hovertemplate:'%{x}<br>shocks %{y:.1%} · n %{customdata}<extra></extra>'},
    {type:'bar', name:'Random days', x:C.map(r => short(r.setting)), y:C.map(r => r.ror_random), marker:{color:css('--s2')}, customdata:C.map(r => r.n_random),
     hovertemplate:'%{x}<br>random %{y:.1%} · n %{customdata}<extra></extra>'}],
    layout({barmode:'group', yaxis:{tickformat:'.0%', zeroline:true}, xaxis:{type:'category', tickangle:-35}}), CFG);
}
const cols = [['Setting', r => short(r.setting)], ['Trades (shock)', r => r.n_real], ['Trades (random)', r => r.n_random],
  ['Win (shock)', r => pct(r.win_real, 0)], ['Win (random)', r => pct(r.win_random, 0)], ['Mean RoR (shock)', r => pct(r.ror_real)],
  ['Mean RoR (random)', r => pct(r.ror_random)], ['Difference', r => pct(r.diff)], ['t', r => num(r.t_diff)]];
document.getElementById('tbl').innerHTML = '<table><thead><tr>' + cols.map(c => `<th>${c[0]}</th>`).join('') + '</tr></thead><tbody>' +
  R.map(r => '<tr>' + cols.map(c => `<td>${c[1](r)}</td>`).join('') + '</tr>').join('') + '</tbody></table>';
draw(); matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
</script></body></html>"""

out = Path(args.random) / "vs_real.html"
out.write_text(HTML.replace("__DATA__", json.dumps(clean(data))), encoding="utf-8")
print(f"\nwrote {out}")
