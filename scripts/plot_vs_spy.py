"""
Every parameter setting in one or more backtest runs, run as a book (BT-10), against buying SPY.
No API calls; reads trades.parquet from the runs and data/raw/prices/daily/SPY_adj.parquet.

Run from the repo root:
    python scripts/plot_vs_spy.py --runs data/results/backtest/oat_nostop data/results/backtest/grid_nostop
    python scripts/plot_vs_spy.py --runs ... --event-set has_news --out data/results/backtest/vs_spy_news.html
Writes data/results/backtest/vs_spy.html by default.

Book: $100k, each trade risks 2% of it (no compounding), one position per ticker, 30% sector cap.
SPY: the same $100k bought on the first trade's entry date (split-adjusted close, no dividends).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.backtest.grid import portfolio_curve, trade_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--event-set", choices=["all", "has_news"], default="all")
ap.add_argument("--min-trades", type=int, default=30, help="settings with fewer trades are not highlighted")
ap.add_argument("--out", default="data/results/backtest/vs_spy.html")
args = ap.parse_args()

CAP = 100_000
gics = pd.read_parquet("data/raw/reference/gics.parquet").set_index("ric")["sector"].to_dict()

trades = []
for r in args.runs:
    t = pd.read_parquet(Path(r) / "trades.parquet")
    t = t[(t.status == "traded") & t.ror.notna()].copy()
    tag = Path(r).name
    t["setting"] = t["label"] if len(args.runs) == 1 else tag + ": " + t["label"]
    trades.append(t)
T = pd.concat(trades, ignore_index=True)
for c in ["entry_date", "exit_date"]:
    T[c] = pd.to_datetime(T[c])
if args.event_set == "has_news":
    lab = pd.read_csv("data/results/llm_labels/labels.csv")
    news = set(lab.loc[lab.n_headlines > 0, "ric"] + "|" + lab.loc[lab.n_headlines > 0, "date"])
    T = T[T.event_id.isin(news)]

spy = pd.read_parquet("data/raw/prices/daily/SPY_adj.parquet")["TRDPRC_1"].dropna()
spy.index = pd.to_datetime(spy.index)
start, end = T.entry_date.min(), T.exit_date.max()
spy = spy[(spy.index >= start) & (spy.index <= end)]
spy_cum = spy / spy.iloc[0] - 1
spy_daily = spy.pct_change().dropna()


def stats_from_equity(cum):
    """total return, max drawdown, annualized Sharpe from a cumulative-return series (daily)"""
    eq = 1 + cum
    d = eq.diff().dropna()  # P&L in units of starting capital (no compounding in the book)
    sh = d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else np.nan
    return float(cum.iloc[-1]), float((eq / eq.cummax() - 1).min()), float(sh)


spy_stats = dict(zip(["total_return", "max_drawdown", "sharpe"], stats_from_equity(spy_cum)))

rows, curves = [], {}
for name, g in T.groupby("setting"):
    taken, eq, st = portfolio_curve(g, gics, capital=CAP)
    if eq.empty:
        continue
    cum = (eq / CAP - 1).reindex(spy_cum.index.union(eq.index)).ffill().fillna(0.0)
    cum = cum[(cum.index >= start) & (cum.index <= end)]
    # weekly correlation of book P&L with SPY (the book books P&L on exit days, so daily is too lumpy)
    wk_book = cum.resample("W-FRI").last().diff().dropna()
    wk_spy = spy_cum.add(1).resample("W-FRI").last().pct_change().dropna()
    j = wk_book.index.intersection(wk_spy.index)
    corr = float(np.corrcoef(wk_book[j], wk_spy[j])[0, 1]) if len(j) > 10 and wk_book[j].std() > 0 else np.nan
    m = trade_metrics(g)
    rows.append({"setting": name, "n": int(m["n"]), "n_taken": st["n_taken"], "win_rate": m["win_rate"],
                 "mean_ror": m["mean_ror"], "total_return": st["total_return"], "max_drawdown": st["max_drawdown"],
                 "sharpe": st["sharpe"], "corr_spy": corr})
    curves[name] = cum

S = pd.DataFrame(rows).sort_values("total_return", ascending=False)
S["beats_spy"] = S.total_return > spy_stats["total_return"]
eligible = S[S.n_taken >= args.min_trades]
best = eligible.iloc[0]["setting"] if len(eligible) else S.iloc[0]["setting"]
base_names = [s for s in S.setting if s == "baseline" or s.endswith(": baseline")]
highlight = [best] + [b for b in base_names if b != best][:1]

# per-trade: the best setting's trades vs SPY's return over each trade's holding period
bt = T[T.setting == best].copy()
spy_at = lambda d: spy.asof(d) if d >= spy.index[0] else np.nan
bt["spy_ret"] = [spy_at(x) / spy_at(e) - 1 if pd.notna(spy_at(x)) else np.nan
                 for e, x in zip(bt.entry_date, bt.exit_date)]
bt = bt.dropna(subset=["spy_ret"])
beta = float(np.polyfit(bt.spy_ret, bt.ror, 1)[0]) if len(bt) > 5 else np.nan

# quarterly: book P&L per quarter (share of capital) vs SPY's quarterly return
def quarterly(cum):
    q = (1 + cum).resample("QE").last()
    return q.diff().fillna(q.iloc[0] - 1)


qs = spy.resample("QE").last()
q_spy = qs.pct_change().fillna(qs.iloc[0] / spy.iloc[0] - 1)
quarters = [str(p) for p in q_spy.index.to_period("Q")]
q_data = {"x": quarters, "SPY": q_spy.tolist()}
for h in highlight:
    q_data[h] = quarterly(curves[h]).reindex(q_spy.index).fillna(0).tolist()


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


def ser(s, step=1):
    s = s.iloc[::step] if step > 1 else s
    return {"x": [d.strftime("%Y-%m-%d") for d in s.index], "y": s.tolist()}


data = {
    "runs": args.runs, "event_set": args.event_set, "start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d"),
    "spy_end": spy.index[-1].strftime("%Y-%m-%d"), "spy": {**spy_stats, **ser(spy_cum)},
    "settings": S.to_dict("records"), "best": best, "highlight": highlight,
    "curves": {k: ser(v, 2) for k, v in curves.items()},
    "trades": {"spy": bt.spy_ret.tolist(), "ror": bt.ror.tolist(), "event": bt.event_id.tolist(), "beta": beta},
    "quarters": q_data, "min_trades": args.min_trades,
}

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Strategy vs SPY</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--other:#c3c2b7;
--neg:#e34948;--pos:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--other:#4a4a46;--neg:#e66767;--pos:#3987e5}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--other:#4a4a46;--neg:#e66767;--pos:#3987e5}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h3{font-size:15px;margin:0 0 2px}
.sub{color:var(--ink2);margin:0 0 16px}.note{color:var(--ink2);font-size:13px;margin:4px 0 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:16px;margin-top:16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;min-width:0;margin-top:16px}
.grid .card{margin-top:0}
.chart{width:100%;height:360px}.tall{height:520px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:12px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile .k{color:var(--ink2);font-size:13px}.tile .v{font-size:24px;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--grid)}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600;cursor:pointer;user-select:none}.scroll{overflow-x:auto;max-height:520px;overflow-y:auto}
thead th{position:sticky;top:0;background:var(--surface)}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:baseline}
</style></head><body><main>
<h1>Strategy vs SPY</h1>
<p class="sub" id="meta"></p>
<div id="tiles"></div>
<div class="card"><h3>Cumulative return: every setting vs SPY</h3>
<p class="note" id="n_lines"></p><div id="c_lines" class="chart tall"></div></div>
<div class="grid">
<div class="card"><h3>Return vs worst drawdown, one dot per setting</h3>
<p class="note">Up and to the right is better: more return, smaller drawdown. Hover for the setting.</p><div id="c_scatter" class="chart"></div></div>
<div class="card"><h3>Return per quarter</h3>
<p class="note">Book P&amp;L per quarter as a share of the $100k, next to SPY's quarterly return.</p><div id="c_q" class="chart"></div></div>
</div>
<div class="card"><h3 id="t_trades"></h3>
<p class="note" id="n_trades"></p><div id="c_trades" class="chart"></div></div>
<div class="card"><h3>All settings</h3><p class="note">Click a column to sort. Book = $100k, 2% risk per trade.</p><div id="tbl" class="scroll"></div></div>
</main>
<script>
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
const CFG = {displayModeBar:false, responsive:true};
const short = s => s.replace(/(^|: )baseline$/, '$1baseline (no stop, wait ≤ 3)');
const H = D.highlight, best = D.best, S = D.settings;
const colorOf = n => n === best ? css('--s1') : (H.includes(n) ? css('--s3') : css('--other'));

document.getElementById('meta').textContent =
  `${D.start} to ${D.end} · events: ${D.event_set} · runs: ${D.runs.join(', ')} · SPY data ends ${D.spy_end}`;
const nBeat = S.filter(r => r.beats_spy).length, b = S.find(r => r.setting === best);
document.getElementById('tiles').innerHTML = '<div class="tiles">' + [
  ['SPY return', pct(D.spy.total_return)], ['SPY worst drawdown', pct(D.spy.max_drawdown)], ['SPY Sharpe', num(D.spy.sharpe)],
  ['Best setting return', pct(b.total_return)], ['Best setting drawdown', pct(b.max_drawdown)], ['Best setting Sharpe', num(b.sharpe)],
  ['Settings beating SPY', `${nBeat} of ${S.length}`]].map(([k, v]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('') + '</div>';

function draw() {
  // lines: other settings faint, highlighted on top, SPY on top of all
  const others = Object.keys(D.curves).filter(n => !H.includes(n));
  const tr = others.map(n => ({type:'scatter', mode:'lines', x:D.curves[n].x, y:D.curves[n].y, name:n, showlegend:false,
      line:{color:css('--other'), width:1}, opacity:0.55, hovertemplate:short(n)+'<br>%{x}: %{y:.1%}<extra></extra>'}));
  H.forEach(n => tr.push({type:'scatter', mode:'lines', x:D.curves[n].x, y:D.curves[n].y, name: n === best ? 'Best: ' + short(n) : short(n),
      line:{color:colorOf(n), width:2}, hovertemplate:short(n)+'<br>%{x}: %{y:.1%}<extra></extra>'}));
  tr.push({type:'scatter', mode:'lines', x:D.spy.x, y:D.spy.y, name:'SPY (buy and hold)', line:{color:css('--s2'), width:2},
      hovertemplate:'SPY<br>%{x}: %{y:.1%}<extra></extra>'});
  tr.push({type:'scatter', mode:'lines', x:[null], y:[null], name:'Other settings', line:{color:css('--other'), width:1}, hoverinfo:'skip'});
  Plotly.react('c_lines', tr, layout({yaxis:{tickformat:'.0%', zeroline:true}, hovermode:'closest'}), CFG);
  document.getElementById('n_lines').textContent = `${S.length} settings. Blue = best total return with ${D.min_trades}+ trades, green = baseline, orange = SPY. Book: $100k, 2% risked per trade, so the book is far less invested than SPY: compare drawdown and Sharpe too.`;

  Plotly.react('c_scatter', [
    {type:'scatter', mode:'markers', x:S.map(r=>r.max_drawdown), y:S.map(r=>r.total_return), customdata:S.map(r=>[short(r.setting), r.n_taken, r.sharpe]),
     marker:{size:S.map(r=>H.includes(r.setting)?12:8), color:S.map(r=>colorOf(r.setting)), line:{color:css('--surface'), width:2}},
     hovertemplate:'%{customdata[0]}<br>return %{y:.1%} · drawdown %{x:.1%}<br>Sharpe %{customdata[2]:.2f} · %{customdata[1]} trades<extra></extra>', showlegend:false},
    {type:'scatter', mode:'markers+text', x:[D.spy.max_drawdown], y:[D.spy.total_return], text:['SPY'], textposition:'top center', textfont:{color:css('--ink2')},
     marker:{size:14, symbol:'diamond', color:css('--s2'), line:{color:css('--surface'), width:2}},
     hovertemplate:'SPY<br>return %{y:.1%} · drawdown %{x:.1%}<extra></extra>', showlegend:false}],
    layout({xaxis:{title:{text:'worst drawdown'}, tickformat:'.0%'}, yaxis:{title:{text:'total return'}, tickformat:'.0%', zeroline:true}}), CFG);

  const Q = D.quarters, qn = Object.keys(Q).filter(k => k !== 'x');
  Plotly.react('c_q', qn.map(k => ({type:'bar', name: k === 'SPY' ? 'SPY' : (k === best ? 'Best: ' : '') + short(k), x:Q.x, y:Q[k],
      marker:{color: k === 'SPY' ? css('--s2') : colorOf(k)}, hovertemplate:(k === 'SPY' ? 'SPY' : short(k))+'<br>%{x}: %{y:.1%}<extra></extra>'})),
    layout({barmode:'group', yaxis:{tickformat:'.0%', zeroline:true}, xaxis:{type:'category'}}), CFG);

  const t = D.trades;
  document.getElementById('t_trades').textContent = `Best setting, each trade: its return vs SPY's return while it was open`;
  document.getElementById('n_trades').textContent = `${short(best)} · ${t.ror.length} trades · slope ${num(t.beta)} return on risk per 1.00 of SPY return. A steep slope means the losses come with market sell-offs.`;
  Plotly.react('c_trades', [{type:'scatter', mode:'markers', x:t.spy, y:t.ror, text:t.event,
      marker:{size:8, color:t.ror.map(v => v < 0 ? css('--neg') : css('--pos')), line:{color:css('--surface'), width:1}},
      hovertemplate:'%{text}<br>SPY %{x:.1%} · trade %{y:.1%}<extra></extra>'}],
    layout({xaxis:{title:{text:"SPY return while the trade was open"}, tickformat:'.0%', zeroline:true},
            yaxis:{title:{text:'trade return on risk'}, tickformat:'.0%', zeroline:true}}), CFG);
}

const cols = [['Setting', r => `<span class="sw" style="background:${colorOf(r.setting)}"></span>${short(r.setting)}`, 'setting'],
  ['Trades', r => r.n_taken, 'n_taken'], ['Win', r => pct(r.win_rate, 0), 'win_rate'], ['Mean RoR', r => pct(r.mean_ror), 'mean_ror'],
  ['Return', r => pct(r.total_return), 'total_return'], ['vs SPY', r => pct(r.total_return - D.spy.total_return), 'total_return'],
  ['Max DD', r => pct(r.max_drawdown), 'max_drawdown'], ['Sharpe', r => num(r.sharpe), 'sharpe'], ['Corr. SPY (weekly)', r => num(r.corr_spy), 'corr_spy']];
let sortKey = 'total_return', asc = false;
function tbl() {
  const rows = [...S].sort((a, b) => { const x = a[sortKey], y = b[sortKey]; return (x < y ? -1 : x > y ? 1 : 0) * (asc ? 1 : -1); });
  const spyRow = `<tr><td><span class="sw" style="background:${css('--s2')}"></span>SPY (buy and hold)</td><td>–</td><td>–</td><td>–</td><td>${pct(D.spy.total_return)}</td><td>–</td><td>${pct(D.spy.max_drawdown)}</td><td>${num(D.spy.sharpe)}</td><td>1.00</td></tr>`;
  document.getElementById('tbl').innerHTML = '<table><thead><tr>' + cols.map((c, i) => `<th data-i="${i}">${c[0]}</th>`).join('') +
    '</tr></thead><tbody>' + spyRow + rows.map(r => '<tr>' + cols.map(c => `<td>${c[1](r)}</td>`).join('') + '</tr>').join('') + '</tbody></table>';
  document.querySelectorAll('#tbl th').forEach(th => th.onclick = () => { const k = cols[+th.dataset.i][2]; asc = sortKey === k ? !asc : k === 'setting'; sortKey = k; tbl(); });
}
draw(); tbl();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { draw(); tbl(); });
</script></body></html>"""

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(HTML.replace("__DATA__", json.dumps(clean(data))), encoding="utf-8")
print(f"wrote {out}: {len(S)} settings, best = {best}, SPY {spy_stats['total_return']:.1%}")
