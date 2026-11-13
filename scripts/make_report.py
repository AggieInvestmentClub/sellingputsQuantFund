"""
Charts for the LLM labels and one backtest run, as a single HTML page (open it in a browser).
No API calls; reads files already on disk.

Run from the repo root:
    python scripts/make_report.py                                   (LLM labels + data/results/backtest/oat_proxy)
    python scripts/make_report.py --run data/results/backtest/oat_panel
Writes <run>/report.html
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
ap.add_argument("--run", default="data/results/backtest/oat_proxy")
args = ap.parse_args()
run = Path(args.run)


def clean(x):
    """JSON-safe: NaN / inf -> None, numpy -> python"""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    return x


data = {"run": str(run), "proxy": "proxy" in run.name}

# ---------------- LLM labels ----------------
lab_path = Path("data/results/llm_labels/labels.csv")
if lab_path.exists():
    L = pd.read_csv(lab_path)
    E = pd.read_csv("data/results/event_study/events_entry.csv")[["ric", "date", "car_e1_10", "low_e1_10"]]
    d = L.merge(E, on=["ric", "date"], how="left")
    g, o = "verdict_gpt-4o", "verdict_Claude-Opus-5"
    sent = d[d.n_headlines > 0]
    both = sent[sent[g].notna() & sent[o].notna()]
    order = ["justified", "unsure", "not justified"]
    ct = pd.crosstab(both[g], both[o]).reindex(index=order, columns=order, fill_value=0)
    nj = d[d[g].eq("not justified") & d[o].eq("not justified")]
    cats = (pd.DataFrame({"opus": nj["category_Claude-Opus-5"].value_counts(),
                          "gpt": nj["category_gpt-4o"].value_counts()}).fillna(0).astype(int))
    cats = cats.assign(t=cats.sum(axis=1)).sort_values("t").drop(columns="t")
    groups = {
        "All events": d,
        "No Reuters news": d[d.n_headlines == 0],
        "GPT-4o not justified": d[d[g].eq("not justified")],
        "Opus not justified": d[d[o].eq("not justified")],
        "Both not justified": nj,
        "Both justified": d[d[g].eq("justified") & d[o].eq("justified")],
    }
    outcome = [{"group": k, "n": int(v.car_e1_10.notna().sum()), "car10": v.car_e1_10.mean(),
                "tail": (v.car_e1_10 < -0.10).mean(), "low_p10": v.low_e1_10.quantile(0.1)}
               for k, v in groups.items()]
    cat_out = (nj.groupby("category_Claude-Opus-5")
               .agg(n=("car_e1_10", "size"), car10=("car_e1_10", "mean"),
                    tail=("car_e1_10", lambda x: (x < -0.10).mean()))
               .query("n >= 10").sort_values("n").reset_index()
               .rename(columns={"category_Claude-Opus-5": "category"}))
    data["llm"] = {"agree": {"x": order, "y": order, "z": ct.values.tolist(), "n": len(both),
                             "pct": float((both[g] == both[o]).mean())},
                   "cats": {"names": cats.index.tolist(), "opus": cats["opus"].tolist(), "gpt": cats["gpt"].tolist(),
                            "n": len(nj)},
                   "outcome": outcome, "cat_outcome": cat_out.to_dict("records"),
                   "n_events": len(d), "n_sent": len(sent)}

# ---------------- backtest ----------------
if (run / "trades.parquet").exists():
    T = pd.read_parquet(run / "trades.parquet")
    tr = T[T.status == "traded"].copy()
    tr["entry_date"] = pd.to_datetime(tr["entry_date"])
    tr["exit_date"] = pd.to_datetime(tr["exit_date"])
    tr["quarter"] = pd.PeriodIndex(tr["entry_date"], freq="Q").astype(str)
    base = tr[tr.label == "baseline"]
    bm = trade_metrics(base)

    # one-at-a-time: each variant vs baseline, all events
    oat = tr.groupby("label").apply(trade_metrics, include_groups=False)
    oat["delta"] = oat["mean_ror"] - bm["mean_ror"]
    oat = oat.drop(index="baseline").sort_values("delta")
    oat_rows = [{"label": k, **v} for k, v in oat[["n", "win_rate", "mean_ror", "delta", "worst5_ror"]]
                .to_dict("index").items()]

    # baseline per quarter
    q = base.groupby("quarter").apply(trade_metrics, include_groups=False).reset_index()

    # variant x quarter heatmap (top / bottom variants by delta keep it readable)
    keep = ["baseline"] + oat.index.tolist()
    hm = tr[tr.label.isin(keep)].groupby(["label", "quarter"])["ror"].mean().unstack()
    hm = hm.reindex(["baseline"] + oat.sort_values("delta", ascending=False).index.tolist())
    status = T[T.label == "baseline"].status.value_counts()

    # event filters (baseline trade params): label window only
    filt = {}
    lab_events = None
    if lab_path.exists():
        lab = pd.read_csv(lab_path).rename(columns=lambda c: c.replace("_gpt-4o", "_g").replace("_Claude-Opus-5", "_o"))
        lab["event_id"] = lab["ric"] + "|" + lab["date"]
        lab_events = lab.set_index("event_id")
        b2 = base[base.event_id.isin(lab_events.index)]
        lv = lab_events.loc[b2.event_id]
        masks = {
            "All (label window)": np.ones(len(b2), bool),
            "Has news": (lv.n_headlines > 0).to_numpy(),
            "Both not justified": (lv.verdict_g.eq("not justified") & lv.verdict_o.eq("not justified")).to_numpy(),
            "Both justified": (lv.verdict_g.eq("justified") & lv.verdict_o.eq("justified")).to_numpy(),
        }
        gics = pd.read_parquet("data/raw/reference/gics.parquet").set_index("ric")["sector"].to_dict()
        for name, m in masks.items():
            sub = b2[m]
            _, eq, st = portfolio_curve(sub, gics)
            met = trade_metrics(sub)
            filt[name] = {"metrics": met.to_dict(), "portfolio": st,
                          "eq_x": [x.strftime("%Y-%m-%d") for x in eq.index], "eq_y": (eq / 100_000 - 1).tolist()}

    # walk-forward
    wf = pd.read_csv(run / "walk_forward.csv") if (run / "walk_forward.csv").exists() else pd.DataFrame()
    if len(wf):  # drop filter settings that are at their defaults, so the labels stay readable
        for default in [" category=any", " min_ratio=None", " max_shock=-0.05", " mcap_floor=5000000000.0"]:
            wf["chosen"] = wf["chosen"].str.replace(default, "", regex=False)
        wf["chosen"] = wf["chosen"].str.replace("event_set=", "events: ", regex=False)
    wf_rows = wf.to_dict("records") if len(wf) else []

    exits = base.exit_reason.value_counts()
    data["bt"] = {"baseline": bm.to_dict(), "n_events": int(T.event_id.nunique()),
                  "status": {"names": status.index.tolist(), "n": status.tolist()},
                  "exits": {"names": exits.index.tolist(), "n": exits.tolist()},
                  "oat": oat_rows, "quarter": q.to_dict("records"),
                  "heat": {"x": hm.columns.tolist(), "y": hm.index.tolist(), "z": hm.values.tolist()},
                  "filters": filt, "wf": wf_rows,
                  "ror_hist": base["ror"].clip(-1, 1).tolist()}

payload = json.dumps(clean(data))

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shock Strategy Report</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;
--neg:#e34948;--pos:#2a78d6;--mid:#f0efec;--warnbg:#fff4dc;--warnink:#6b4a00}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--neg:#e66767;--pos:#3987e5;--mid:#383835;--warnbg:#3a2c0a;--warnink:#fab219}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--neg:#e66767;--pos:#3987e5;--mid:#383835;--warnbg:#3a2c0a;--warnink:#fab219}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:40px 0 4px}h3{font-size:15px;margin:0 0 2px}
.sub{color:var(--ink2);margin:0 0 16px}.note{color:var(--ink2);font-size:13px;margin:4px 0 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;min-width:0}
.chart{width:100%;height:340px}.tall{height:520px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:12px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile .k{color:var(--ink2);font-size:13px}.tile .v{font-size:24px;font-weight:600}
.banner{background:var(--warnbg);color:var(--warnink);border-radius:10px;padding:10px 14px;margin:12px 0;font-size:14px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--grid)}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}.scroll{overflow-x:auto}
nav{display:flex;gap:16px;flex-wrap:wrap;font-size:14px;margin:8px 0}nav a{color:var(--s1)}
</style></head><body><main>
<h1>Shock strategy report</h1>
<p class="sub" id="meta"></p>
<nav><a href="#llm">LLM labels</a><a href="#bt">Backtest</a><a href="#wf">Walk-forward</a></nav>

<section id="llm"></section>
<section id="bt"></section>
<section id="wf"></section>
</main>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const pct = (x, d=1) => x==null ? "–" : (x*100).toFixed(d) + "%";
const num = (x, d=3) => x==null ? "–" : x.toFixed(d);
const el = (tag, html, cls) => { const e = document.createElement(tag); if (cls) e.className = cls; e.innerHTML = html; return e; };
function layout(extra={}) {
  const ax = {gridcolor: css('--grid'), linecolor: css('--axis'), zerolinecolor: css('--axis'), tickfont:{color: css('--muted')},
              title:{font:{color: css('--ink2'), size:13}}, automargin:true};
  return Object.assign({paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{family:'system-ui,-apple-system,"Segoe UI",sans-serif', color: css('--ink'), size:12},
    margin:{l:10,r:10,t:10,b:10}, xaxis:Object.assign({}, ax, extra.xaxis||{}), yaxis:Object.assign({}, ax, extra.yaxis||{}),
    legend:{orientation:'h', y:1.12, x:0, font:{color: css('--ink2')}}, hoverlabel:{bgcolor: css('--surface'), bordercolor: css('--axis'), font:{color: css('--ink')}},
    bargap:0.25}, Object.fromEntries(Object.entries(extra).filter(([k])=>!['xaxis','yaxis'].includes(k))));
}
const CFG = {displayModeBar:false, responsive:true};
const charts = [];
function plot(id, traces, lay) { charts.push([id, traces, lay]); }
function card(parent, title, note, id, cls='chart') {
  const c = el('div', `<h3>${title}</h3>` + (note ? `<p class="note">${note}</p>` : '') + (id ? `<div id="${id}" class="${cls}"></div>` : ''), 'card');
  parent.appendChild(c); return c;
}
function diverging() { return [[0, css('--neg')], [0.5, css('--mid')], [1, css('--pos')]]; }
function tiles(parent, items) {
  const t = el('div', items.map(([k, v]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`).join(''), 'tiles');
  parent.appendChild(t);
}
function table(parent, cols, rows) {
  const h = '<div class="scroll"><table><thead><tr>' + cols.map(c => `<th>${c[0]}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map(r => '<tr>' + cols.map(c => `<td>${c[1](r)}</td>`).join('') + '</tr>').join('') + '</tbody></table></div>';
  parent.appendChild(el('div', h));
}

document.getElementById('meta').textContent = `Run: ${D.run}` + (D.proxy ? ' · proxy option prices' : '');

// ---------------- LLM ----------------
if (D.llm) {
  const s = document.getElementById('llm');
  s.appendChild(el('h2', 'LLM labels: GPT-4o vs Claude Opus 5'));
  s.appendChild(el('p', `${D.llm.n_events} shocks, ${D.llm.n_sent} with Reuters news before the close. Outcome = abnormal return over 10 days, entering the day after the shock.`, 'sub'));
  const g = el('div', '', 'grid'); s.appendChild(g);
  const a = D.llm.agree;
  card(g, `Verdicts agree on ${pct(a.pct,0)} of ${a.n} events`, 'Rows: GPT-4o. Columns: Opus. Cell = number of events.', 'c_agree');
  plot('c_agree', [{type:'heatmap', x:a.x, y:a.y, z:a.z, colorscale:[[0, css('--mid')],[1, css('--s1')]], showscale:false, xgap:2, ygap:2,
     text:a.z.map(r=>r.map(String)), texttemplate:'%{text}', textfont:{size:14},
     hovertemplate:'GPT-4o %{y}<br>Opus %{x}<br>%{z} events<extra></extra>'}],
     layout({xaxis:{title:{text:'Opus'}, side:'bottom'}, yaxis:{title:{text:'GPT-4o'}}}));
  const c = D.llm.cats;
  card(g, `Categories when both say "not justified" (${c.n} events)`, 'Most of the list is "unclear" or macro: no company-specific reason found.', 'c_cats', 'chart tall');
  plot('c_cats', [
     {type:'bar', orientation:'h', name:'Opus', y:c.names, x:c.opus, marker:{color:css('--s1')}, hovertemplate:'%{y}<br>Opus: %{x}<extra></extra>'},
     {type:'bar', orientation:'h', name:'GPT-4o', y:c.names, x:c.gpt, marker:{color:css('--s2')}, hovertemplate:'%{y}<br>GPT-4o: %{x}<extra></extra>'}],
     layout({barmode:'group', xaxis:{title:{text:'events'}}}));
  const o = D.llm.outcome;
  const g2 = el('div', '', 'grid'); s.appendChild(g2);
  card(g2, 'Mean 10-day abnormal return by label group', 'Bars left of zero kept falling. "Both justified" is the only group that stands out.', 'c_out');
  plot('c_out', [{type:'bar', orientation:'h', y:o.map(r=>r.group), x:o.map(r=>r.car10),
     marker:{color:o.map(r=>r.car10<0?css('--neg'):css('--pos'))}, customdata:o.map(r=>[r.n, r.tail]),
     text:o.map(r=>pct(r.car10)), textposition:'outside', cliponaxis:false,
     hovertemplate:'%{y}<br>mean %{x:.2%}<br>n = %{customdata[0]}<extra></extra>'}],
     layout({xaxis:{tickformat:'.1%', zeroline:true}, yaxis:{autorange:'reversed'}}));
  card(g2, 'Tail: share that fell another 10%+ vs the market', 'The filter is useful only if "not justified" has a thinner tail than all events.', 'c_tail');
  plot('c_tail', [{type:'bar', orientation:'h', y:o.map(r=>r.group), x:o.map(r=>r.tail), marker:{color:css('--s1')},
     text:o.map(r=>pct(r.tail)), textposition:'outside', cliponaxis:false, customdata:o.map(r=>r.n),
     hovertemplate:'%{y}<br>%{x:.1%} fell 10%+<br>n = %{customdata}<extra></extra>'}],
     layout({xaxis:{tickformat:'.0%', rangemode:'tozero'}, yaxis:{autorange:'reversed'}}));
  const co = D.llm.cat_outcome;
  const cc = card(s, 'Both "not justified": outcome by Opus category (10+ events)', 'Analyst downgrades have the thinnest tail; contagion and management changes the fattest.', 'c_catout');
  cc.style.marginTop = '16px';
  plot('c_catout', [{type:'scatter', mode:'markers+text', x:co.map(r=>r.car10), y:co.map(r=>r.tail), text:co.map(r=>r.category),
     textposition:'top center', textfont:{color:css('--ink2')}, customdata:co.map(r=>r.n),
     marker:{size:co.map(r=>Math.max(10, Math.sqrt(r.n)*3)), color:css('--s1'), line:{color:css('--surface'), width:2}},
     hovertemplate:'%{text}<br>mean %{x:.2%}<br>tail %{y:.1%}<br>n = %{customdata}<extra></extra>'}],
     layout({xaxis:{title:{text:'mean 10-day abnormal return'}, tickformat:'.1%', zeroline:true}, yaxis:{title:{text:'share falling 10%+'}, tickformat:'.0%'}}));
}

// ---------------- backtest ----------------
if (D.bt) {
  const B = D.bt, s = document.getElementById('bt');
  s.appendChild(el('h2', 'Backtest: bull put spread, one change at a time'));
  if (D.proxy) s.appendChild(el('div', '<b>Proxy prices.</b> Puts are priced with Black-Scholes from pre-shock realized vol (no real IV spike or skew). These charts check the machinery, not the strategy. Rerun with <code>--pricer panel</code> once the OPRA option panel is on this machine.', 'banner'));
  const b = B.baseline;
  tiles(s, [['Events', B.n_events.toLocaleString()], ['Baseline trades', b.n.toLocaleString()], ['Win rate', pct(b.win_rate,0)],
            ['Mean return on risk', pct(b.mean_ror)], ['Worst 5% trade', pct(b.worst5_ror,0)], ['t (by entry date)', num(b.t_by_date,2)]]);
  const g = el('div', '', 'grid'); s.appendChild(g);
  card(g, 'Baseline: what happened to each event', 'Most shocks never pass the entry trigger and credit check.', 'c_status');
  plot('c_status', [{type:'bar', orientation:'h', y:B.status.names, x:B.status.n, marker:{color:css('--s1')}, text:B.status.n, textposition:'outside', cliponaxis:false,
     hovertemplate:'%{y}: %{x}<extra></extra>'}], layout({yaxis:{autorange:'reversed'}}));
  card(g, 'Baseline trades: return on risk', 'P&L divided by the spread\'s max loss. -100% = full loss.', 'c_hist');
  plot('c_hist', [{type:'histogram', x:B.ror_hist, xbins:{start:-1, end:1.0001, size:0.05}, marker:{color:css('--s1'), line:{color:css('--surface'), width:1}},
     hovertemplate:'%{x}<br>%{y} trades<extra></extra>'}], layout({xaxis:{tickformat:'.0%', title:{text:'return on risk'}}, yaxis:{title:{text:'trades'}}}));
  const oat = B.oat;
  const c1 = card(s, 'Each parameter change vs baseline (mean return on risk, all events)', `Baseline mean ${pct(b.mean_ror)}. Right = better than baseline. Hover for trade count and win rate.`, 'c_oat', 'chart tall');
  c1.style.marginTop = '16px';
  document.getElementById('c_oat').style.height = Math.max(420, oat.length * 22) + 'px';
  plot('c_oat', [{type:'bar', orientation:'h', y:oat.map(r=>r.label), x:oat.map(r=>r.delta),
     marker:{color:oat.map(r=>r.delta<0?css('--neg'):css('--pos'))}, customdata:oat.map(r=>[r.n, r.win_rate, r.mean_ror]),
     hovertemplate:'%{y}<br>vs baseline %{x:+.2%}<br>mean %{customdata[2]:.2%} · win %{customdata[1]:.0%} · n %{customdata[0]}<extra></extra>'}],
     layout({xaxis:{tickformat:'+.1%', zeroline:true}}));
  const q = B.quarter;
  const g2 = el('div', '', 'grid'); g2.style.marginTop = '16px'; s.appendChild(g2);
  card(g2, 'Baseline mean return on risk by quarter', 'All events, baseline parameters.', 'c_q');
  plot('c_q', [{type:'bar', x:q.map(r=>r.quarter), y:q.map(r=>r.mean_ror), marker:{color:q.map(r=>r.mean_ror<0?css('--neg'):css('--pos'))},
     customdata:q.map(r=>[r.n, r.win_rate]), hovertemplate:'%{x}<br>mean %{y:.2%}<br>n %{customdata[0]} · win %{customdata[1]:.0%}<extra></extra>'}],
     layout({yaxis:{tickformat:'.0%'}, xaxis:{type:'category', nticks:12}}));
  card(g2, 'Baseline trades per quarter', 'Shown separately so the return chart keeps one axis.', 'c_qn');
  plot('c_qn', [{type:'bar', x:q.map(r=>r.quarter), y:q.map(r=>r.n), marker:{color:css('--s1')}, hovertemplate:'%{x}: %{y} trades<extra></extra>'}],
     layout({xaxis:{type:'category', nticks:12}}));
  const h = B.heat;
  const c2 = card(s, 'Every parameter variant × quarter (mean return on risk)', 'Blue = profit, red = loss, gray ≈ 0. A variant that is only good in a few quarters is luck, not an edge.', 'c_heat', 'chart tall');
  c2.style.marginTop = '16px';
  document.getElementById('c_heat').style.height = Math.max(460, h.y.length * 18) + 'px';
  const lim = 0.3;
  plot('c_heat', [{type:'heatmap', x:h.x, y:h.y, z:h.z, zmin:-lim, zmax:lim, colorscale:diverging(), xgap:1, ygap:1,
     colorbar:{tickformat:'.0%', outlinewidth:0, tickfont:{color:css('--muted')}},
     hovertemplate:'%{y}<br>%{x}: %{z:.1%}<extra></extra>'}], layout({yaxis:{autorange:'reversed'}, xaxis:{type:'category', nticks:16}}));
  const F = B.filters, names = Object.keys(F);
  if (names.length) {
    const colors = [css('--s1'), css('--s2'), css('--s3'), css('--s4')];
    const g3 = el('div', '', 'grid'); g3.style.marginTop = '16px'; s.appendChild(g3);
    card(g3, 'Book equity by LLM filter (baseline trades, label window)', '2% of $100k risked per trade, one per ticker, 30% sector cap. Cumulative return.', 'c_eq');
    plot('c_eq', names.map((n, i) => ({type:'scatter', mode:'lines', name:n, x:F[n].eq_x, y:F[n].eq_y, line:{color:colors[i], width:2},
      hovertemplate:n+'<br>%{x}: %{y:.1%}<extra></extra>'})), layout({yaxis:{tickformat:'.0%'}, hovermode:'x unified'}));
    const c4 = card(g3, 'By LLM filter', null, null);
    table(c4, [['Filter', r=>r[0]], ['Trades', r=>r[1].metrics.n], ['Win', r=>pct(r[1].metrics.win_rate,0)], ['Mean RoR', r=>pct(r[1].metrics.mean_ror)],
               ['Worst 5%', r=>pct(r[1].metrics.worst5_ror,0)], ['Book return', r=>pct(r[1].portfolio.total_return)], ['Max DD', r=>pct(r[1].portfolio.max_drawdown)],
               ['Sharpe', r=>num(r[1].portfolio.sharpe,2)]], Object.entries(F));
  }
  card(s, 'Baseline exits', null, 'c_exit').style.marginTop = '16px';
  plot('c_exit', [{type:'bar', orientation:'h', y:B.exits.names, x:B.exits.n, marker:{color:css('--s1')}, text:B.exits.n, textposition:'outside', cliponaxis:false,
     hovertemplate:'%{y}: %{x}<extra></extra>'}], layout({yaxis:{autorange:'reversed'}}));

  // ---------------- walk-forward ----------------
  const w = B.wf, ws = document.getElementById('wf');
  ws.appendChild(el('h2', 'Walk-forward by quarter'));
  ws.appendChild(el('p', 'Each quarter: pick the best combination on the previous 4 quarters, then record how it did in this quarter (out-of-sample). Compare with always using the baseline.', 'sub'));
  if (w.length) {
    let cs = 0, cb = 0;
    const cumC = w.map(r => cs += (r.test_sum_ror||0)), cumB = w.map(r => cb += ((r.base_mean_ror||0) * (r.base_n||0)));
    const g4 = el('div', '', 'grid'); ws.appendChild(g4);
    card(g4, 'Out-of-sample mean return on risk per test quarter', null, 'c_wf');
    plot('c_wf', [
      {type:'bar', name:'Walk-forward pick', x:w.map(r=>r.test_quarter), y:w.map(r=>r.test_mean_ror), marker:{color:css('--s1')}, customdata:w.map(r=>[r.chosen, r.test_n]),
       hovertemplate:'%{x}<br>pick: %{customdata[0]}<br>mean %{y:.2%} · n %{customdata[1]}<extra></extra>'},
      {type:'bar', name:'Always baseline', x:w.map(r=>r.test_quarter), y:w.map(r=>r.base_mean_ror), marker:{color:css('--s2')}, customdata:w.map(r=>r.base_n),
       hovertemplate:'%{x}<br>baseline mean %{y:.2%} · n %{customdata}<extra></extra>'}],
      layout({barmode:'group', yaxis:{tickformat:'.0%'}, xaxis:{type:'category', nticks:16}}));
    card(g4, 'Cumulative out-of-sample return on risk (sum over trades)', 'If the pick line is not clearly above baseline, the parameter search is not adding anything.', 'c_wfc');
    plot('c_wfc', [
      {type:'scatter', mode:'lines+markers', name:'Walk-forward pick', x:w.map(r=>r.test_quarter), y:cumC, line:{color:css('--s1'), width:2}, marker:{size:8}},
      {type:'scatter', mode:'lines+markers', name:'Always baseline', x:w.map(r=>r.test_quarter), y:cumB, line:{color:css('--s2'), width:2}, marker:{size:8}}],
      layout({xaxis:{type:'category', nticks:16}, yaxis:{title:{text:'sum of return on risk'}}, hovermode:'x unified'}));
    const c5 = card(ws, 'Choice per test quarter', null, null); c5.style.marginTop = '16px';
    table(c5, [['Test quarter', r=>r.test_quarter], ['Trained on', r=>r.train], ['Chosen', r=>`<span style="text-align:left">${r.chosen}</span>`],
               ['Train mean', r=>pct(r.train_mean_ror)], ['Test n', r=>r.test_n], ['Test mean', r=>pct(r.test_mean_ror)],
               ['Baseline mean', r=>pct(r.base_mean_ror)]], w);
  } else ws.appendChild(el('p', 'No walk-forward folds in this run.', 'note'));
}

function render() { for (const [id, t, l] of charts) { const e = document.getElementById(id); if (e) Plotly.react(e, t, l(), CFG); } }
// layouts read CSS variables, so wrap them to re-read on theme change
for (const c of charts) { const lay = c[2]; c[2] = () => lay; }
render();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => location.reload());
</script></body></html>"""

out = run / "report.html" if run.exists() else Path("data/results/llm_labels/report.html")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(HTML.replace("__DATA__", payload), encoding="utf-8")
print(f"wrote {out}")
