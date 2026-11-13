"""
Backtest the bull put spread on the shock events: one-at-a-time parameter sweep or a small grid,
crossed with event filters (LLM verdicts, categories, shock size, market cap), reported per quarter,
plus a walk-forward test that picks the best combination on past quarters and scores it on the next.
Config: strategy/backtest_grid.yaml.

Run from the repo root:
    python -u scripts/run_backtest.py --limit 50                         (smoke test, proxy prices)
    python -u scripts/run_backtest.py --mode oat                         (baseline + one change at a time)
    python -u scripts/run_backtest.py --mode grid                        (full product of grid_dims)
    python -u scripts/run_backtest.py --pricer panel --start 2024-04-08  (real OPRA option panel)
--pricer proxy (default) prices puts with Black-Scholes from pre-shock realized vol: it checks the
plumbing, it does NOT measure the strategy. Use --pricer panel once data/processed/options/event_puts.parquet
exists (scripts/build_option_panel.py).

Writes data/results/backtest/<mode>_<pricer>/:
    trades.parquet          every simulated (combination, event): status, strikes, credit, P&L, ror
    status.csv              why events were not traded, per combination
    summary.csv             metrics per (combination x event filter), all quarters
    quarterly_params.csv    all events, each parameter variant x quarter (the per-quarter baseline)
    quarterly_filters.csv   baseline parameters, each event filter x quarter
    walk_forward.csv        per test quarter: chosen candidate, its out-of-sample result, baseline
    walk_forward.txt        stitched out-of-sample result vs always-baseline
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import yaml

from src.backtest.engine import BASELINE, simulate_event
from src.backtest.grid import (combo_id, event_filters, event_mask, expand_grid, one_at_a_time,
                               portfolio_curve, trade_metrics)
from src.backtest.market import PanelPricer, ProxyPricer, fname, load_rates, load_stock
from src.backtest.walk_forward import quarter_folds, quarterly_table, walk_forward

PANEL = Path("data/processed/options/event_puts.parquet")


def build_events(start, end, limit, path="data/results/event_study/events.csv"):
    ev = pd.read_csv(path)
    ev = ev[(ev.date >= start) & (ev.date <= end)].reset_index(drop=True)
    if limit:
        ev = ev.head(limit)
    ev["event_id"] = ev["ric"] + "|" + ev["date"]
    lab_path = Path("data/results/llm_labels/labels.csv")
    if lab_path.exists():
        lab = pd.read_csv(lab_path).rename(columns=lambda c: c.replace("_gpt-4o", "_gpt4o")
                                           .replace("_Claude-Opus-5", "_opus"))
        keep = ["ric", "date", "n_headlines", "mcap_lost", "market_cap",
                *[c for c in lab.columns if c.endswith(("_gpt4o", "_opus"))]]
        ev = ev.merge(lab[keep], on=["ric", "date"], how="left")
    for c in ["n_headlines", "verdict_gpt4o", "verdict_opus", "category_gpt4o", "category_opus",
              "impact_mid_opus", "market_cap", "mcap_lost"]:
        if c not in ev:
            ev[c] = np.nan
    missing = ev["market_cap"].isna()
    for i in ev.index[missing]:
        p = Path("data/raw/market_cap") / f"{fname(ev.at[i, 'ric'])}.parquet"
        if p.exists():
            m = pd.read_parquet(p)["market_cap"].dropna()
            m = m[pd.to_datetime(m.index) < pd.Timestamp(ev.at[i, "date"])]
            ev.at[i, "market_cap"] = float(m.iloc[-1]) if len(m) else np.nan
    ev["ratio_opus"] = (-ev["abnormal"] * ev["market_cap"]) / ev["impact_mid_opus"].replace(0, np.nan)
    ev.loc[ev["impact_mid_opus"].eq(0), "ratio_opus"] = np.inf
    return ev


def run_ric(job):
    """worker: every event of one stock x every trade combination"""
    ric, events, combos, cfg, pricer_kind, panel_rows = job
    stock = load_stock(ric)
    out = []
    if stock is None:
        return [{"label": lab, "event_id": e["event_id"], "status": "no price file"}
                for e in events for lab, _ in combos]
    rates = load_rates()
    if pricer_kind == "panel":
        pricer = PanelPricer(panel_rows, rates, half_spread=cfg.get("half_spread", 0.05))
    else:
        pricer = ProxyPricer(rates, half_spread=cfg.get("half_spread", 0.05),
                             **{k: v for k, v in cfg.get("proxy", {}).items() if k != "half_spread"})
    earn = pd.read_parquet("data/raw/earnings/earnings_dates.parquet")
    earn = list(pd.to_datetime(earn.loc[earn["ric"] == ric, "report_date"]))
    for e in events:
        pricer.start(e["event_id"], stock, e["date"])
        for lab, params in combos:
            try:
                r = simulate_event(e, stock, pricer, earn, params)
            except Exception as ex:
                r = {"status": f"error: {type(ex).__name__}: {str(ex)[:80]}"}
            out.append({"label": lab, "event_id": e["event_id"], **r})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="strategy/backtest_grid.yaml")
    ap.add_argument("--mode", choices=["oat", "grid"], default="oat")
    ap.add_argument("--pricer", choices=["proxy", "panel"], default="proxy")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--events", default="data/results/event_study/events.csv",
                    help="event list (e.g. random_events.csv for the random-entry control)")
    ap.add_argument("--panel", default=str(PANEL), help="option panel built for those events")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    base = {**BASELINE, **cfg.get("baseline", {})}
    out = Path(args.out or f"data/results/backtest/{args.mode}_{args.pricer}")
    out.mkdir(parents=True, exist_ok=True)

    # 1. trade combinations
    if args.mode == "oat":
        combos = [(lab, {**cfg.get("baseline", {}), **p}) for lab, p in one_at_a_time(cfg["space"], base)]
    else:
        grid = expand_grid(cfg["space"], base, cfg["grid_dims"])
        combos = [("baseline", dict(cfg.get("baseline", {})))]
        combos += [(" ".join(f"{k}={v}" for k, v in g.items()), {**cfg.get("baseline", {}), **g}) for g in grid]
    print(f"{len(combos)} trade combinations ({args.mode})")

    # 2. events and option data
    events = build_events(args.start, args.end, args.limit, args.events)
    panel = None
    if args.pricer == "panel":
        if not Path(args.panel).exists():
            raise SystemExit(f"{args.panel} not found: run scripts/build_option_panel.py first, or use --pricer proxy")
        panel = pd.read_parquet(args.panel)
        events = events[events["event_id"].isin(set(panel["event_id"]))].reset_index(drop=True)
    print(f"{len(events)} events {events.date.min()} .. {events.date.max()}, pricer={args.pricer}")

    # 3. simulate (parallel by stock)
    jobs = []
    for ric, g in events.groupby("ric"):
        rows = None if panel is None else panel[panel["event_id"].isin(set(g["event_id"]))]
        jobs.append((ric, g[["event_id", "ric", "date"]].to_dict("records"), combos, cfg, args.pricer, rows))
    t0, res = time.time(), []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(run_ric, jobs, chunksize=4), 1):
            res.extend(r)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} stocks ({time.time() - t0:.0f}s)", flush=True)
    trades = pd.DataFrame(res).merge(events[["event_id", "ric", "date"]], on="event_id", how="left")
    trades.to_parquet(out / "trades.parquet", index=False)
    trades.pivot_table(index="label", columns="status", values="event_id", aggfunc="count",
                       fill_value=0).to_csv(out / "status.csv")
    traded = trades[trades["status"] == "traded"].copy()
    print(f"simulated {len(trades):,} (combination, event) pairs, traded {len(traded):,}")
    if traded.empty:
        raise SystemExit("no trades: see status.csv")

    # 4. event filters x combinations
    filters = event_filters(cfg["event_filters"])
    ev = events.set_index("event_id")
    fid = lambda f: " ".join(f"{k}={v}" for k, v in f.items())
    cands = []
    for f in filters:
        keep = set(ev.index[event_mask(ev, **f).to_numpy()])
        cands.append(traded[traded["event_id"].isin(keep)].assign(filter=fid(f)))
    cands = pd.concat(cands, ignore_index=True)
    cands["candidate"] = cands["label"] + " | " + cands["filter"]
    summary = cands.groupby(["label", "filter"]).apply(trade_metrics, include_groups=False)
    summary.sort_values("mean_ror", ascending=False).to_csv(out / "summary.csv")

    default_filter = fid(filters[0])
    quarterly_table(cands[cands["filter"] == default_filter], ["label"]).to_csv(out / "quarterly_params.csv")
    quarterly_table(cands[cands["label"] == "baseline"], ["filter"]).to_csv(out / "quarterly_filters.csv")

    # 5. portfolio view for the baseline under each event filter
    gics = pd.read_parquet("data/raw/reference/gics.parquet").set_index("ric")["sector"].to_dict()
    port = []
    for f, g in cands[cands["label"] == "baseline"].groupby("filter"):
        _, _, st = portfolio_curve(g, gics)
        port.append({"filter": f, **st})
    pd.DataFrame(port).to_csv(out / "portfolio_baseline.csv", index=False)

    # 6. walk-forward over every (combination x filter) candidate
    wf = cfg.get("walk_forward", {})
    quarters = sorted(pd.PeriodIndex(pd.to_datetime(cands["entry_date"]), freq="Q").astype(str).unique())
    folds = quarter_folds(quarters, wf.get("train_quarters", 4), wf.get("expanding", False))
    base_cand = f"baseline | {default_filter}"
    table, oos = walk_forward(cands, "candidate", folds, wf.get("score", "mean_ror"),
                              wf.get("min_trades", 20), baseline=base_cand)
    table.to_csv(out / "walk_forward.csv", index=False)
    base_oos = cands[(cands["candidate"] == base_cand)
                     & pd.PeriodIndex(pd.to_datetime(cands["entry_date"]), freq="Q").astype(str)
                     .isin(set(table.get("test_quarter", [])))]
    lines = [f"walk-forward: {len(folds)} folds, train {wf.get('train_quarters', 4)} quarters "
             f"({'expanding' if wf.get('expanding') else 'rolling'}), score {wf.get('score', 'mean_ror')}, "
             f"{cands['candidate'].nunique()} candidates, pricer={args.pricer}",
             "", "stitched out-of-sample (chosen each quarter):", trade_metrics(oos).round(4).to_string() if len(oos) else "(none)",
             "", f"always baseline ({base_cand}) over the same quarters:", trade_metrics(base_oos).round(4).to_string(),
             "", "how often each candidate was chosen:",
             table["chosen"].value_counts().to_string() if len(table) else "(none)"]
    if args.pricer == "proxy":
        lines += ["", "PROXY PRICES (Black-Scholes from pre-shock realized vol): checks the code, not the strategy."]
    (out / "walk_forward.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
