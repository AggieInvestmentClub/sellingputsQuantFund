"""
Shock-screening agent (AG, docs/code_reference/AG_agent.md): for each shock, anonymized Reuters news
+ market evidence -> LLM -> take / skip. Builds on the LLM filter (LLM-01..05) and its cache.

    python -u scripts/run_agent.py --dry-run --limit 3                 # print prompts, no API calls
    python -u scripts/run_agent.py --start 2024-04-01 --limit 20       # first batch
    python -u scripts/run_agent.py --start 2024-04-01                  # every option-period event
    python -u scripts/run_agent.py --ric MPC.N --date 2026-10-27       # one shock (live use)

Model (src/agent/llm.py, OpenAI-compatible): AGENT_MODEL / AGENT_BASE_URL / AGENT_API_KEY in .env, else the LLM filter's LLM_* settings.
--provider openrouter uses OPENROUTER_* (Llama 3 70B: training data ends Dec 2023, so it cannot know
how any 2024-2026 event turned out: every option-period event is a clean test).

Needs: data/results/event_study/events.csv, data/raw/prices/daily/, market_cap/, reference/gics.parquet,
earnings/earnings_dates.parquet, universe/security_master.parquet, news/reuters_events + reuters_stories
(scripts/download_event_news.py). Writes data/results/agent/agent_results.csv (batch) and caches every
answer in data/results/agent/cache/ (re-runs are free and identical).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.agent.evidence import (abnormal_on, earnings_context, event_news, load_close, market_cap_before,
                                members_on, peer_context, price_context)
from src.agent.screen import AGENT_VERSION, build_agent_prompt, final_call, parse_agent
from src.agent.valuation import gate, valuation_context, valuation_lines
from src.agent.llm import TAMU_BASE_URL, call_llm, company_aliases

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def model_settings(provider=None, model=None):
    if provider == "openrouter":
        s = (os.getenv("OPENROUTER_API_KEY"), os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
             os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-70b-instruct"))
    else:
        s = (os.getenv("AGENT_API_KEY") or os.getenv("LLM_API_KEY"),
             os.getenv("AGENT_BASE_URL") or os.getenv("LLM_BASE_URL", TAMU_BASE_URL),
             os.getenv("AGENT_MODEL") or os.getenv("LLM_MODEL"))
    return s[0], s[1], model or s[2]


class Agent:
    """loads the reference tables once; screen() runs one shock"""

    def __init__(self, data_root="data", provider=None, model=None, dry_run=False, fv_gate=False):
        root = Path(data_root)
        self.raw, self.out = root / "raw", root / "results" / "agent"
        self.cache = self.out / "cache"
        self.key, self.base, self.model = model_settings(provider, model)
        self.dry_run = dry_run
        self.fv_gate = fv_gate            # True: the fair-value gate (AG-15) can veto a TAKE
        if not dry_run and not (self.key and self.model):
            raise SystemExit("No API key / model: set AGENT_* or LLM_* in .env (or --provider openrouter). "
                             "Use --dry-run to only look at prompts.")
        self.master = pd.read_parquet(self.raw / "universe" / "security_master.parquet")
        na = lambda x: None if pd.isna(x) else str(x)          # security_master has <NA> roots / names
        self.names = {r.ric: (na(r.name), na(r.ticker), na(r.lseg_option_root)) for r in self.master.itertuples()}
        gp = self.raw / "reference" / "gics.parquet"
        self.gics = pd.read_parquet(gp) if gp.exists() else pd.DataFrame(columns=["ric", "sector", "industry_group", "industry"])
        self.gics_ix = self.gics.set_index("ric")
        ep = self.raw / "earnings" / "earnings_dates.parquet"
        self.earnings = pd.read_parquet(ep) if ep.exists() else pd.DataFrame(columns=["ric", "report_date"])
        cp = root / "results" / "llm" / "cutoff.json"
        self.cutoff = None
        if cp.exists():
            d = json.loads(cp.read_text())
            m = d.get("confirmed_cutoff") or d.get("probe_cutoff")
            self.cutoff = pd.Timestamp(m) + pd.offsets.MonthEnd(0) if m else None

    def screen(self, ric, date, abnormal=None):
        date = pd.Timestamp(date).normalize()
        if abnormal is None or not np.isfinite(abnormal):
            a = abnormal_on(self.raw, ric, date)
            if a is None:
                return {"event_id": f"{ric}|{date:%Y-%m-%d}", "ric": ric, "date": date.date(),
                        "status": "no price data for this date (update data/raw/prices/daily first)"}
            abnormal = a[0]
        name, ticker, root = self.names.get(ric, (None, ric.split(".")[0], None))
        heads, stories = event_news(self.raw, ric, date)
        price = price_context(load_close(self.raw, ric), date)
        peers = peer_context(self.raw, ric, date, self.gics, members_on(self.master, date))
        earn = earnings_context(self.earnings, ric, date)
        cap = market_cap_before(self.raw, ric, date)
        sector = self.gics_ix.loc[ric, "sector"] if ric in self.gics_ix.index else None
        industry = self.gics_ix.loc[ric, "industry"] if ric in self.gics_ix.index else None
        val = valuation_context(self.raw, ric, date, self.gics, members_on(self.master, date))
        fv_ok, fv_why = gate(val)
        msgs, facts = build_agent_prompt(heads, stories, company_aliases(name, ticker, ric, root),
                                         sector, industry, cap, abnormal, price, peers, earn,
                                         valuation_lines(val) if val else None)
        row = {"event_id": f"{ric}|{date:%Y-%m-%d}", "ric": ric, "date": date.date(), "name": name,
               "abnormal": abnormal, "market_cap": cap, "has_news": len(heads) > 0,
               "post_cutoff": None if self.cutoff is None else bool(date > self.cutoff),
               "ret_21d_before": price and price["ret_21d"], "peer_median": peers and peers["median"],
               "next_earnings_in_window": earn["next_in_window"], "agent_version": AGENT_VERSION,
               **({f"val_{k}": v for k, v in val.items()} if val else {}), "fv_pass": fv_ok, "fv_why": "; ".join(fv_why)}
        if self.dry_run:
            return {**row, "status": "dry-run", "prompt": msgs[1]["content"]}
        res = call_llm(msgs, self.cache, self.model, self.key, self.base, max_tokens=900)
        p = parse_agent(res["content"])
        call = final_call(p, abnormal, cap, earn, fv_fail="; ".join(fv_why) if (self.fv_gate and not fv_ok) else None)
        return {**row, **call, "decision": p["decision"], "verdict": p["verdict"], "confidence": p["confidence"],
                "impact_mid": p["impact_mid"], "market_read": p["market_read"], "key_risk": p["key_risk"],
                "reason": p["reason"], "parse_ok": p["parse_ok"], "model": res["model"], "cached": res["cached"],
                "status": "ok"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--provider", choices=["openrouter"], help="default: AGENT_* / LLM_* from .env")
    ap.add_argument("--model")
    ap.add_argument("--start", default="2024-04-01", help="first shock date (option data starts 2024-04)")
    ap.add_argument("--end")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ric")
    ap.add_argument("--date")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fv-gate", action="store_true", help="let the fair-value gate (AG-15) veto a TAKE")
    args = ap.parse_args()

    agent = Agent(args.data_root, args.provider, args.model, args.dry_run, args.fv_gate)
    agent.out.mkdir(parents=True, exist_ok=True)

    if args.ric:
        r = agent.screen(args.ric, args.date or pd.Timestamp.today())
        if args.dry_run and "prompt" in r:
            print(r.pop("prompt"))
        print(json.dumps(r, indent=1, default=str))
        return

    ev = pd.read_csv(Path(args.data_root) / "results" / "event_study" / "events.csv", parse_dates=["date"])
    ev = ev[ev["date"] >= args.start]
    if args.end:
        ev = ev[ev["date"] <= args.end]
    ev = ev.sort_values("date")
    if args.limit:
        ev = ev.head(args.limit)
    print(f"{len(ev)} shocks | model {agent.model} @ {agent.base} | cutoff {agent.cutoff}", flush=True)

    rows, t0 = [], time.time()
    for i, e in enumerate(ev.itertuples(), 1):
        try:
            r = agent.screen(e.ric, e.date, e.abnormal)
        except Exception as ex:          # keep going; failed rows are not cached and rerun next time
            r = {"event_id": f"{e.ric}|{e.date:%Y-%m-%d}", "ric": e.ric, "date": e.date.date(),
                 "status": f"error: {type(ex).__name__}: {str(ex)[:200]}"}
        if args.dry_run:
            print(f"===== {r['event_id']} ({r.get('name')}) =====\n{r.pop('prompt', r['status'])}\n")
        rows.append(r)
        if i % 25 == 0:
            print(f"  {i}/{len(ev)} ({time.time() - t0:.0f}s)", flush=True)
    if args.dry_run:
        return
    out = pd.DataFrame(rows)
    path = agent.out / "agent_results.csv"
    out.to_csv(path, index=False)
    ok = out[out["status"] == "ok"]
    errs = out.loc[out["status"] != "ok", "status"]
    if len(ok):
        print(f"saved {path}: {len(out)} shocks | ok {len(ok)} | take {int(ok['agent_take'].sum())} "
              f"({ok['agent_take'].mean():.0%}) | decision=take {(ok['decision'] == 'take').mean():.0%} | "
              f"errors {len(errs)}")
    else:
        print(f"saved {path}: {len(out)} shocks | ALL FAILED")
    if len(errs):
        print("errors:", errs.value_counts().head(3).to_dict())
        print("diagnose the API with: py -3.12 -u scripts/agent_diag.py")
    if not len(ok):
        return
    if ok["model"].nunique() > 1:
        print(f"WARNING: several model versions answered: {list(ok['model'].unique())}")


if __name__ == "__main__":
    main()
