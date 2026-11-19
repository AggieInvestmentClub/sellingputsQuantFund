"""
AG: one shock -> one take / skip decision.

Builds on the LLM filter (now in src/agent/llm.py): the same anonymized news prompt (LLM-03), the same
OpenAI-compatible client with on-disk cache (LLM-04), the same parser (LLM-05). What the agent adds:
a market-evidence block (AG-02..04, numbers only, nothing that names the company) and an explicit
trade decision that has to weigh news AND market evidence.

The trade rules (strikes, entry, exits) are NOT the agent's job: they stay as backtested
(short delta 0.15, enter within 3 days, no stop). The agent only picks which shocks to trade.
"""
import json
import re

import numpy as np

from src.agent.llm import build_event_prompt, news_category, overreaction_ratio, parse_verdict

AGENT_VERSION = "ag-v1"

DECISION_SPEC = """
Market evidence (computed from prices up to the Day 0 close; numbers only):
{context}

Your job is to decide whether to SELL A PUT SPREAD with the short strike well below the Day 0 low,
held 2-5 weeks. That trade wins if the stock does not keep falling. Our research shows that selling
puts on random days does about as well as selling after drops, so only say "take" when the evidence
shows the drop is clearly bigger than the news justifies AND the selling is likely done.
Typical "skip" signs: guidance cut, fraud/accounting, regulatory action, debt or liquidity stress,
the whole industry fell too, the stock was already in a steep downtrend, unclear news.
Typical "take" signs: one-off or misunderstood news with a small dollar impact, industry flat,
stock was healthy before, very heavy volume (capitulation).
Most drops are NOT clear overreactions: "take" should be the minority answer.

Return JSON:
{{"category": one of ["analyst downgrade", "peer/sector contagion", "lawsuit", "regulatory probe",
  "short seller report", "product recall", "dilution/offering", "guidance cut", "management change",
  "M&A", "fraud/accounting", "macro/other", "unclear"],
 "est_impact_usd": {{"low": number, "mid": number, "high": number}},
 "impact_basis": "one sentence: what the dollar estimate is based on",
 "verdict": "justified" | "not justified" | "unsure",
 "market_read": "one sentence: what the market evidence says",
 "decision": "take" | "skip",
 "key_risk": "one sentence: the most likely way the trade loses",
 "reason": "at most two sentences",
 "confidence": number between 0 and 1}}"""


def _pct(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:+.1f}%"


def context_block(price, peers, earn):
    """[AG-07] the market evidence as plain lines for the prompt (rounded so it can't fingerprint the company)"""
    lines = []
    if price:
        lines += [f"- Return over the 5 / 21 / 63 trading days BEFORE Day 0: "
                  f"{_pct(price['ret_5d'])} / {_pct(price['ret_21d'])} / {_pct(price['ret_63d'])}",
                  f"- Before Day 0 the stock was {_pct(price['off_52w_high'])} from its 52-week high and "
                  f"{_pct(price['close_vs_200dma'])} vs its 200-day average",
                  f"- Annualized volatility over the prior 60 days: {price['rv_60d'] * 100:.0f}%"]
        if price.get("volume_x") is not None:
            lines.append(f"- Day 0 volume: {price['volume_x']:.1f}x its 20-day average")
    else:
        lines.append("- Price history: not available")
    if peers:
        lines.append(f"- Same-{peers['level'].replace('_', ' ')} stocks on Day 0 ({peers['n']}): median "
                     f"{_pct(peers['median'])}, {peers['share_down_3pct'] * 100:.0f}% of them fell 3% or more")
    else:
        lines.append("- Industry peers on Day 0: not available")
    if earn:
        ds = earn["days_since_last"]
        lines.append(f"- Last earnings report: {'unknown' if ds is None else f'{ds} days before Day 0'}; "
                     f"next report within 5 weeks: {'yes' if earn['next_in_window'] else 'no'}")
    return "\n".join(lines)


def build_agent_prompt(heads, stories, aliases, sector, industry, market_cap, abnormal, price, peers, earn,
                       valuation_lines=None):
    """
    [AG-08] LLM-03's anonymized news prompt with the market evidence and the decision added
    returns (messages, facts) like LLM-03
    """
    msgs, facts = build_event_prompt(heads, stories, aliases, sector, industry, market_cap, abnormal)
    user = msgs[1]["content"]
    i = user.rfind("Return JSON:")
    body = user[:i].rstrip() if i >= 0 else user.rstrip()
    ctx = context_block(price, peers, earn)
    if valuation_lines:                     # only when fundamentals exist, so old prompts (and the cache) stay identical
        ctx += "\n" + "\n".join(valuation_lines)
    msgs[1] = {"role": "user", "content": body + "\n" + DECISION_SPEC.format(context=ctx)}
    return msgs, facts


def parse_agent(content):
    """[AG-09] LLM-05 plus decision / market_read / key_risk; anything malformed -> decision "skip" """
    v = parse_verdict(content)
    extra = {"decision": "skip", "market_read": "", "key_risk": ""}
    try:
        d = json.loads(re.sub(r"^```(json)?|```$", "", content.strip()).strip())
        dec = str(d.get("decision", "skip")).strip().lower()
        extra = {"decision": dec if dec in ("take", "skip") else "skip",
                 "market_read": str(d.get("market_read", "")), "key_risk": str(d.get("key_risk", ""))}
    except (ValueError, TypeError, AttributeError):
        pass
    return {**v, **extra}


def final_call(parsed, abnormal, market_cap, earn, threshold=3.0, fv_fail=None):
    """
    [AG-10] the agent's trade flag. take only if ALL hold:
      the model said take, the answer parsed, its verdict is "not justified", the category may be
      traded (STR-2.3a), and no earnings report falls inside the trade window (STR-2.2a)
    The overreaction ratio (STR-2.4a) is reported but not required: the model already weighed it,
    and the backtest can test ratio >= threshold as a separate filter.
    returns dict: agent_take, tradeable, ratio_mid, why_not
    """
    cat, tradeable = news_category(parsed["category"])
    ratio = overreaction_ratio(abnormal, market_cap, parsed["impact_mid"])
    checks = [("model said skip", parsed["decision"] == "take"),
              ("unparseable answer", parsed["parse_ok"]),
              (f"verdict {parsed['verdict']}", parsed["verdict"] == "not justified"),
              (f"category {cat} not traded", tradeable),
              ("earnings inside trade window", not (earn and earn["next_in_window"])),
              (f"fair-value gate: {fv_fail}", not fv_fail)]       # AG-15, only when the veto is switched on
    failed = [why for why, ok in checks if not ok]
    return {"agent_take": not failed, "category": cat, "tradeable": tradeable, "ratio_mid": ratio,
            "ratio_ok": bool(ratio is not None and ratio >= threshold), "why_not": "; ".join(failed)}
