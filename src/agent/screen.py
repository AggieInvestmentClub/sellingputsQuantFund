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

from src.agent.llm import build_event_prompt, extract_json, news_category, norm_verdict, overreaction_ratio, parse_verdict

AGENT_VERSION = "ag-v1"          # default: keeps the cached v1 answers comparable (data/results/agent/agent_results.csv)
AGENT_VERSION_V2 = "ag-v2"       # [AG-50] structured XML reasoning + strict anonymization; select with version="ag-v2"
VERSIONS = (AGENT_VERSION, AGENT_VERSION_V2)

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
                       valuation_lines=None, version=AGENT_VERSION, peer_names=(), extra_names=()):
    """
    [AG-08] LLM-03's anonymized news prompt with the market evidence and the decision added
    returns (messages, facts) like LLM-03. version="ag-v2" -> build_agent_prompt_v2 (AG-50).
    """
    if version == AGENT_VERSION_V2:
        return build_agent_prompt_v2(heads, stories, aliases, sector, industry, market_cap, abnormal, price, peers,
                                     earn, valuation_lines, peer_names, extra_names)
    if version != AGENT_VERSION:
        raise ValueError(f"unknown agent version {version}; one of {VERSIONS}")
    msgs, facts = build_event_prompt(heads, stories, aliases, sector, industry, market_cap, abnormal)
    user = msgs[1]["content"]
    i = user.rfind("Return JSON:")
    body = user[:i].rstrip() if i >= 0 else user.rstrip()
    ctx = context_block(price, peers, earn)
    if valuation_lines:                     # only when fundamentals exist, so old prompts (and the cache) stay identical
        ctx += "\n" + "\n".join(valuation_lines)
    msgs[1] = {"role": "user", "content": body + "\n" + DECISION_SPEC.format(context=ctx)}
    return msgs, facts


def parse_agent(content, version=AGENT_VERSION, market_cap=None):
    """[AG-09] LLM-05 plus decision / market_read / key_risk; anything malformed -> decision "skip".
    [AG-47] JSON found with extract_json (fences, text around it, truncation); content None (LLM unavailable)
    -> verdict "unavailable", decision "skip". version="ag-v2" -> parse_agent_v2."""
    if version == AGENT_VERSION_V2:
        return parse_agent_v2(content, market_cap)
    v = parse_verdict(content)
    extra = {"decision": "skip", "market_read": "", "key_risk": ""}
    d = extract_json(content) if content is not None else None
    if d is not None:
        dec = re.sub(r"[^a-z]", "", str(d.get("decision", "skip")).lower())
        extra = {"decision": dec if dec in ("take", "skip") else "skip",
                 "market_read": str(d.get("market_read", "")), "key_risk": str(d.get("key_risk", ""))}
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
    if parsed.get("verdict") == "unavailable":                  # AG-46: endpoint down -> never a TAKE, say why
        checks.insert(0, ("LLM unavailable", False))
    failed = [why for why, ok in checks if not ok]
    return {"agent_take": not failed, "category": cat, "tradeable": tradeable, "ratio_mid": ratio,
            "ratio_ok": bool(ratio is not None and ratio >= threshold), "why_not": "; ".join(failed)}


# ---------------------------------------------------------------- [AG-50] agent v2: structured reasoning
DECISION_SPEC_V2 = """
Market evidence (computed from prices up to the Day 0 close; numbers only):
{context}

Volatility context:
{vol}

Your job is to decide whether to SELL A PUT SPREAD with the short strike well below the Day 0 low, held 2-5 weeks.
That trade wins if the stock does not keep falling. Selling puts on random days does about as well as selling after
drops, so only say "take" when the drop is clearly bigger than the news justifies AND the selling is likely done.
Most drops are NOT clear overreactions: "take" should be the minority answer.

Reason in four short steps, each inside its tag (at most three sentences per tag):
<impairment>Does the news permanently reduce future cash flows? Say none, minor, material or severe, and why.</impairment>
<shock_type>Exogenous (macro, sector, analyst or rating action, peer contagion, one-off event) or endogenous (the
company's own guidance, accounting, liquidity, management, competitive position)? Say which and why.</shock_type>
<vol_context>Is the Day 0 move large for this stock's normal volatility, and is volatility likely to stay high?</vol_context>
<overreaction>Estimate the hit to company value as % of market value (low / mid / high) and compare it with the drop.
Give a calibrated confidence: 0.5 = coin flip; 0.8 or more only when the evidence is strong and consistent.</overreaction>

Then give the final answer as JSON inside <answer></answer>:
<answer>{{"category": one of ["analyst downgrade", "peer/sector contagion", "lawsuit", "regulatory probe",
  "short seller report", "product recall", "dilution/offering", "guidance cut", "management change",
  "M&A", "fraud/accounting", "macro/other", "unclear"],
 "impairment": "none" | "minor" | "material" | "severe",
 "shock_type": "exogenous" | "endogenous" | "mixed",
 "est_impact_pct": {{"low": number, "mid": number, "high": number}},
 "verdict": "justified" | "not justified" | "unsure",
 "market_read": "one sentence", "decision": "take" | "skip", "key_risk": "one sentence",
 "reason": "at most two sentences", "confidence": number between 0 and 1}}</answer>"""

SECTIONS = ("impairment", "shock_type", "vol_context", "overreaction")


def vol_lines(price, extra=None):
    """[AG-50] the move in standard deviations of the stock's own prior 60-day volatility, plus any extra lines"""
    out = []
    if price and price.get("rv_60d") and np.isfinite(price["rv_60d"]) and price["rv_60d"] > 0:
        daily = price["rv_60d"] / np.sqrt(252)
        out.append(f"- Day 0 move: {price['ret_shock'] * 100:+.1f}% = {price['ret_shock'] / daily:+.1f} daily "
                   f"standard deviations of the prior 60 days (annualized vol {price['rv_60d'] * 100:.0f}%)")
    else:
        out.append("- Prior volatility: not available")
    out.append("- Option-implied volatility for this stock: not available (not in the evidence)")
    return "\n".join(out + list(extra or []))


def build_agent_prompt_v2(heads, stories, aliases, sector, industry, market_cap, abnormal, price, peers, earn,
                          valuation_lines=None, peer_names=(), extra_names=(), vol_extra=None):
    """
    [AG-50] v2 prompt: strict anonymization (AG-48: case-safe tickers, peer and executive names, $ buckets,
    size band instead of $B), the same market evidence block (AG-07), a volatility block, and the four
    structured reasoning sections (<impairment>, <shock_type>, <vol_context>, <overreaction>) before a JSON
    <answer>. Call the LLM with llm_kwargs("ag-v2") (plain text, not JSON mode). returns (messages, facts)
    """
    msgs, facts = build_event_prompt(heads, stories, aliases, sector, industry, market_cap, abnormal, strict=True,
                                     peer_names=peer_names, extra_names=extra_names)
    user = msgs[1]["content"]
    i = user.rfind("Return JSON:")
    body = user[:i].rstrip() if i >= 0 else user.rstrip()
    ctx = context_block(price, peers, earn)
    if valuation_lines:
        ctx += "\n" + "\n".join(valuation_lines)
    msgs[1] = {"role": "user", "content": body + "\n" + DECISION_SPEC_V2.format(context=ctx, vol=vol_lines(price, vol_extra))}
    msgs[0] = {"role": "system", "content": msgs[0]["content"].replace(
        "Answer in JSON only, with exactly the keys requested.",
        "Write the four tagged reasoning sections, then the JSON answer inside <answer></answer>.")}
    return msgs, {**facts, "version": AGENT_VERSION_V2}


def llm_kwargs(version=AGENT_VERSION):
    """[AG-50] call_llm settings per version: v2 answers in tagged text + JSON, so JSON mode is off"""
    return {"max_tokens": 900} if version == AGENT_VERSION else {"max_tokens": 1500, "json_mode": False}


def parse_agent_v2(content, market_cap=None):
    """
    [AG-50] parse a v2 answer. Tolerant: any section may be missing (sections_found lists the ones present), the
    JSON may sit outside <answer> or be cut off; est_impact_pct (% of market value) -> impact_low/mid/high in $
    when market_cap is known (so the overreaction ratio works as in v1); est_impact_usd is accepted too.
    Malformed -> parse_ok False, decision skip. content None -> verdict "unavailable".
    """
    out = {"category": "unclear", "impact_low": np.nan, "impact_mid": np.nan, "impact_high": np.nan,
           "verdict": "unsure", "reason": "", "impact_basis": "", "confidence": np.nan, "parse_ok": False,
           "decision": "skip", "market_read": "", "key_risk": "", "impairment": "", "shock_type": "",
           "sections_found": [], "version": AGENT_VERSION_V2}
    if content is None:
        return {**out, "verdict": "unavailable", "reason": "LLM unavailable"}
    for sec in SECTIONS:
        m = re.search(rf"<{sec}>(.*?)(</{sec}>|<(?:{'|'.join(SECTIONS)}|answer)>|$)", content, flags=re.S | re.I)
        if m and m.group(1).strip():
            out[f"text_{sec}"] = m.group(1).strip()[:600]
            out["sections_found"].append(sec)
    d = extract_json(content)
    if d is None:
        return out
    v = norm_verdict(d.get("verdict", "unsure"))

    def num(x):
        try:
            f = float(x)
            return f if np.isfinite(f) else np.nan
        except (TypeError, ValueError):
            return np.nan
    pct = d.get("est_impact_pct")
    usd = d.get("est_impact_usd")
    if isinstance(pct, dict) and market_cap and np.isfinite(market_cap):
        for k in ("low", "mid", "high"):
            out[f"impact_{k}"] = num(pct.get(k)) / 100 * market_cap
        out["impact_basis"] = f"est_impact_pct {pct}"
    elif isinstance(usd, dict):
        for k in ("low", "mid", "high"):
            out[f"impact_{k}"] = num(usd.get(k))
    dec = re.sub(r"[^a-z]", "", str(d.get("decision", "skip")).lower())
    imp = str(d.get("impairment", "")).strip().lower()
    st = str(d.get("shock_type", "")).strip().lower()
    out.update({"category": str(d.get("category", "unclear")),
                "verdict": v if v in ("justified", "not justified", "unsure") else "unsure",
                "reason": str(d.get("reason", "")), "confidence": num(d.get("confidence")),
                "decision": dec if dec in ("take", "skip") else "skip",
                "market_read": str(d.get("market_read", "")), "key_risk": str(d.get("key_risk", "")),
                "impairment": imp if imp in ("none", "minor", "material", "severe") else "",
                "shock_type": st if st in ("exogenous", "endogenous", "mixed") else "",
                "parse_ok": v in ("justified", "not justified", "unsure")})
    return out
