"""
AG-00: the agent's LLM client and news rules, self-contained (the team's branch has no src/llm).

Copied from the earlier LLM filter (LLM-01..05) and strategy_logic (STR-2.3a / 2.4a) so the agent
does not depend on files that were removed:
  company_aliases / anonymize / build_event_prompt  anonymized news prompt (no names, tickers, dates)
  call_llm   any OpenAI-compatible endpoint (TAMU by default) with an on-disk cache; stream=False
             because TAMU (Open WebUI) streams SSE otherwise
  parse_verdict   tolerant JSON parser
  news_category / overreaction_ratio   STR-2.3a / STR-2.4a
Model and endpoint come from .env (AGENT_* or LLM_*), so switching models needs no code change.
"""
import hashlib
import math
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

SYSTEM = ("You are an equity analyst. You judge whether a stock's one-day drop is justified by the news. "
          "You only know what is in the message. Answer in JSON only, with exactly the keys requested.")

USER_TEMPLATE = """Company: a {sector} company ({industry}), market value about ${cap_bn}B before the news.
On Day 0 the stock fell {abn_pct}% more than the market, about ${lost_bn}B of market value.

News published up to the close of Day 0 (oldest first):
{headlines}

Story excerpts:
{stories}

Return JSON:
{{"category": one of ["analyst downgrade", "peer/sector contagion", "lawsuit", "regulatory probe",
  "short seller report", "product recall", "dilution/offering", "guidance cut", "management change",
  "M&A", "fraud/accounting", "macro/other", "unclear"],
 "est_impact_usd": {{"low": number, "mid": number, "high": number}},
 "impact_basis": "one sentence: what the dollar estimate is based on",
 "verdict": "justified" | "not justified" | "unsure",
 "reason": "at most two sentences",
 "confidence": number between 0 and 1}}"""

RECOGNITION_Q = ("\n\nForget the task above. Which company is this, and what did its stock do over the "
                 "following month? If you are not sure, say so. Return JSON: "
                 '{"company": string or null, "ticker": string or null, "next_month": string, "confidence": number}')

LEGAL_SUFFIXES = r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings?|group|sa|nv|ag|se|lp|llc|the|class [a-z]|cl [a-z])\b\.?"
MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december|"
          "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"


def company_aliases(name, ticker=None, ric=None, root=None, extra=()):
    """
    [AG-0L2] (was LLM-02) strings that identify the company, to scrub from prompts

    name "Apple Inc" -> ["Apple Inc", "Apple"]; "Exxon Mobil Corp" -> ["Exxon Mobil Corp", "Exxon Mobil",
    "ExxonMobil", "Exxon"]; plus ticker ("AAPL"), RIC ("AAPL.OQ"), RIC base, option root, extra
    single letters / 1-char tickers are skipped (they would wipe normal words)
    """
    out = set()
    if name and isinstance(name, str):
        out.add(name.strip())
        core = re.sub(LEGAL_SUFFIXES, " ", name, flags=re.I)
        core = re.sub(r"[,.&]", " ", core)
        core = re.sub(r"\s+", " ", core).strip()
        if core:
            out.add(core)
            out.add(core.replace(" ", ""))
            first = core.split(" ")[0]
            if len(first) >= 4:
                out.add(first)
        words = re.sub(r"[,.]", " ", name).split()
        for k in range(len(words), 0, -1):          # "Testco Holdings Inc" -> "Testco Holdings", "Testco"
            pre = " ".join(words[:k])
            if len(pre) >= 4:
                out.add(pre)
    for x in (ticker, ric, root, *(extra or ())):
        if x and isinstance(x, str):
            out.add(x)
            out.add(x.split(".")[0])
    return sorted((a for a in out if len(a) >= 2), key=len, reverse=True)


def anonymize(text, aliases):
    """
    [AG-0L1] (was LLM-01) remove what could identify the company or the date

    aliases (LLM-02) -> "the Company" (whole words, case-insensitive; tickers also inside $AAPL, (AAPL.O));
    years 1990-2039 -> removed, month names (+ day numbers) -> removed, weekdays -> "a weekday",
    numeric dates (2025-03-14, 3/14/2025, 14-Mar) -> removed. Dollar amounts and percentages stay.
    """
    if not isinstance(text, str) or not text:
        return ""
    t = text
    for a in aliases:
        t = re.sub(rf"(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])", "the Company", t, flags=re.I)
    t = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", t)
    t = re.sub(r"\b\d{1,2}/\d{1,2}(/\d{2,4})?\b", "", t)
    t = re.sub(rf"\b\d{{1,2}}[- ]({MONTHS})\b\.?", "", t, flags=re.I)
    t = re.sub(rf"\b({MONTHS})\b\.?(\s+\d{{1,2}}(st|nd|rd|th)?\b,?)?", "", t, flags=re.I)
    t = re.sub(r"\b(19[9]\d|20[0-3]\d)\b", "", t)
    t = re.sub(rf"\b({WEEKDAYS})\b", "a weekday", t, flags=re.I)
    t = re.sub(LEGAL_SUFFIXES.replace(r"\b(", r"(?<=the Company)\s+(", 1), "", t, flags=re.I)
    t = re.sub(r"(the Company[’']?s?\s*){2,}", "the Company ", t)
    return re.sub(r"[ \t]{2,}", " ", t).strip()


def _round_to(x, step):
    return step * round(x / step)


def build_event_prompt(headlines, stories, aliases, sector, industry, market_cap, abnormal,
                       max_headlines=30, max_story_chars=2000):
    """
    [AG-0L3] (was LLM-03) chat messages for one event, anonymized (spec section 5 of docs/llm_filter_spec.md)

    headlines: DataFrame (event_time UTC, headline), ALREADY cut to before the shock-day close (DAT-09)
    stories: DataFrame (event_time, text) of up to 2 story texts, same cut
    market_cap: pre-shock $; abnormal: shock-day abnormal return (negative)
    returns (messages, facts) where facts holds the rounded numbers that went into the prompt
    """
    cap_bn = max(5, int(_round_to(market_cap / 1e9, 5))) if market_cap and np.isfinite(market_cap) else None
    lost = -abnormal * market_cap if market_cap and np.isfinite(market_cap) else np.nan
    lost_bn = float(f"{lost / 1e9:.2g}") if np.isfinite(lost) and lost > 0 else None
    abn_pct = _round_to(-abnormal * 100, 0.5)
    h = headlines.sort_values("event_time").head(max_headlines) if len(headlines) else headlines
    hl = "\n".join(f"- {anonymize(x, aliases)}" for x in h["headline"]) if len(h) else "- (no Reuters headline)"
    st = [anonymize(str(x)[:max_story_chars], aliases) for x in (stories["text"] if len(stories) else [])]
    st = "\n\n".join(f"[{i + 1}] {x}" for i, x in enumerate(st) if x) or "(none)"
    user = USER_TEMPLATE.format(sector=sector or "US-listed", industry=industry or "industry not given",
                                cap_bn=cap_bn if cap_bn is not None else "unknown", abn_pct=abn_pct,
                                lost_bn=lost_bn if lost_bn is not None else "unknown",
                                headlines=hl, stories=st)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return messages, {"cap_bn": cap_bn, "lost_bn": lost_bn, "abn_pct": abn_pct, "n_headlines": len(h)}


TAMU_BASE_URL = "https://chat-api.tamu.ai/openai"


def call_llm(messages, cache_dir, model, api_key=None, base_url=TAMU_BASE_URL,
             temperature=0.0, max_tokens=800, json_mode=True, session=None, retries=5, timeout=120):
    """
    [AG-0L4] (was LLM-04) one chat completion (OpenAI-compatible /chat/completions) with an on-disk cache
    (sha256 of base_url + model + messages + settings)

    cached answers are returned without calling the API, so re-runs are free and identical.
    temperature 0; on DeepSeek thinking mode is also disabled (it ignores temperature).
    If the server rejects response_format (HTTP 400), retries once without it (the prompt asks for JSON anyway).
    Retries 429 / 5xx with backoff.
    returns dict: content, model (version string the server reports), usage, cached (bool), key
    """
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if "deepseek" in base_url:
        payload["thinking"] = {"type": "disabled"}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    key = hashlib.sha256(json.dumps({**payload, "base_url": base_url}, sort_keys=True).encode()).hexdigest()
    payload["stream"] = False       # TAMU (Open WebUI) streams SSE by default; kept out of the cache key so old caches still hit
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        return {**saved["result"], "cached": True, "key": key}
    if not api_key:
        raise RuntimeError("LLM_API_KEY missing: put it in .env (see .env.example)")
    import requests
    s = session or requests.Session()
    url = base_url.rstrip("/") + "/chat/completions"
    for attempt in range(retries):
        r = s.post(url, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
        if r.status_code == 200:
            break
        if r.status_code == 400 and "response_format" in payload and "response_format" in r.text:
            payload = {k: v for k, v in payload.items() if k != "response_format"}
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(min(60, 2 ** attempt * 3))
            continue
        raise RuntimeError(f"LLM API HTTP {r.status_code}: {r.text[:300]}")
    body = r.json()
    result = {"content": body["choices"][0]["message"]["content"], "model": body.get("model"),
              "usage": body.get("usage", {})}
    path.write_text(json.dumps({"request": payload, "result": result,
                                "saved": pd.Timestamp.now(tz="UTC").isoformat()}, indent=1), encoding="utf-8")
    return {**result, "cached": False, "key": key}


def parse_verdict(content):
    """
    [AG-0L5] (was LLM-05) parse and check the model's JSON answer

    returns dict: category, impact_low / mid / high ($, floats), verdict ("justified" / "not justified" /
    "unsure"), reason, impact_basis, confidence, parse_ok. Anything malformed -> parse_ok False,
    verdict "unsure" (an unparseable answer is never traded)
    """
    out = {"category": "unclear", "impact_low": np.nan, "impact_mid": np.nan, "impact_high": np.nan,
           "verdict": "unsure", "reason": "", "impact_basis": "", "confidence": np.nan, "parse_ok": False}
    try:
        txt = content.strip()
        txt = re.sub(r"^```(json)?|```$", "", txt).strip()
        d = json.loads(txt)
        imp = d.get("est_impact_usd") or {}
        out.update({"category": str(d.get("category", "unclear")),
                    "impact_low": float(imp.get("low")), "impact_mid": float(imp.get("mid")),
                    "impact_high": float(imp.get("high")),
                    "verdict": str(d.get("verdict", "unsure")).strip().lower(),
                    "reason": str(d.get("reason", "")), "impact_basis": str(d.get("impact_basis", "")),
                    "confidence": float(d.get("confidence", np.nan))})
        if out["verdict"] not in ("justified", "not justified", "unsure"):
            out["verdict"] = "unsure"
            return out
        out["parse_ok"] = True
    except (ValueError, TypeError, AttributeError):
        pass
    return out


NEWS_CATEGORIES = ("analyst downgrade", "peer/sector contagion", "lawsuit", "regulatory probe",
                   "short seller report", "product recall", "dilution/offering", "guidance cut",
                   "management change", "M&A", "fraud/accounting", "macro/other", "unclear")
# spec 1.2 exclusions (binary / the news IS the fundamental) and 2.3 "other / unclassified: not traded at first"
NOT_TRADED = ("guidance cut", "management change", "M&A", "fraud/accounting", "macro/other", "unclear")


def news_category(category, enabled=None):
    """
    [AG-0R1] (STR-2.3a rule) the LLM's news category and whether that category may be traded

    category: one of NEWS_CATEGORIES (anything else counts as "unclear")
    enabled: optional collection of categories switched ON (spec 4 toggles: each alone vs all);
        None = every category except NOT_TRADED
    returns (category, tradeable)
    """
    cat = category if category in NEWS_CATEGORIES else "unclear"
    if enabled is not None:
        return cat, cat in set(enabled)
    return cat, cat not in NOT_TRADED


def overreaction_ratio(abnormal_ret, market_cap, est_impact):
    """
    [AG-0R2] (STR-2.4a rule) market cap lost / LLM's estimated $ hit to company value

    abnormal_ret: shock-day abnormal return (negative, STR-2.1b); market_cap: pre-shock $
    est_impact: LLM estimate in $ (low / mid / high: call once per estimate)
    ex: -10% on a $20B company = $2B lost; impact $10M -> 200x
    est_impact <= 0 (news worth nothing) -> inf; returns None if any input is missing or the move isn't a drop
    """
    if abnormal_ret is None or market_cap is None or est_impact is None:
        return None
    if any(isinstance(x, float) and math.isnan(x) for x in (abnormal_ret, market_cap, est_impact)):
        return None
    lost = -abnormal_ret * market_cap
    if lost <= 0:
        return None
    if est_impact <= 0:
        return math.inf
    return lost / est_impact
