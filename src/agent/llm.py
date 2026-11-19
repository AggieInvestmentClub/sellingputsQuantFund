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
from functools import lru_cache
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

USER_TEMPLATE_STRICT = """Company: a {size} {sector} company ({industry}).
On Day 0 the stock fell {abn_pct}% more than the market.

News published up to the close of Day 0 (oldest first; names, tickers, dates and exact dollar figures removed):
{headlines}

Story excerpts:
{stories}

Return JSON:"""

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


def anonymize(text, aliases, strict=False, market_cap=None, extra_names=(), peer_names=()):
    """
    [AG-0L1] (was LLM-01) remove what could identify the company or the date

    aliases (LLM-02) -> "the Company" (whole words, case-insensitive; tickers also inside $AAPL, (AAPL.O));
    years 1990-2039 -> removed, month names (+ day numbers) -> removed, weekdays -> "a weekday",
    numeric dates (2025-03-14, 3/14/2025, 14-Mar) -> removed. Dollar amounts and percentages stay.
    """
    if not isinstance(text, str) or not text:
        return ""
    if strict:
        return anonymize_strict(text, aliases, market_cap, extra_names, peer_names)
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


# ---------------------------------------------------------------- [AG-48] strict anonymization (agent v2)
_TITLE = (r"(?i:chief\s+(?:executive|financial|operating|technology|investment|commercial|medical|scientific)"
          r"(?:\s+officer)?|ceo|cfo|coo|cto|cio|executive\s+chairman|chairman|chairwoman|chairperson|chair|"
          r"president|co-founder|founder|general\s+counsel|analyst|head)")
_NAME = r"(?:[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?|[A-Z]\.)(?:\s+(?:[A-Z]\.|[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)){1,2}"
_NOT_NAME = {"The", "Company", "Inc", "Corp", "Reuters", "Wall", "Street", "Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Officer", "Executive", "Financial", "Operating", "Board", "Group", "Said",
             "And", "Of", "On", "In", "For", "New", "York", "Exchange", "Securities"}
_RIC = r"\(?\b[A-Z]{1,5}\.(?:N|O|OQ|K|P|A|L|TO|DE)\b\)?"
_DOLLAR = (r"(?:US)?\$\s?(\d[\d,]*(?:\.\d+)?)\s*(trillion|tln|tn|billion|bln|bn|b|million|mln|mn|m|thousand|k)?\b"
           r"(\s*(?:a|per)\s+share)?")
_UNIT = {"trillion": 1e12, "tln": 1e12, "tn": 1e12, "billion": 1e9, "bln": 1e9, "bn": 1e9, "b": 1e9,
         "million": 1e6, "mln": 1e6, "mn": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}


def cap_bucket(market_cap):
    """market value as a size band (an exact $B figure plus a date can fingerprint the company)"""
    if market_cap is None or not np.isfinite(market_cap):
        return "unknown size"
    b = market_cap / 1e9
    for lim, lab in ((5, "small (under $5B)"), (20, "mid-size ($5-20B)"), (100, "large ($20-100B)"),
                     (500, "very large ($100-500B)")):
        if b < lim:
            return lab
    return "mega-cap (over $500B)"


def bucket_dollars(text, market_cap=None):
    """
    [AG-48] replace idiosyncratic dollar figures (a $4.2 billion charge, a $123.45 price, $2.15 a share) with
    scaled buckets; percentages are kept. "$4.2 billion" -> "[$ amount, ~1-5% of market value]" when the market
    cap is known, else "[$ billions]"; per-share amounts -> "[per-share $ amount]"; bare prices -> "[$ figure]".
    """
    def rep(m):
        num, unit, per_share = m.group(1), (m.group(2) or "").lower(), m.group(3)
        if per_share:
            return "[per-share $ amount]"
        try:
            v = float(num.replace(",", ""))
        except ValueError:
            return "[$ figure]"
        if not unit:
            return "[$ figure]"
        v *= _UNIT[unit]
        if market_cap and np.isfinite(market_cap) and market_cap > 0:
            pct = v / market_cap * 100
            for lim, lab in ((1, "under 1%"), (5, "~1-5%"), (20, "~5-20%"), (50, "~20-50%")):
                if pct < lim:
                    return f"[$ amount, {lab} of market value]"
            return "[$ amount, over half of market value]"
        return "[$ billions]" if v >= 1e9 else "[$ millions]" if v >= 1e6 else "[$ amount]"
    return re.sub(_DOLLAR, rep, text, flags=re.I)


def scrub_people(text):
    """[AG-48] executive / analyst names next to a title -> "[name]" ("CEO Jane Doe", "Jane Doe, chief executive")"""
    def keep_if_not_name(m, repl):
        name = m.group("name")
        if any(w.strip(".,") in _NOT_NAME for w in name.split()):
            return m.group(0)
        return repl(m)
    t = re.sub(rf"\b(?P<title>{_TITLE})(?P<sep>(?:\s+of\s+the\s+Company)?,?\s+)(?P<name>{_NAME})",
               lambda m: keep_if_not_name(m, lambda m: f"{m.group('title')}{m.group('sep')}[name]"), text)
    t = re.sub(rf"(?P<name>{_NAME}),\s+(?P<rest>(?:(?:the|its|a)\s+)?(?:Company's\s+|company's\s+)?(?:former\s+|new\s+|interim\s+)?{_TITLE})",
               lambda m: keep_if_not_name(m, lambda m: f"[name], {m.group('rest')}"), t)
    return t


@lru_cache(maxsize=8)
def _names_regex(names):
    if not names:
        return None
    alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z0-9])(?:{alt})(?![A-Za-z0-9])")


def anonymize_strict(text, aliases, market_cap=None, extra_names=(), peer_names=()):
    """
    [AG-48] agent v2 anonymization, stricter than LLM-01:
      - tickers / RICs (short all-caps aliases) are matched CASE-SENSITIVELY, so a ticker such as ON, IT, ALL,
        NOW or KEY no longer wipes ordinary words (LLM-01 matched every alias case-insensitively)
      - one-word name aliases ("General", "American") are matched case-sensitively as proper nouns
      - any other RIC in the text (a peer's "(XYZ.N)") -> "[ticker]"
      - executive / analyst names next to a title -> "[name]"; extra_names (products, brands, people from
        reference data when we have it) -> "[name]"
      - dollar figures -> scaled buckets (bucket_dollars); dates removed as in LLM-01; percentages kept
    """
    t = text
    for a in sorted(set(aliases), key=len, reverse=True):
        ticker_like = a.isupper() and len(a.replace(".", "")) <= 6
        one_word = " " not in a.strip()
        flags = 0 if (ticker_like or one_word) else re.I
        t = re.sub(rf"(?<![A-Za-z0-9$]){re.escape(a)}(?![A-Za-z0-9])", "the Company", t, flags=flags)
        if ticker_like:
            t = re.sub(rf"\${re.escape(a)}\b", "the Company", t)
    t = re.sub(_RIC, "[ticker]", t)
    rx = _names_regex(tuple(sorted(set(peer_names))))       # other listed companies (peers, customers) -> generic
    if rx is not None:
        t = rx.sub("[another company]", t)
    for n in sorted({x for x in extra_names if isinstance(x, str) and len(x) >= 3}, key=len, reverse=True):
        t = re.sub(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", "[name]", t)
    t = scrub_people(t)
    t = bucket_dollars(t, market_cap)
    t = anonymize(t, [])                                     # LLM-01 date / weekday rules, no aliases left
    return t


def _round_to(x, step):
    return step * round(x / step)


def build_event_prompt(headlines, stories, aliases, sector, industry, market_cap, abnormal,
                       max_headlines=30, max_story_chars=2000, strict=False, extra_names=(), peer_names=()):
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
    kw = ({"strict": True, "market_cap": market_cap, "extra_names": tuple(extra_names),
           "peer_names": tuple(peer_names)} if strict else {})
    hl = "\n".join(f"- {anonymize(x, aliases, **kw)}" for x in h["headline"]) if len(h) else "- (no Reuters headline)"
    st = [anonymize(str(x)[:max_story_chars], aliases, **kw) for x in (stories["text"] if len(stories) else [])]
    st = "\n\n".join(f"[{i + 1}] {x}" for i, x in enumerate(st) if x) or "(none)"
    if strict:                                             # [AG-48] no exact $ sizes in the v2 prompt
        user = USER_TEMPLATE_STRICT.format(sector=sector or "US-listed", industry=industry or "industry not given",
                                           size=cap_bucket(market_cap), abn_pct=abn_pct, headlines=hl, stories=st)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        return messages, {"cap_bn": cap_bn, "lost_bn": lost_bn, "abn_pct": abn_pct, "n_headlines": len(h),
                          "size": cap_bucket(market_cap)}
    user = USER_TEMPLATE.format(sector=sector or "US-listed", industry=industry or "industry not given",
                                cap_bn=cap_bn if cap_bn is not None else "unknown", abn_pct=abn_pct,
                                lost_bn=lost_bn if lost_bn is not None else "unknown",
                                headlines=hl, stories=st)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return messages, {"cap_bn": cap_bn, "lost_bn": lost_bn, "abn_pct": abn_pct, "n_headlines": len(h)}


TAMU_BASE_URL = "https://chat-api.tamu.ai/openai"


class LLMUnavailable(RuntimeError):
    """the endpoint could not answer (network down, circuit open, retries exhausted): never a TAKE"""


class CircuitBreaker:
    """
    [AG-46] stops hammering a dead endpoint: after `threshold` consecutive failed calls it OPENS for `cooldown`
    seconds (every call fails fast with LLMUnavailable), then lets one trial call through (half-open).
    A success closes it again. `clock` is injectable for tests.
    """

    def __init__(self, threshold=4, cooldown=300.0, clock=time.monotonic):
        self.threshold, self.cooldown, self.clock = threshold, cooldown, clock
        self.failures, self.opened_at = 0, None

    @property
    def state(self):
        if self.opened_at is None:
            return "closed"
        return "half-open" if self.clock() - self.opened_at >= self.cooldown else "open"

    def allow(self):
        return self.state != "open"

    def success(self):
        self.failures, self.opened_at = 0, None

    def failure(self):
        self.failures += 1
        if self.failures >= self.threshold or self.state == "half-open":
            self.opened_at = self.clock()


BREAKER = CircuitBreaker()          # one per process: a whole run stops calling after 4 straight failures
RETRYABLE = (429, 500, 502, 503, 504)


def backoff_delay(attempt, base=2.0, cap=60.0, rng=None):
    """[AG-46] exponential backoff with full jitter: uniform(0.5, 1.5) x min(cap, base x 2^attempt) seconds"""
    import random
    r = (rng or random).uniform(0.5, 1.5)
    return min(cap, base * 2 ** attempt) * r


def call_llm(messages, cache_dir, model, api_key=None, base_url=TAMU_BASE_URL,
             temperature=0.0, max_tokens=800, json_mode=True, session=None, retries=5, timeout=120,
             breaker=None, sleep=time.sleep):
    """
    [AG-0L4] (was LLM-04) one chat completion (OpenAI-compatible /chat/completions) with an on-disk cache
    (sha256 of base_url + model + messages + settings)

    cached answers are returned without calling the API, so re-runs are free and identical (the cache is read
    even when the endpoint is down or the circuit is open).
    temperature 0; on DeepSeek thinking mode is also disabled (it ignores temperature).
    If the server rejects response_format (HTTP 400), retries once without it (the prompt asks for JSON anyway).
    [AG-46] Retries 429 / 5xx, connection errors, timeouts and unreadable (e.g. SSE) bodies with exponential
    backoff + jitter; a circuit breaker (default BREAKER) fails fast after repeated failures.
    Raises LLMUnavailable when the endpoint cannot answer; RuntimeError for non-retryable HTTP errors.
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
    breaker = BREAKER if breaker is None else breaker
    if not breaker.allow():
        raise LLMUnavailable(f"circuit open after {breaker.failures} consecutive failures; cooling down")
    import requests
    s = session or requests.Session()
    url = base_url.rstrip("/") + "/chat/completions"
    last, body = "no attempt", None
    for attempt in range(retries):
        try:
            r = s.post(url, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last = f"{type(e).__name__}: {str(e)[:200]}"
            if attempt < retries - 1:
                sleep(backoff_delay(attempt))
            continue
        if r.status_code == 200:
            try:
                body = r.json()
                content = body["choices"][0]["message"]["content"]
                break
            except (ValueError, KeyError, IndexError, TypeError) as e:
                body, last = None, f"unreadable 200 body ({type(e).__name__}): {r.text[:200]}"
                if attempt < retries - 1:
                    sleep(backoff_delay(attempt))
                continue
        if r.status_code == 400 and "response_format" in payload and "response_format" in r.text:
            payload = {k: v for k, v in payload.items() if k != "response_format"}
            continue
        if r.status_code in RETRYABLE:
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if attempt < retries - 1:
                sleep(backoff_delay(attempt))
            continue
        breaker.failure()
        raise RuntimeError(f"LLM API HTTP {r.status_code}: {r.text[:300]}")
    if body is None:
        breaker.failure()
        raise LLMUnavailable(f"no answer after {retries} attempts: {last}")
    breaker.success()
    result = {"content": content, "model": body.get("model"), "usage": body.get("usage", {})}
    path.write_text(json.dumps({"request": payload, "result": result,
                                "saved": pd.Timestamp.now(tz="UTC").isoformat()}, indent=1), encoding="utf-8")
    return {**result, "cached": False, "key": key}


FAIL_LOG = Path("data/results/agent/llm_failures.log")


def safe_call_llm(messages, cache_dir, model, api_key=None, base_url=TAMU_BASE_URL, log_path=FAIL_LOG,
                  context="", **kw):
    """
    [AG-46] call_llm that never raises: on any failure returns content None + unavailable True and appends one
    line to data/results/agent/llm_failures.log. parse_verdict(None) -> verdict "unavailable" -> never a TAKE.
    """
    try:
        return {**call_llm(messages, cache_dir, model, api_key, base_url, **kw), "unavailable": False, "error": ""}
    except Exception as e:                                   # noqa: BLE001 - offline must never crash a run
        err = f"{type(e).__name__}: {str(e)[:300]}"
        try:
            lp = Path(log_path)
            lp.parent.mkdir(parents=True, exist_ok=True)
            with lp.open("a", encoding="utf-8") as fh:
                fh.write(f"{pd.Timestamp.now(tz='UTC').isoformat()}\t{context}\t{err}\n")
        except OSError:
            pass
        return {"content": None, "model": None, "usage": {}, "cached": False, "key": None,
                "unavailable": True, "error": err}


def extract_json(content):
    """
    [AG-47] the JSON object inside a model answer, tolerant of: ```json fences, text before / after the object,
    an <answer>...</answer> wrapper, trailing commas, and an answer cut off mid-object (closes open strings,
    brackets and braces). Returns a dict, or None.
    """
    if not isinstance(content, str) or not content.strip():
        return None
    t = content.strip()
    m = re.search(r"<answer>(.*?)(</answer>|$)", t, flags=re.S | re.I)
    if m:
        t = m.group(1).strip()
    m = re.search(r"```(?:json)?\s*(.*?)(```|$)", t, flags=re.S | re.I)
    if m and "{" in m.group(1):
        t = m.group(1).strip()
    i = t.find("{")
    if i < 0:
        return None
    t = t[i:]
    for cand in (t, _close_json(t)):
        for c in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
            try:
                d = json.loads(c)
                return d if isinstance(d, dict) else None
            except ValueError:
                pass
    found = _first_object(t)
    if found is not _NO_OBJECT:
        return found
    if True:                                            # truncated: drop the last incomplete member, retry
        cut = t
        for _ in range(20):
            k = cut.rfind(",")
            if k <= 0:
                break
            cut = cut[:k]
            try:
                d = json.loads(_close_json(cut))
                return d if isinstance(d, dict) else None
            except ValueError:
                continue
    return None


_NO_OBJECT = object()


def _first_object(t):
    """the first balanced {...} in t (text may follow); _NO_OBJECT if the braces never balance"""
    depth, in_str, esc = 0, False, False
    for j, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    d = json.loads(re.sub(r",\s*([}\]])", r"\1", t[:j + 1]))
                    return d if isinstance(d, dict) else None
                except ValueError:
                    return None
    return _NO_OBJECT


def _close_json(t):
    """close an answer truncated mid-object: open string, then open [ and { in reverse order"""
    stack, in_str, esc = [], False, False
    for ch in t:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    out = t + ('"' if in_str else "")
    out = re.sub(r",\s*$", "", out.rstrip())
    out = re.sub(r':\s*$', ': null', out)
    return out + "".join(reversed(stack))


def norm_verdict(v):
    """'Not Justified.' / 'not_justified' / 'NOT-JUSTIFIED' -> 'not justified'; unknown -> None"""
    v = re.sub(r"[^a-z ]", " ", str(v).lower().replace("_", " ").replace("-", " "))
    v = re.sub(r"\s+", " ", v).strip()
    return v if v in ("justified", "not justified", "unsure", "unavailable") else None


def _num(x):
    try:
        f = float(x)
        return f if math.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def parse_verdict(content):
    """
    [AG-0L5] (was LLM-05) parse and check the model's JSON answer

    returns dict: category, impact_low / mid / high ($, floats), verdict ("justified" / "not justified" /
    "unsure"), reason, impact_basis, confidence, parse_ok. Anything malformed -> parse_ok False,
    verdict "unsure" (an unparseable answer is never traded).
    [AG-47] content None (endpoint down, see safe_call_llm) -> verdict "unavailable", parse_ok False.
    JSON is found with extract_json (fences, surrounding text, truncation); verdict spelling is normalized;
    a missing or non-numeric impact gives NaN instead of failing the whole answer.
    """
    out = {"category": "unclear", "impact_low": np.nan, "impact_mid": np.nan, "impact_high": np.nan,
           "verdict": "unsure", "reason": "", "impact_basis": "", "confidence": np.nan, "parse_ok": False}
    if content is None:
        return {**out, "verdict": "unavailable", "reason": "LLM unavailable"}
    d = extract_json(content)
    if d is None:
        return out
    imp = d.get("est_impact_usd") or {}
    if not isinstance(imp, dict):
        imp = {"mid": imp}
    v = norm_verdict(d.get("verdict", "unsure"))
    out.update({"category": str(d.get("category", "unclear")),
                "impact_low": _num(imp.get("low")), "impact_mid": _num(imp.get("mid")),
                "impact_high": _num(imp.get("high")),
                "verdict": v if v in ("justified", "not justified", "unsure") else "unsure",
                "reason": str(d.get("reason", "")), "impact_basis": str(d.get("impact_basis", "")),
                "confidence": _num(d.get("confidence"))})
    out["parse_ok"] = v in ("justified", "not justified", "unsure")
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
