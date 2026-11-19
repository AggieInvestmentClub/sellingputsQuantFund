"""Phase 4 agent upgrades (AG-46..AG-54): LLM resilience, robust parsing, strict anonymization, the v2 prompt,
safe resting take-profit, the ablation statistics and the macro classifier. No network: everything is mocked."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agent import execution as X  # noqa: E402
from src.agent import llm as L  # noqa: E402
from src.agent import macro as M  # noqa: E402
from src.agent import screen as S  # noqa: E402
from src.agent.evidence import event_news, peer_names  # noqa: E402


# ---------------------------------------------------------------- AG-46 call_llm resilience
class Resp:
    def __init__(self, code=200, body=None, text=None):
        self.status_code, self._body = code, body
        self.text = text if text is not None else json.dumps(body or {})

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class Session:
    def __init__(self, seq):
        self.seq, self.calls = list(seq), 0

    def post(self, *a, **k):
        self.calls += 1
        x = self.seq.pop(0) if self.seq else self.last
        self.last = x
        if isinstance(x, Exception):
            raise x
        return x


OK = Resp(200, {"choices": [{"message": {"content": '{"verdict": "unsure"}'}}], "model": "m"})
MSG = [{"role": "user", "content": "x"}]


def test_retries_connection_errors_then_succeeds(tmp_path):
    s = Session([requests.exceptions.ConnectionError("dns"), requests.exceptions.Timeout("slow"), OK])
    sleeps = []
    r = L.call_llm(MSG, tmp_path, "m", "k", session=s, breaker=L.CircuitBreaker(), sleep=sleeps.append)
    assert r["content"] == '{"verdict": "unsure"}' and s.calls == 3 and len(sleeps) == 2


def test_unreadable_200_body_is_retried(tmp_path):
    s = Session([Resp(200, None, text="data: {...}\n\n"), OK])
    r = L.call_llm(MSG, tmp_path, "m", "k", session=s, breaker=L.CircuitBreaker(), sleep=lambda x: None)
    assert r["model"] == "m" and s.calls == 2


def test_exhausted_retries_raise_unavailable_and_open_breaker(tmp_path):
    b = L.CircuitBreaker(threshold=2, cooldown=100, clock=lambda: 0.0)
    for _ in range(2):
        s = Session([requests.exceptions.ConnectionError("down")])
        with pytest.raises(L.LLMUnavailable):
            L.call_llm([{"role": "user", "content": str(_)}], tmp_path, "m", "k", session=s, breaker=b,
                       retries=3, sleep=lambda x: None)
    assert b.state == "open"
    s = Session([OK])
    with pytest.raises(L.LLMUnavailable, match="circuit open"):
        L.call_llm([{"role": "user", "content": "new"}], tmp_path, "m", "k", session=s, breaker=b)
    assert s.calls == 0                                     # failed fast, no request


def test_breaker_half_open_then_closes():
    t = {"now": 0.0}
    b = L.CircuitBreaker(threshold=1, cooldown=10, clock=lambda: t["now"])
    b.failure()
    assert b.state == "open" and not b.allow()
    t["now"] = 11
    assert b.state == "half-open" and b.allow()
    b.success()
    assert b.state == "closed"


def test_cache_served_even_when_breaker_open(tmp_path):
    L.call_llm(MSG, tmp_path, "m", "k", session=Session([OK]), breaker=L.CircuitBreaker())
    b = L.CircuitBreaker(threshold=1, clock=lambda: 0.0)
    b.failure()
    assert L.call_llm(MSG, tmp_path, "m", "k", session=Session([]), breaker=b)["cached"]


def test_backoff_has_jitter_and_cap():
    import random
    rng = random.Random(1)
    d = [L.backoff_delay(a, base=2, cap=60, rng=rng) for a in range(10)]
    assert all(0 < x <= 90 for x in d) and d[0] < 3.1 and len(set(d)) == len(d)


def test_safe_call_never_raises_and_logs(tmp_path):
    log = tmp_path / "fail.log"
    s = Session([requests.exceptions.ConnectionError("down")])
    r = L.safe_call_llm(MSG, tmp_path, "m", "k", log_path=log, context="AAA.N|2025-01-02", session=s,
                        breaker=L.CircuitBreaker(), retries=2, sleep=lambda x: None)
    assert r["unavailable"] and r["content"] is None and "AAA.N" in log.read_text()
    p = S.parse_agent(r["content"])
    call = S.final_call(p, -0.08, 2e10, {"next_in_window": False})
    assert p["verdict"] == "unavailable" and not call["agent_take"] and "LLM unavailable" in call["why_not"]


def test_non_retryable_http_raises_runtime(tmp_path):
    with pytest.raises(RuntimeError, match="HTTP 401"):
        L.call_llm(MSG, tmp_path, "m", "k", session=Session([Resp(401, {"e": 1})]), breaker=L.CircuitBreaker())


# ---------------------------------------------------------------- AG-47 parsing
GOOD = {"category": "lawsuit", "est_impact_usd": {"low": 1, "mid": 2, "high": 3}, "verdict": "not justified",
        "decision": "take", "reason": "r", "confidence": 0.8}


@pytest.mark.parametrize("wrap", [
    lambda j: j, lambda j: "```json\n" + j + "\n```", lambda j: "Here is my answer:\n" + j + "\nHope that helps.",
    lambda j: "Sure!\n```\n" + j + "\n```\nDone", lambda j: "<answer>" + j + "</answer>",
    lambda j: j[:-1] + ",}"])
def test_extract_json_variants(wrap):
    d = L.extract_json(wrap(json.dumps(GOOD)))
    assert d["verdict"] == "not justified" and d["decision"] == "take"


def test_escaped_quotes_and_trailing_text():
    d = L.extract_json('pre {"reason": "he said \\"no}\\" ok", "verdict": "unsure"} post {x}')
    assert d == {"reason": 'he said "no}" ok', "verdict": "unsure"}


def test_truncated_json_is_closed():
    t = json.dumps(GOOD)[:-25]
    d = L.extract_json(t)
    assert d is not None and d["category"] == "lawsuit"


def test_parse_tolerates_missing_impact_and_verdict_spelling():
    v = L.parse_verdict(json.dumps({"category": "lawsuit", "verdict": "Not_Justified."}))
    assert v["parse_ok"] and v["verdict"] == "not justified" and np.isnan(v["impact_mid"])
    assert not L.parse_verdict("no json here")["parse_ok"]
    assert L.parse_verdict(None)["verdict"] == "unavailable"


def test_parse_agent_with_preamble_takes():
    p = S.parse_agent("Thinking...\n" + json.dumps(GOOD) + "\nThanks")
    assert p["decision"] == "take" and p["parse_ok"] and p["verdict"] == "not justified"


# ---------------------------------------------------------------- AG-48 / AG-49 anonymization
def test_v1_unchanged_but_strict_keeps_ordinary_words():
    al = L.company_aliases("ON Semiconductor Corp", "ON", "ON.OQ")
    txt = "ON Semiconductor (ON.OQ) falls; analysts focus on outlook on Tuesday"
    v1 = L.anonymize(txt, al)
    assert "focus the Company outlook" in v1                 # the old bug, kept so v1 prompts / cache stay identical
    v2 = L.anonymize(txt, al, strict=True)
    assert "focus on outlook" in v2 and "Semiconductor" not in v2 and "ON" not in v2.replace("on", "")


def test_strict_scrubs_people_peers_rics_and_dollars():
    al = L.company_aliases("Testco Holdings Inc", "TST", "TST.N")
    txt = ("Testco CEO Jane Doe-Smith says $4.2 billion writedown; John Q. Public, the company's chief financial "
           "officer, resigns. Rival Wolfspeed (WOLF.N) down 3%. Shares at $45.20, EPS $1.05 per share, margin 12.5%.")
    out = L.anonymize(txt, al, strict=True, market_cap=100e9, peer_names=("Wolfspeed",))
    for leak in ("Testco", "TST", "Jane", "Doe-Smith", "John", "Wolfspeed", "WOLF", "4.2", "45.20", "1.05"):
        assert leak not in out, leak
    assert "[$ amount, ~1-5% of market value]" in out and "[per-share $ amount]" in out and "12.5%" in out
    assert "CEO [name]" in out and "[name], the company's chief financial" in out and "[another company]" in out


def test_dollar_buckets_without_market_cap():
    assert L.bucket_dollars("a $350 million charge and a $2bn deal") == "a [$ millions] charge and a [$ billions] deal"


def test_people_scrub_leaves_non_names():
    assert L.scrub_people("The CEO said The Company would grow") == "The CEO said The Company would grow"


def test_peer_names_from_master():
    m = pd.DataFrame({"ric": ["A.N", "B.N", "C.N"], "name": ["Wolfspeed Inc", "Target Corp", "Apple Inc"]})
    pn = peer_names(m, exclude_ric="C.N")
    assert "Wolfspeed" in pn and "Target" not in pn and "Apple" not in pn


def test_stories_after_close_are_dropped(tmp_path):
    p = tmp_path / "news" / "reuters_stories"
    p.mkdir(parents=True)
    st = pd.DataFrame({"storyId": ["urn:newsml:reuters.com:20250310:nA:0", "urn:newsml:reuters.com:20250310:nB:0",
                                   "urn:newsml:reuters.com:20250310:nC:0"],
                       "event_time": pd.to_datetime(["2025-03-10 19:59", "2025-03-10 20:00", None], utc=True),
                       "text": ["before close (15:59 NY)", "at the close (16:00 NY)", "no timestamp"]})
    st.to_parquet(p / "AAA_N_20250310.parquet", index=False)
    _, stories = event_news(tmp_path, "AAA.N", "2025-03-10")
    assert stories["text"].tolist() == ["before close (15:59 NY)"]


# ---------------------------------------------------------------- AG-50 v2 prompt / parser
HEADS = pd.DataFrame({"event_time": pd.to_datetime(["2025-03-10 13:00"], utc=True),
                      "headline": ["Testco (TST.N) CEO Jane Doe exits; $1.5 billion buyback paused"]})
PRICE = {"ret_shock": -0.09, "ret_5d": -0.01, "ret_21d": 0.02, "ret_63d": 0.05, "off_52w_high": -0.08,
         "close_vs_200dma": 0.03, "rv_60d": 0.32, "volume_x": 5.0}


def test_v2_prompt_structure_and_anonymity():
    msgs, facts = S.build_agent_prompt(HEADS, pd.DataFrame(), ["Testco", "TST.N", "TST"], "Tech", "Software", 30e9,
                                       -0.08, PRICE, None, {"days_since_last": 20, "next_in_window": False},
                                       version=S.AGENT_VERSION_V2)
    u = msgs[1]["content"]
    for tag in ("<impairment>", "<shock_type>", "<vol_context>", "<overreaction>", "<answer>"):
        assert tag in u
    for leak in ("Testco", "TST", "Jane", "1.5 billion", "$30B", "30B"):
        assert leak not in u
    assert "large ($20-100B)" in u and "standard deviations" in u and facts["version"] == "ag-v2"
    assert S.llm_kwargs("ag-v2")["json_mode"] is False


def test_v1_prompt_is_unchanged_by_default():
    a, _ = S.build_agent_prompt(HEADS, pd.DataFrame(), ["Testco", "TST.N", "TST"], "Tech", "Software", 30e9, -0.08,
                                PRICE, None, {"days_since_last": 20, "next_in_window": False})
    b, _ = S.build_agent_prompt(HEADS, pd.DataFrame(), ["Testco", "TST.N", "TST"], "Tech", "Software", 30e9, -0.08,
                                PRICE, None, {"days_since_last": 20, "next_in_window": False}, version="ag-v1")
    assert a == b and "Return JSON:" in a[1]["content"] and "<impairment>" not in a[1]["content"]
    with pytest.raises(ValueError):
        S.build_agent_prompt(HEADS, pd.DataFrame(), [], None, None, 1e9, -0.1, PRICE, None, None, version="v9")


V2_FULL = """<impairment>Minor: a one-off charge.</impairment>
<shock_type>Exogenous: sector rotation.</shock_type>
<vol_context>3 sigma move; vol should fade.</vol_context>
<overreaction>Impact 1-3% vs a 9% drop. Confidence 0.75.</overreaction>
<answer>{"category": "peer/sector contagion", "impairment": "minor", "shock_type": "exogenous",
"est_impact_pct": {"low": 1, "mid": 2, "high": 3}, "verdict": "not justified", "market_read": "m",
"decision": "take", "key_risk": "k", "reason": "r", "confidence": 0.75}</answer>"""


def test_parse_v2_full_and_impact_from_pct():
    p = S.parse_agent(V2_FULL, version="ag-v2", market_cap=50e9)
    assert p["parse_ok"] and p["decision"] == "take" and p["impairment"] == "minor" and p["shock_type"] == "exogenous"
    assert p["impact_mid"] == pytest.approx(1e9) and p["sections_found"] == list(S.SECTIONS)
    call = S.final_call(p, -0.09, 50e9, {"next_in_window": False})
    assert call["agent_take"] and call["ratio_mid"] == pytest.approx(4.5)


@pytest.mark.parametrize("drop", ["<shock_type>", "<vol_context>", "</answer>"])
def test_parse_v2_tolerates_missing_sections(drop):
    txt = V2_FULL
    if drop == "</answer>":
        txt = txt.replace("</answer>", "")[:-30]                       # cut off mid-answer
    else:
        start = txt.index(drop)
        end = txt.index(drop.replace("<", "</")) + len(drop) + 1
        txt = txt[:start] + txt[end:]
    p = S.parse_agent_v2(txt, 50e9)
    assert p["category"] == "peer/sector contagion"
    assert len(p["sections_found"]) >= 3


def test_parse_v2_garbage_and_unavailable():
    assert S.parse_agent_v2("I cannot help with that")["decision"] == "skip"
    assert S.parse_agent_v2(None)["verdict"] == "unavailable"


# ---------------------------------------------------------------- AG-51 / AG-52 execution
def test_resting_tp_never_below_a_penny():
    r = X.resting_take_profit(0.03, "2026-11-20", "2026-12-11", "2026-12-05")
    assert r["action"] == "hold" and r["limit"] is None and "too small" in r["warning"]


def test_resting_tp_close_now_when_natural_meets_rule():
    # credit 0.24, rule price on day 7 of 21 ~ 0.11; the spread's ask is 0.08 -> close now at 0.08
    r = X.resting_take_profit(0.24, "2026-11-20", "2026-12-11", "2026-11-27", sb=0.07, sa=0.09, lb=0.01, la=0.02)
    assert r["action"] == "close now" and r["limit"] == pytest.approx(0.08)


def test_resting_tp_wide_market_warning_keeps_rule_price():
    r = X.resting_take_profit(0.24, "2026-11-20", "2026-12-11", "2026-11-27", sb=0.10, sa=0.20, lb=0.03, la=0.06)
    assert r["action"] == "rest" and "wide market" in r["warning"]
    assert r["limit"] == X.take_profit_price(0.24, "2026-11-20", "2026-12-11", "2026-11-27")["limit"]


def test_order_desk_warns_on_spy(tmp_path):
    cmd = [sys.executable, str(ROOT / "scripts" / "order_desk.py"), "open", "--underlying", "SPY", "--expiry",
           "2026-12-11", "--short", "655", "--long", "650", "--sb", "0.62", "--sa", "0.66", "--lb", "0.40", "--la",
           "0.44", "--qty", "1"]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path).stdout
    assert "EARLY" in out
    cmd[cmd.index("SPY")] = "XSP"
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path).stdout
    assert "EARLY" not in out and "XSP put spread" in out


# ---------------------------------------------------------------- AG-53 ablation statistics
def test_cluster_bootstrap_and_masks():
    sys.path.insert(0, str(ROOT / "scripts"))
    import scripts.agent_diag as D
    x = np.array([0.1, 0.1, -0.2, 0.05, 0.05, 0.05])
    w = np.array(["a", "a", "b", "c", "c", "c"])
    m, lo, hi = D.cluster_boot(x, w, n=2000)
    assert m == pytest.approx(x.mean()) and lo <= m <= hi
    d = pd.DataFrame({"agent_take": [True, True, False, True], "fv_pass": [True, False, True, True],
                      "fv_has": [True, True, True, False], "confidence": [0.8, 0.9, 0.9, 0.5]})
    mk = D.masks(d)
    assert mk["H1"][0].tolist() == [True, False, False, False] and mk["H1"][1].tolist() == [False, True, False, False]
    assert mk["H2"][0].tolist() == [True, True, False, False] and mk["H3"][1].tolist() == [False, True, True, True]


def test_ablation_preregistration_hash_is_printed():
    import scripts.agent_diag as D
    spec, sha = D.load_spec(ROOT / "strategy" / "preregistered_agent_ablation.md")
    assert len(sha) == 16 and spec["pass"]["min_n_holdout"] == 10 and spec["holdout_from"] == "2026-06-01"


# ---------------------------------------------------------------- AG-54 macro classifier
def _px(n=300, shock=-0.03, tlt=0.0, gld=0.0, hyg5=0.0, trend=0.0005):
    idx = pd.bdate_range("2025-01-01", periods=n)
    spy = 500 * np.cumprod(np.r_[1, np.full(n - 1, 1 + trend)])
    spy[-1] = spy[-2] * (1 + shock)
    tl = np.full(n, 90.0); tl[-1] = 90 * (1 + tlt)
    gl = np.full(n, 200.0); gl[-1] = 200 * (1 + gld)
    hy = np.full(n, 80.0); hy[-1] = 80 * (1 + hyg5)
    return pd.DataFrame({"SPY": spy, "TLT": tl, "GLD": gl, "HYG": hy}, index=idx)


def test_macro_calm_uptrend_is_tradeable():
    px = _px(shock=-0.016)
    c = M.classify(M.macro_features(px, px.index[-1], iv9=0.15, iv30=0.18))
    assert c["label"] == "tradeable VRP dislocation" and not c["flags"]


def test_macro_dash_for_cash_with_credit_stress_is_break():
    px = _px(shock=-0.03, tlt=-0.01, gld=-0.02, hyg5=-0.03)
    c = M.classify(M.macro_features(px, px.index[-1], iv9=0.35, iv30=0.25))
    assert c["label"] == "systemic/liquidity break" and len(c["flags"]) >= 3


def test_macro_llm_only_makes_it_more_cautious():
    det = {"label": "tradeable VRP dislocation", "flags": [], "reasons": ["calm"]}
    assert M.combine(det, {"systemic": True, "reason": "bank run"})["label"] == "unclear"
    assert M.combine(det, {"systemic": False})["label"] == "tradeable VRP dislocation"
    assert M.classify_headlines(["Stocks fall"], llm=None)["source"] == "offline"
    assert M.classify_headlines(["Stocks fall"], llm=lambda m: None)["systemic"] is None
    v = M.classify_headlines(["Banks"], llm=lambda m: '{"systemic": true, "reason": "funding", "confidence": 0.7}')
    assert v["systemic"] is True and v["source"] == "llm"


def test_macro_features_point_in_time():
    px = _px()
    px.loc[px.index[-1] + pd.offsets.BDay(1), :] = [1.0, 1.0, 1.0, 1.0]     # a future crash
    f = M.macro_features(px, px.index[-2])
    assert f["spy_dd63"] > -0.05
