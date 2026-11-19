"""AG tests: point-in-time evidence, look-ahead guards, parsing and the take rule. No network."""
import json

import numpy as np
import pandas as pd
import pytest

from src.agent import screen as S
from src.agent.evidence import _drop_late_ids, earnings_context, event_news, load_close, peer_context, price_context


def _write_px(raw, ric, closes, start="2024-01-01"):
    d = raw / "prices" / "daily"
    d.mkdir(parents=True, exist_ok=True)
    idx = pd.bdate_range(start, periods=len(closes), name="date")
    pd.DataFrame({"TRDPRC_1": closes, "ACVOL_UNS": 1e6}, index=idx).to_parquet(d / f"{ric.replace('.', '_')}_adj.parquet")
    return idx


def test_price_context_is_point_in_time(tmp_path):
    closes = list(np.linspace(100, 120, 300)) + [108.0] + [500.0] * 5   # shock, then a huge future move
    idx = _write_px(tmp_path, "AAA.N", closes)
    shock = idx[300]
    pc = price_context(load_close(tmp_path, "AAA.N"), shock)
    assert pc["ret_shock"] == pytest.approx(108 / 120 - 1)
    assert pc["ret_5d"] > 0 and pc["off_52w_high"] == pytest.approx(0.0)   # pre-shock high, no future
    saturday = shock + pd.offsets.Week(weekday=5)
    assert price_context(load_close(tmp_path, "AAA.N"), saturday) is None          # not a trading day


def test_peer_context(tmp_path):
    idx = _write_px(tmp_path, "AAA.N", [100.0] * 100)
    for i, r in enumerate(["B.N", "C.N", "D.N"]):
        _write_px(tmp_path, r, [100.0] * 99 + [96.0 if i < 2 else 101.0])
    gics = pd.DataFrame({"ric": ["AAA.N", "B.N", "C.N", "D.N", "E.N"], "sector": "X",
                         "industry_group": "G", "industry": ["I"] * 4 + ["Other"]})
    pc = peer_context(tmp_path, "AAA.N", idx[-1], gics, {"B.N", "C.N", "D.N"})
    assert pc["n"] == 3 and pc["median"] == pytest.approx(-0.04) and pc["share_down_3pct"] == pytest.approx(2 / 3)
    assert peer_context(tmp_path, "AAA.N", idx[-1], gics, set()) is None      # no members -> nothing


def test_earnings_window():
    e = pd.DataFrame({"ric": ["A", "A"], "report_date": pd.to_datetime(["2024-01-10", "2024-03-01"])})
    assert earnings_context(e, "A", "2024-02-01") == {"days_since_last": 22, "next_in_window": True}
    assert earnings_context(e, "A", "2024-01-15")["next_in_window"] is False


def test_late_story_ids_dropped(tmp_path):
    n = pd.DataFrame({"storyId": ["urn:newsml:reuters.com:20240531:nA:0", "urn:newsml:newsroom:20240626:nB:0"],
                      "event_time": pd.to_datetime(["2024-05-31 13:00", "2024-05-31 00:00"], utc=True),
                      "headline": ["real", "from the future"]})
    assert _drop_late_ids(n, "2024-05-31")["headline"].tolist() == ["real"]
    p = tmp_path / "news" / "reuters_events"
    p.mkdir(parents=True)
    n.to_parquet(p / "AAA_N_20240531.parquet", index=False)
    heads, _ = event_news(tmp_path, "AAA.N", "2024-05-31")
    assert heads["headline"].tolist() == ["real"]


def test_prompt_has_evidence_and_is_anonymized():
    heads = pd.DataFrame({"event_time": pd.to_datetime(["2024-05-31 13:00"], utc=True),
                          "headline": ["Testco Inc (TST.N) shares fall on CEO exit"]})
    price = {"ret_shock": -0.1, "ret_5d": -0.02, "ret_21d": 0.01, "ret_63d": 0.05, "off_52w_high": -0.1,
             "close_vs_200dma": 0.02, "rv_60d": 0.3, "volume_x": 4.0}
    msgs, _ = S.build_agent_prompt(heads, pd.DataFrame(), ["Testco Inc", "Testco", "TST.N", "TST"], "Tech", "Software",
                                   20e9, -0.08, price, None, {"days_since_last": 30, "next_in_window": False})
    u = msgs[1]["content"]
    assert "Testco" not in u and "TST" not in u
    assert "4.0x its 20-day average" in u and '"decision": "take" | "skip"' in u
    assert u.count("Return JSON:") == 1


def _answer(**kw):
    d = {"category": "analyst downgrade", "est_impact_usd": {"low": 1e8, "mid": 2e8, "high": 5e8},
         "verdict": "not justified", "decision": "take", "reason": "r", "confidence": 0.7,
         "market_read": "m", "key_risk": "k", "impact_basis": "b"}
    d.update(kw)
    return json.dumps(d)


def test_parse_and_take_rule():
    earn = {"days_since_last": 30, "next_in_window": False}
    ok = S.final_call(S.parse_agent(_answer()), -0.1, 20e9, earn)
    assert ok["agent_take"] and ok["ratio_mid"] == pytest.approx(10.0) and ok["why_not"] == ""
    assert not S.final_call(S.parse_agent(_answer(decision="skip")), -0.1, 20e9, earn)["agent_take"]
    assert not S.final_call(S.parse_agent(_answer(verdict="justified")), -0.1, 20e9, earn)["agent_take"]
    assert not S.final_call(S.parse_agent(_answer()), -0.1, 20e9, {**earn, "next_in_window": True})["agent_take"]
    bad = S.parse_agent("I think you should buy")
    assert bad["decision"] == "skip" and not S.final_call(bad, -0.1, 20e9, earn)["agent_take"]
    assert S.parse_agent(_answer(decision="BUY"))["decision"] == "skip"
    assert not S.final_call(S.parse_agent(_answer(category="guidance cut")), -0.1, 20e9, earn)["agent_take"]  # STR-2.3a
