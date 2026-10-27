"""
Tests for src/events/build_events.py (EVT-01 .. EVT-06).

Run from the repo root:  python -m pytest tests/test_events.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.events.build_events import dedupe_events, find_shocks, member_mask, near_earnings, rolling_beta, summarize  # noqa: E402
from strategy.strategy_logic import estimate_beta  # noqa: E402


def _market(n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2016-01-04", periods=n)
    spy = pd.Series(rng.normal(0, 0.01, n), index=d)
    stock = 1.5 * spy + pd.Series(rng.normal(0, 0.01, n), index=d)
    return d, spy, stock


def test_rolling_beta_matches_estimate_beta():
    d, spy, stock = _market()
    rb = rolling_beta(stock, spy)
    for day in [d[260], d[300], d[399]]:
        assert rb[day] == pytest.approx(estimate_beta(stock, spy, day), rel=1e-9)
    assert np.isnan(rb[d[100]])          # not enough history


def test_member_mask_uses_previous_month_end():
    m = pd.DataFrame({"date": pd.to_datetime(["2016-01-31", "2016-02-29"]), "ric": ["X.N", "X.N"]})
    dates = pd.to_datetime(["2016-01-15", "2016-02-10", "2016-03-31", "2016-04-01"])
    assert member_mask(dates, "X.N", m).tolist() == [False, True, True, False]


def test_near_earnings_weekday_buffer():
    dates = pd.to_datetime(["2016-03-04", "2016-03-10", "2016-03-11"])     # Fri, Thu, Fri
    # earnings Mon 2016-03-07: Fri 03-04 is 1 weekday away, Thu 03-10 is 3, Fri 03-11 is 4
    assert near_earnings(dates, ["2016-03-07"]).tolist() == [True, True, False]
    assert near_earnings(dates, []).tolist() == [False, False, False]


def test_find_shocks_and_car():
    d, spy, stock = _market()
    stock.iloc[300] = -0.12                                # planted shock
    stock.iloc[301:306] = 0.01                             # rebound
    spy.iloc[300:306] = 0.0
    close = (1 + stock).cumprod() * 100
    close.iloc[0] = 100
    m = pd.DataFrame({"date": pd.date_range("2015-12-31", "2017-12-31", freq="ME"), "ric": "X.N"})
    ev = find_shocks("X.N", close, spy, m, earnings_dates=[], threshold=-0.10)
    hit = ev[ev["date"] == d[300]]
    assert len(hit) == 1
    assert hit["car_5"].iloc[0] == pytest.approx(0.05, abs=1e-9)   # 5 days of +1% with SPY flat
    # earnings on the shock day removes it
    ev2 = find_shocks("X.N", close, spy, m, earnings_dates=[d[300]], threshold=-0.10)
    assert d[300] not in set(ev2["date"])


def test_dedupe_share_classes_and_cooldown():
    master = pd.DataFrame({"ric": ["GOOG.OQ", "GOOGL.OQ"], "company_id": ["A", "A"],
                           "first_member": pd.to_datetime(["2015-04-30", "2015-04-30"]),
                           "last_member": pd.to_datetime(["2026-09-30", "2026-09-30"])})
    ev = pd.DataFrame({"ric": ["GOOG.OQ", "GOOGL.OQ", "GOOG.OQ", "GOOG.OQ"],
                       "date": pd.to_datetime(["2020-03-02", "2020-03-02", "2020-03-10", "2020-06-01"]),
                       "abnormal": [-0.06, -0.06, -0.07, -0.08]})
    out = dedupe_events(ev, master)
    # GOOGL dropped (same company, same day, GOOG chosen alphabetically on a tie), Mar 10 within 20 days
    assert out["date"].tolist() == list(pd.to_datetime(["2020-03-02", "2020-06-01"]))
    assert set(out["ric"]) == {"GOOG.OQ"}


def test_dedupe_keeps_events_from_both_rics_on_different_days():
    # one stock that moved exchange: BKR.N then BKR.OQ
    master = pd.DataFrame({"ric": ["BKR.N", "BKR.OQ"], "company_id": ["B", "B"],
                           "first_member": pd.to_datetime(["2017-07-31", "2021-01-31"]),
                           "last_member": pd.to_datetime(["2020-12-31", "2026-09-30"])})
    ev = pd.DataFrame({"ric": ["BKR.N", "BKR.OQ"], "date": pd.to_datetime(["2019-05-01", "2023-05-01"]),
                       "abnormal": [-0.06, -0.07]})
    assert dedupe_events(ev, master)["ric"].tolist() == ["BKR.N", "BKR.OQ"]


def test_summarize():
    ev = pd.DataFrame({"abnormal": [-0.06, -0.12, -0.20],
                       **{f"car_{h}": [0.01, 0.02, 0.03] for h in (1, 5, 10, 20)}})
    s = summarize(ev)
    row = s[(s.threshold == -0.10) & (s.horizon == 5)].iloc[0]
    assert row["n"] == 2 and row["mean"] == pytest.approx(0.025)
