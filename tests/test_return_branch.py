"""Tests for the return branch account (AG-42): overlap limits, open-risk cap, compounding, stats. No data files."""
import numpy as np
import pandas as pd
import pytest

from scripts import index_branch as IB
from scripts.return_branch import account_stats, entry_dates_x, simulate


def _t(rows, k=1, risk=0.10, cap=0.10):
    t = pd.DataFrame(rows, columns=["date", "exit", "ror_net"])
    t["date"], t["exit"] = pd.to_datetime(t["date"]), pd.to_datetime(t["exit"])
    return t.assign(k=k, risk=risk, cap=cap)


OVERLAP = [("2020-01-02", "2020-01-20", 0.05), ("2020-01-06", "2020-01-24", 0.05),
           ("2020-01-08", "2020-01-28", 0.05), ("2020-01-10", "2020-01-30", 0.05)]


def test_one_open_skips_overlaps_and_matches_index_branch():
    t = _t(OVERLAP)
    taken, booked, mo = simulate(t)
    assert len(taken) == 1 and mo == pytest.approx(0.10)
    assert len(IB.one_open(t)) == 1
    # an exit on the entry day does not free the slot (same as index_branch.one_open)
    t2 = _t([("2020-01-02", "2020-01-10", 0.05), ("2020-01-10", "2020-01-20", 0.05), ("2020-01-13", "2020-01-30", 0.05)])
    assert list(simulate(t2)[0]["date"].dt.day) == [2, 13] == list(IB.one_open(t2)["date"].dt.day)


def test_k_and_cap_limits():
    assert len(simulate(_t(OVERLAP, k=3, risk=0.10, cap=0.30))[0]) == 3          # 4th blocked by k
    assert len(simulate(_t(OVERLAP, k=3, risk=0.10, cap=0.20))[0]) == 2          # cap binds before k
    assert len(simulate(_t(OVERLAP, k=2, risk=0.20, cap=0.30))[0]) == 1          # 0.2 + 0.2 > 0.3
    _, _, mo = simulate(_t(OVERLAP, k=2, risk=0.20, cap=0.40))
    assert mo == pytest.approx(0.40)


def test_risk_is_taken_on_booked_equity_at_entry():
    t = _t([("2020-01-02", "2020-01-10", -1.0), ("2020-01-13", "2020-01-20", 0.10)], risk=0.20, cap=0.20)
    taken, booked, _ = simulate(t)
    assert taken["risk_amt"].tolist() == pytest.approx([0.20, 0.16])            # 20% of 1.0, then of 0.8
    assert booked.sum() == pytest.approx(-0.20 + 0.016)
    # overlapping: the second entry is sized before the first one's P&L is booked
    t = _t([("2020-01-02", "2020-01-20", -1.0), ("2020-01-06", "2020-01-24", 0.10)], k=2, risk=0.10, cap=0.20)
    assert simulate(t)[0]["risk_amt"].tolist() == pytest.approx([0.10, 0.10])


def test_account_stats_matches_index_branch_for_one_open():
    rng = np.random.default_rng(0)
    d = pd.bdate_range("2015-01-05", periods=60, freq="10B")
    rows = [(x, x + pd.Timedelta(days=9), float(r)) for x, r in zip(d, rng.normal(0.02, 0.2, len(d)))]
    t = _t(rows)
    taken, booked, mo = simulate(t)
    st = account_stats(booked, pd.Timestamp("2015-01-01"), pd.Timestamp("2017-12-31"), taken=taken, max_open=mo)
    ref = IB.stats(IB.one_open(t), 0.10)
    assert st["total"] == pytest.approx(ref["total"])
    assert st["maxdd"] == pytest.approx(ref["maxdd"])
    assert st["n"] == ref["n"]


def test_drawdown_and_worst_month():
    t = _t([("2021-03-01", "2021-03-15", 0.5), ("2021-04-01", "2021-04-20", -1.0), ("2021-05-03", "2021-05-20", 0.2)],
           risk=0.10, cap=0.10)
    taken, booked, mo = simulate(t)
    st = account_stats(booked, pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"), taken=taken, max_open=mo)
    assert st["maxdd"] == pytest.approx(-0.105 / 1.05)
    assert st["worst_month"] == pytest.approx(-0.105 / 1.05)
    assert st["max_losses"] == 1 and st["years_active"] == 1


def test_lower_threshold_finds_more_shocks():
    idx = pd.bdate_range("2022-01-03", periods=12)
    S = pd.Series([100, 98.8, 99.5, 100, 100.2, 100.4, 98.8, 99.2, 99.5, 100, 101, 102.0], index=idx)
    f = pd.DataFrame({"S": S, "ret1": S.pct_change(), "ret2": S.pct_change(2), "guard_ok": True})
    assert len(entry_dates_x(f, "T10")) == 2 and len(entry_dates_x(f, "T15")) == 1
    f["guard_ok"] = False
    assert len(entry_dates_x(f, "T10+guard")) == 0
