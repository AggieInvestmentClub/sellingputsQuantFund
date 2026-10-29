"""
Tests for src/data/validate.py (DAT-05 .. DAT-07), on made-up data.

Run from the repo root:  python -m pytest tests/test_validate.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.validate import membership_issues, price_issues, split_factors  # noqa: E402


def _px(closes, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=len(closes), name="date")
    return pd.DataFrame({"TRDPRC_1": closes}, index=idx)


# ---------- DAT-05 price_issues ----------

def test_price_issues_clean():
    out = price_issues(_px(100 + np.arange(30) * 0.5))
    assert out["problems"] == ""
    assert out["rows"] == 30 and out["max_gap_days"] == 0 and out["longest_stale"] == 1


def test_price_issues_empty_and_missing_column():
    assert price_issues(pd.DataFrame())["problems"] == "empty"
    assert "no TRDPRC_1" in price_issues(pd.DataFrame({"OPEN_PRC": [1.0]}, index=[pd.Timestamp("2024-01-02")]))["problems"]


def test_price_issues_finds_each_problem():
    closes = [100.0, 101, np.nan, 102, 0, 103, 210, 210, 210, 210, 210, 211]
    px = _px(closes)
    px = pd.concat([px, px.iloc[[-1]]])                      # duplicate last date
    px = px.drop(px.index[3])                                # skip one weekday (gap of 1, below limit)
    out = price_issues(px)
    assert out["nan_close"] == 1 and out["nonpositive"] == 1 and out["duplicate_dates"] == 1
    assert out["big_moves"] == 1 and out["big_move_dates"] == "2024-01-09"   # 103 -> 210
    assert out["longest_stale"] == 5
    assert "same close 5 days" in out["problems"]


def test_price_issues_gap():
    px = pd.concat([_px([100.0, 100.5, 101.0]), _px([101.0, 102.0], start="2024-02-01")])
    out = price_issues(px)
    assert out["max_gap_days"] == 20                         # Jan 4 .. Feb 1: 20 weekdays missing
    assert "gap of 20 weekdays" in out["problems"]


# ---------- DAT-06 membership_issues ----------

def _snaps(spec):
    return pd.DataFrame([(pd.Timestamp(d), r) for d, rics in spec.items() for r in rics], columns=["date", "ric"])


def test_membership_basic_problems():
    m = _snaps({"2024-01-31": ["A.N", "B.N", "B.N", ""], "2024-03-31": ["A.N", "B.N"]})
    out = membership_issues(m, min_size=2, max_size=5)
    kinds = out["kind"].tolist()
    assert "blank_ric" in kinds and "duplicate_ric" in kinds
    assert out.loc[out["kind"] == "missing_month", "date"].tolist() == [pd.Timestamp("2024-02-29")]


def test_membership_size():
    m = _snaps({"2024-01-31": ["A.N"], "2024-02-29": ["A.N", "B.N"]})
    out = membership_issues(m, min_size=2, max_size=5)
    assert out[["kind", "date"]].values.tolist() == [["size", pd.Timestamp("2024-01-31")]]


def test_membership_against_leavers():
    m = _snaps({"2024-01-31": ["A.N", "B.N", "C.N"],
                "2024-02-29": ["A.N", "C.N"],            # B left with an event: fine
                "2024-03-31": ["C.N"],                   # A left with no event
                "2024-04-30": ["C.N"]})
    ch = pd.DataFrame({"ric": ["B.N", "C.N", "Z.N"], "change": ["Leaver"] * 3,
                       "date": pd.to_datetime(["2024-02-15", "2024-03-10", "2024-04-05"])})
    out = membership_issues(m, ch, min_size=1)
    got = set(map(tuple, out[["kind", "ric"]].values.tolist()))
    assert got == {("removed_without_leaver", "A.N"),     # dropped, no event
                   ("leaver_still_member", "C.N"),        # event, but still listed
                   ("leaver_not_member", "Z.N")}          # event for a RIC never listed


def test_membership_against_joiners():
    m = _snaps({"2024-01-31": ["A.N"], "2024-02-29": ["A.N", "B.N", "D.N"]})
    ch = pd.DataFrame({"ric": ["B.N"], "change": ["Joiner"], "date": pd.to_datetime(["2024-02-10"])})
    out = membership_issues(m, ch, min_size=1)
    assert out[["kind", "ric"]].values.tolist() == [["added_without_joiner", "D.N"]]


# ---------- DAT-07 split_factors ----------

def test_split_factors_four_for_one():
    idx = pd.bdate_range("2020-08-25", periods=6)
    raw = pd.Series([500.0, 504, 506.09, 129, 130, 131], index=idx)      # 4-for-1 after day 3
    adj = pd.Series([125.0, 126, 126.5225, 129, 130, 131], index=idx)
    out = split_factors(adj, raw)
    assert len(out) == 1
    assert out.loc[0, "date"] == idx[3]
    assert abs(out.loc[0, "split"] - 4.0) < 1e-9


def test_split_factors_none_and_reverse():
    idx = pd.bdate_range("2023-01-02", periods=4)
    px = pd.Series([10.0, 11, 12, 13], index=idx)
    assert split_factors(px, px).empty
    raw = pd.Series([1.0, 1.1, 12, 13], index=idx)                       # 1-for-10 reverse split
    out = split_factors(px, raw)
    assert abs(out.loc[0, "split"] - 0.1) < 1e-9
