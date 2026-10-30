"""
Tests for src/events/analysis.py (EVT-07, EVT-08), on made-up data.

Run from the repo root:  python -m pytest tests/test_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.events.analysis import entry_returns, robust_summary  # noqa: E402
from src.events.build_events import find_shocks  # noqa: E402


def test_entry_returns_by_hand():
    d = pd.bdate_range("2020-01-01", periods=8)
    close = pd.Series([100.0, 90, 92, 95, 93, 99, 100, 101], index=d)   # shock on d[1]
    spy = pd.Series(0.0, index=d)
    ev = pd.DataFrame({"ric": ["X.N"], "date": [d[1]], "beta": [1.0]})
    out = entry_returns(ev, close, spy, lags=(1,), holds=(3,))
    # entry at d[2] close (92), hold d[3..5]: 95, 93, 99
    assert out.loc[0, "ret_e1_3"] == pytest.approx(99 / 92 - 1)
    assert out.loc[0, "low_e1_3"] == pytest.approx(93 / 92 - 1)
    assert out.loc[0, "car_e1_3"] == pytest.approx(95 / 92 - 1 + 93 / 95 - 1 + 99 / 93 - 1)


def test_entry_returns_market_part_removed_and_nan_at_end():
    d = pd.bdate_range("2020-01-01", periods=6)
    close = pd.Series([100.0, 90, 91, 92, 93, 94], index=d)
    spy = pd.Series([0, 0, 0.01, 0.01, 0.01, 0.01], index=d)
    ev = pd.DataFrame({"ric": ["X.N"], "date": [d[1]], "beta": [2.0]})
    out = entry_returns(ev, close, spy, lags=(1, 3), holds=(2,))
    assert out.loc[0, "car_e1_2"] == pytest.approx((92 / 91 - 1 - 0.02) + (93 / 92 - 1 - 0.02))
    assert np.isnan(out.loc[0, "ret_e3_2"])          # needs d[6], only 6 days


def test_entry_lag_zero_matches_find_shocks():
    rng = np.random.default_rng(1)
    d = pd.bdate_range("2016-01-04", periods=330)
    spy_ret = pd.Series(rng.normal(0, 0.01, len(d)), index=d)
    r = 1.2 * spy_ret + rng.normal(0, 0.01, len(d))
    r.iloc[300] = -0.12
    close = 100 * (1 + r).cumprod()
    m = pd.DataFrame({"date": pd.date_range("2015-12-31", "2017-06-30", freq="ME"), "ric": "X.N"})
    ev = find_shocks("X.N", close, spy_ret, m, [])
    assert len(ev) >= 1
    out = entry_returns(ev, close, spy_ret, lags=(0,), holds=(5, 10))
    for h in (5, 10):
        assert out[f"car_e0_{h}"].to_numpy() == pytest.approx(ev[f"car_{h}"].to_numpy())
        assert out[f"ret_e0_{h}"].to_numpy() == pytest.approx(ev[f"ret_{h}"].to_numpy())


def test_robust_summary_counts_a_crowded_day_once():
    ev = pd.DataFrame({"date": pd.to_datetime(["2020-03-09"] * 4 + ["2021-01-05", "2022-06-01"]),
                       "x": [-0.20, -0.20, -0.20, -0.20, 0.05, 0.03]})
    s = robust_summary(ev, ["x"]).iloc[0]
    assert s["n"] == 6 and s["n_dates"] == 3
    assert s["mean"] == pytest.approx((4 * -0.20 + 0.05 + 0.03) / 6)
    assert s["pct_tail"] == pytest.approx(4 / 6)
    daily = pd.Series([-0.20, 0.05, 0.03])
    assert s["t_by_date"] == pytest.approx(daily.mean() / (daily.std(ddof=1) / np.sqrt(3)))
