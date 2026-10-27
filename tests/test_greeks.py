"""
Tests for src/options/greeks.py (OPT-01 .. OPT-04).

Run from the repo root:  python -m pytest tests/test_greeks.py
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.options.greeks import bs_delta, delta_from_iv, pv_dividends, years_to_expiry  # noqa: E402


def test_textbook_value():
    # S=K=100, T=1, r=5%, vol=20% -> d1 = 0.35, N(d1) = 0.6368
    assert bs_delta(100, 100, 1, 0.05, 0.20, kind="call") == pytest.approx(0.6368, abs=1e-4)
    assert bs_delta(100, 100, 1, 0.05, 0.20, kind="put") == pytest.approx(-0.3632, abs=1e-4)


def test_put_call_delta_parity():
    # call delta - put delta = exp(-qT)
    for S, K, q in [(100, 90, 0.0), (50, 60, 0.02), (300, 250, 0.01)]:
        c = bs_delta(S, K, 0.1, 0.03, 0.4, q=q, kind="call")
        p = bs_delta(S, K, 0.1, 0.03, 0.4, q=q, kind="put")
        assert c - p == pytest.approx(math.exp(-q * 0.1), abs=1e-12)


def test_put_delta_range_and_moneyness():
    far_otm = bs_delta(100, 50, 30 / 365, 0.02, 0.3)
    far_itm = bs_delta(100, 150, 30 / 365, 0.02, 0.3)
    assert -0.001 < far_otm <= 0
    assert -1 <= far_itm < -0.999
    # lower strike -> smaller (less negative) put delta
    assert bs_delta(100, 90, 0.1, 0.02, 0.3) > bs_delta(100, 95, 0.1, 0.02, 0.3)


def test_expiry_and_bad_inputs():
    assert bs_delta(90, 100, 0, 0.02, 0.3) == -1.0
    assert bs_delta(110, 100, 0, 0.02, 0.3) == 0.0
    assert bs_delta(100, 100, 0.1, 0.02, 0) is None
    assert bs_delta(100, 100, 0.1, 0.02, float("nan")) is None
    assert bs_delta(-1, 100, 0.1, 0.02, 0.3) is None
    with pytest.raises(ValueError):
        bs_delta(100, 100, 0.1, 0.02, 0.3, kind="straddle")


def test_years_to_expiry():
    assert years_to_expiry("2022-05-18", "2022-06-17") == pytest.approx(30 / 365)
    assert years_to_expiry("2022-06-20", "2022-06-17") == 0


def test_pv_dividends_window():
    divs = [("2022-05-12", 0.60), ("2022-08-10", 0.60)]
    # only the May dividend is inside (as_of, expiry]
    assert pv_dividends("2022-05-01", "2022-06-17", divs, 0.0) == pytest.approx(0.60)
    # ex-date on as_of itself is already paid -> excluded
    assert pv_dividends("2022-05-12", "2022-06-17", divs, 0.0) == 0
    assert pv_dividends("2022-05-01", "2022-06-17", divs, 0.05) < 0.60


def test_dividend_makes_put_delta_more_negative():
    no_div = delta_from_iv(116, 115, "2022-05-02", "2022-06-17", 0.40, 0.006)
    with_div = delta_from_iv(116, 115, "2022-05-02", "2022-06-17", 0.40, 0.006, dividends=[("2022-05-12", 0.60)])
    assert with_div < no_div


def test_percent_units_rejected():
    with pytest.raises(ValueError):
        delta_from_iv(116, 115, "2022-05-02", "2022-06-17", 40.0, 0.006)
    with pytest.raises(ValueError):
        delta_from_iv(116, 115, "2022-05-02", "2022-06-17", 0.40, 0.6)
