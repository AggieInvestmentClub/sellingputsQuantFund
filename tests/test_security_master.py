"""
Tests for src/data/security_master.py (DAT-01 .. DAT-04).

Run from the repo root:  python -m pytest tests/test_security_master.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.security_master import companies, lseg_option_root, massive_ticker_from_ric, massive_ticker_on  # noqa: E402


def test_lseg_option_root():
    assert lseg_option_root("META.OQ") == "META"
    assert lseg_option_root("BRKb.N") == "BRKB"
    assert lseg_option_root("SIVB.OQ^C23") == "SIVB"


def test_massive_ticker_from_ric():
    assert massive_ticker_from_ric("META.OQ") == "META"
    assert massive_ticker_from_ric("BRKb.N") == "BRK.B"
    assert massive_ticker_from_ric("SIVB.OQ^C23") == "SIVB"
    assert massive_ticker_from_ric("A.N") == "A"


def test_massive_ticker_on_ticker_change():
    t = pd.DataFrame({"ric": ["META.OQ", "META.OQ", "SIVB.OQ^C23"],
                      "massive_ticker": ["FB", "META", "SIVB"],
                      "valid_from": [pd.Timestamp("2012-05-18"), pd.Timestamp("2022-06-09"), pd.NaT],
                      "source": ["events", "events", "ric_fallback"]})
    assert massive_ticker_on(t, "META.OQ", "2022-05-13") == "FB"
    assert massive_ticker_on(t, "META.OQ", "2022-06-09") == "META"
    assert massive_ticker_on(t, "META.OQ", "2010-01-01") == "FB"
    assert massive_ticker_on(t, "SIVB.OQ^C23", "2023-01-15") == "SIVB"
    assert massive_ticker_on(t, "XYZ.N", "2023-01-15") is None


def test_companies_collapses_share_classes():
    master = pd.DataFrame({"ric": ["GOOG.OQ", "GOOGL.OQ", "AAPL.OQ"],
                           "company_id": ["5030853586", "5030853586", "4295905573"]})
    out = companies(["GOOGL.OQ", "AAPL.OQ", "GOOG.OQ", "NEW.N"], master)
    assert out == {"5030853586": ["GOOG.OQ", "GOOGL.OQ"], "4295905573": ["AAPL.OQ"], "RIC:NEW.N": ["NEW.N"]}
