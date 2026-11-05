"""
Tests for src/options/chains.py (OPT-05..09) and src/events/option_panel.py (EVT-09..12), made-up data.

Run from the repo root:  python -m pytest tests/test_option_panel.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.events.option_panel import add_iv_delta, event_windows, root_volume, select_event_puts  # noqa: E402
from src.options.chains import bs_delta_vec, bs_price, implied_vol, occ_root, parse_occ  # noqa: E402
from src.options.greeks import bs_delta  # noqa: E402


# ---------- OPT-05 / OPT-06 ----------

def test_parse_occ():
    p = parse_occ(pd.Series(["O:AAPL250321P00200000", "O:BRKB241115C00450500", "O:A240419C00135000", "junk"]))
    assert p.loc[0, "root"] == "AAPL" and p.loc[0, "kind"] == "put" and p.loc[0, "strike"] == 200.0
    assert p.loc[0, "expiry"] == pd.Timestamp("2025-03-21")
    assert p.loc[1, "root"] == "BRKB" and p.loc[1, "strike"] == 450.5 and p.loc[1, "kind"] == "call"
    assert p.loc[2, "root"] == "A"
    assert p.loc[3].isna().all()


def test_occ_root():
    assert occ_root("BRK.B") == "BRKB" and occ_root("BF.B") == "BFB" and occ_root("META") == "META"
    assert occ_root(None) is None


# ---------- OPT-07 .. OPT-09 ----------

def test_bs_price_put_call_parity_and_intrinsic():
    S, K, T, r, v = 100.0, 95.0, 0.25, 0.04, 0.3
    call, put = bs_price(S, K, T, r, v, "call"), bs_price(S, K, T, r, v, "put")
    assert call - put == pytest.approx(S - K * np.exp(-r * T))
    assert bs_price(90, 100, 0, 0.04, 0.3, "put") == pytest.approx(10)


def test_implied_vol_round_trip_and_bad_prices():
    S = np.array([100, 100, 250, 40.0])
    K = np.array([95, 80, 240, 40.0])
    T = np.array([0.08, 0.1, 0.05, 0.03])
    r = np.full(4, 0.045)
    true_vol = np.array([0.25, 0.6, 0.35, 1.2])
    price = bs_price(S, K, T, r, true_vol, "put")
    assert implied_vol(price, S, K, T, r) == pytest.approx(true_vol, abs=1e-6)
    assert np.isnan(implied_vol(0.0, 100, 95, 0.1, 0.04))           # no price
    assert np.isnan(implied_vol(4.0, 90, 100, 0.1, 0.04))            # below intrinsic (10)
    assert np.isnan(implied_vol(2.0, 100, 95, 0.0, 0.04))            # expired


def test_delta_vec_matches_opt03():
    d = bs_delta_vec([100, 120], [95, 95], [0.1, 0.1], [0.04, 0.04], [0.3, 0.3])
    assert d[0] == pytest.approx(bs_delta(100, 95, 0.1, 0.04, 0.3))
    assert d[1] == pytest.approx(bs_delta(120, 95, 0.1, 0.04, 0.3))


# ---------- EVT-09 .. EVT-12 ----------

TICKERS = pd.DataFrame({"ric": ["META.OQ", "META.OQ"], "massive_ticker": ["FB", "META"],
                        "valid_from": pd.to_datetime(["2012-05-18", "2022-06-09"]), "source": "events"})


def test_event_windows():
    ev = pd.DataFrame({"ric": ["META.OQ", "META.OQ", "META.OQ"],
                       "date": pd.to_datetime(["2021-03-01", "2022-03-01", "2024-06-03"])})
    w = event_windows(ev, TICKERS, first_day="2022-01-03", last_day="2024-06-28")
    assert len(w) == 2                                                 # 2021 event dropped
    first = w.iloc[0]
    assert first["root"] == "FB" and first["event_id"] == "META.OQ|2022-03-01"
    assert first["min_expiry"] == pd.Timestamp("2022-03-16")          # Mar 2 + 14 days
    assert first["max_expiry"] == pd.Timestamp("2022-04-19")          # Mar 15 + 35 days
    assert bool(first["complete"]) and not bool(w.iloc[1]["complete"])


def _day(rows):
    d = pd.DataFrame(rows, columns=["ticker", "close", "volume"])
    for c in ["open", "high", "low"]:
        d[c] = d["close"]
    d["transactions"] = 1
    return pd.concat([d, parse_occ(d["ticker"])], axis=1)


def test_select_event_puts_and_root_volume():
    ev = pd.DataFrame({"ric": ["META.OQ"], "date": pd.to_datetime(["2024-06-03"])})
    w = event_windows(ev, TICKERS, first_day="2024-01-02", last_day="2024-12-31")
    day = _day([("O:META240614P00450000", 3.0, 100),     # expiry before day+1+14 (Jun 18) -> out
                ("O:META240705P00450000", 6.0, 50),      # in
                ("O:META240705C00500000", 4.0, 70),      # call -> out
                ("O:META240816P00400000", 2.0, 10),      # after max expiry (Jun 17 + 35 = Jul 22) -> out
                ("O:AAPL240705P00180000", 1.0, 30)])     # other root -> out
    out = select_event_puts(day, "2024-06-04", w)
    assert out["ticker"].tolist() == ["O:META240705P00450000"]
    assert out.loc[0, "event_id"] == "META.OQ|2024-06-03"
    assert select_event_puts(day, "2024-05-31", w).empty                # before the shock
    v = root_volume(day, "2024-06-04", ["META"]).iloc[0]
    assert v["opt_volume"] == 230 and v["put_volume"] == 160 and v["contracts_traded"] == 4


def test_add_iv_delta_recovers_vol_and_uses_dividends():
    d0 = pd.Timestamp("2024-06-04")
    S, K, T, r, vol = 100.0, 95.0, 31 / 365, 0.05, 0.4
    div = pd.DataFrame({"ric": ["X.N"], "ex_date": [pd.Timestamp("2024-06-20")], "amount": [1.0]})
    pv = 1.0 * np.exp(-r * 16 / 365)
    price = float(bs_price(S - pv, K, T, r, vol, "put"))
    panel = pd.DataFrame({"event_id": ["e"], "ric": ["X.N"], "date": [d0], "expiry": [pd.Timestamp("2024-07-05")],
                          "strike": [K], "close": [price]})
    closes = pd.DataFrame({"ric": ["X.N"], "date": [d0], "close": [S]})
    rates = pd.Series([r], index=[pd.Timestamp("2024-06-03")])
    out = add_iv_delta(panel, closes, rates, div).iloc[0]
    assert out["pv_div"] == pytest.approx(pv)
    assert out["iv"] == pytest.approx(vol, abs=1e-6)
    assert out["delta"] == pytest.approx(bs_delta(S - pv, K, T, r, vol))
