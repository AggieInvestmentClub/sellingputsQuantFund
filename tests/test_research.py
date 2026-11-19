"""Tests for the overfitting checks (AG-21/22) and the fair-value evidence (AG-14..16). No network."""
import numpy as np
import pandas as pd
import pytest

from src.agent import valuation as V
from src.backtest.overfit import deflated_sharpe, deflated_sharpe_from_matrix, pbo_cscv


def test_deflated_sharpe_penalizes_many_trials():
    one = deflated_sharpe(0.1, 250, 1, 0.0)
    many = deflated_sharpe(0.1, 250, 200, 0.004)
    assert one > 0.9 and many < one


def test_dsr_and_pbo_on_noise():
    rng = np.random.default_rng(1)
    R = rng.normal(0, 0.01, (400, 50))           # 50 settings, no skill
    d = deflated_sharpe_from_matrix(R)
    assert d["n_trials"] == 50 and d["dsr"] < 0.95
    p = pbo_cscv(R, n_splits=8)
    assert p["n_combos"] == 70 and 0.2 < p["pbo"] < 0.8   # noise: around a coin flip


def test_pbo_low_when_one_setting_is_real():
    rng = np.random.default_rng(2)
    R = rng.normal(0, 0.01, (400, 20))
    R[:, 3] += 0.004                             # one setting with a real edge
    assert pbo_cscv(R, n_splits=8)["pbo"] < 0.1


def _setup(tmp_path):
    d = tmp_path / "prices" / "daily"
    d.mkdir(parents=True)
    idx = pd.bdate_range("2023-01-02", "2024-06-28", name="date")
    for ric, eps in (("AAA.N", 5.0), ("B.N", 5.0), ("C.N", 5.0), ("D.N", 5.0)):
        pd.DataFrame({"TRDPRC_1": 100.0, "ACVOL_UNS": 1e6}, index=idx).to_parquet(d / f"{ric.replace('.', '_')}_adj.parquet")
        f = tmp_path / "fundamentals"
        f.mkdir(exist_ok=True)
        w = pd.date_range("2023-01-06", "2024-06-28", freq="W-FRI")
        e = np.full(len(w), eps)
        if ric == "AAA.N":
            e = np.linspace(4.0, 6.0, len(w))     # EPS rising -> P/E falling
            e[-2:] = [99.0, 99.0]                 # values dated AFTER the shock must be ignored
        rows = [("eps_ntm", dt, v) for dt, v in zip(w, e)] + [("target", w[0], 120.0), ("n_est", w[0], 12.0)]
        pd.DataFrame(rows, columns=["field", "date", "value"]).to_parquet(f / f"{ric.replace('.', '_')}.parquet")
    V._fund.cache_clear()
    gics = pd.DataFrame({"ric": ["AAA.N", "B.N", "C.N", "D.N"], "sector": "X", "industry_group": "G", "industry": "I"})
    return idx, gics


def test_valuation_point_in_time_and_gate(tmp_path):
    idx, gics = _setup(tmp_path)
    shock = pd.Timestamp("2024-06-14")
    v = V.valuation_context(tmp_path, "AAA.N", shock, gics)
    assert v["fwd_pe"] < 100 / 5.5 and v["fwd_pe"] > 100 / 6.1          # no 99.0 from after the shock
    assert v["pe_pctile_3y"] < 0.1                                       # cheapest it has been
    assert v["pe_vs_sector"] < 1.0 and v["target_gap"] == pytest.approx(0.2)
    ok, why = V.gate(v)
    assert ok and why == []
    bad = {**v, "pe_pctile_3y": 0.9, "rev_30d": -0.05}
    ok, why = V.gate(bad)
    assert not ok and len(why) == 2
    assert V.gate(None) == (True, ["no fundamentals file"])
    assert "Forward P/E" in V.valuation_lines(v)[0]
