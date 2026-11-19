"""Tests for scripts/etf_branch.py (AG-33..36): ticker parsing, vol-scaled shock, portfolio risk cap."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.etf_branch as E  # noqa: E402

SPEC, _ = E.prereg()


def test_parse_tickers_root_put_only():
    t = pd.Series(["O:QQQ250321P00450000", "O:QQQ250321C00450000", "O:QQQ1250321P00450000",
                   "O:SPY250321P00450500", "O:QQQ250321P00012500"])
    p = E.parse_tickers(t, "QQQ")
    assert p.loc[0, "expiry"] == pd.Timestamp("2025-03-21") and p.loc[0, "strike"] == 450.0
    assert p.loc[1:3, "expiry"].isna().all()           # call, adjusted root, other root
    assert p.loc[4, "strike"] == 12.5


def test_cost_model_tight_vs_other():
    c = np.array([0.10, 1.00])
    assert np.allclose(E.half_spread("SPY", c, SPEC), [0.01, 0.05])
    assert np.allclose(E.half_spread("XLU", c, SPEC), [0.02, 0.10])


def _series(rets, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(rets) + 1)
    return pd.Series(100 * np.cumprod(np.r_[1.0, 1 + np.asarray(rets)]), index=idx)


def test_shock_scales_with_relative_vol():
    rng = np.random.default_rng(1)
    n = 300
    spy = _series(rng.normal(0, 0.01, n))
    etf_r = rng.normal(0, 0.02, n)                    # twice SPY's volatility -> threshold about -3%
    etf_r[-1] = -0.02                                  # a -2% day: a shock for SPY, not for this ETF
    etf = _series(etf_r)
    flags, r1, s = E.shock_flags(etf, spy, SPEC)
    assert 1.6 < s.iloc[-1] < 2.5
    assert not flags.iloc[-1]
    etf_r[-1] = -0.05
    flags, _, _ = E.shock_flags(_series(etf_r), spy, SPEC)
    assert flags.iloc[-1]


def test_spy_keeps_the_fixed_rule():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.006, 300)                      # calm SPY: scaling would lower the bar, the rule must not
    r[-1] = -0.012
    spy = _series(r)
    flags, _, s = E.shock_flags(spy, spy, SPEC, is_spy=True)
    assert (s == 1.0).all() and not flags.iloc[-1]
    r[-1] = -0.016
    flags, _, _ = E.shock_flags(_series(r), _series(r), SPEC, is_spy=True)
    assert flags.iloc[-1]


def _cand(root, d, exit_, width=2.0, credit=0.10, ror=0.05):
    return {"root": root, "date": pd.Timestamp(d), "exit": pd.Timestamp(exit_), "expiry": pd.Timestamp(exit_),
            "width": width, "credit": credit, "cost": credit - ror * (width - credit), "ror_net": ror}


def test_portfolio_risk_cap_and_one_per_etf():
    # $2,500: budget $250, cap $750; each $2-wide spread risks $190 -> 1 contract; 4 same-day entries -> 3 fit
    c = pd.DataFrame([_cand("SPY", "2025-01-06", "2025-01-24"), _cand("QQQ", "2025-01-06", "2025-01-24"),
                      _cand("IWM", "2025-01-06", "2025-01-24"), _cand("DIA", "2025-01-06", "2025-01-24"),
                      _cand("SPY", "2025-01-10", "2025-01-31"),      # SPY still open -> ignored
                      _cand("DIA", "2025-01-27", "2025-02-14")])     # after the others closed -> taken
    t, skipped = E.portfolio(c, 2500, SPEC)
    assert list(t["root"]) == ["SPY", "QQQ", "IWM", "DIA"]
    assert list(t["date"].dt.day) == [6, 6, 6, 27]
    assert skipped == 1
    assert (t["contracts"] == 1).all() and np.isclose(t["risk_usd"].iloc[0], 190.0)


def test_portfolio_sizes_by_budget_and_skips_too_wide():
    c = pd.DataFrame([_cand("XLF", "2025-01-06", "2025-01-24", width=1.0, credit=0.05),   # $95 -> 2 contracts
                      _cand("XLE", "2025-01-06", "2025-01-24", width=3.0, credit=0.05)])  # $295 > 1.15 x 250 -> skip
    t, _ = E.portfolio(c, 2500, SPEC)
    assert list(t["root"]) == ["XLF"] and int(t["contracts"].iloc[0]) == 2


def test_cluster_boot_interval_contains_mean():
    x = np.r_[np.full(20, 0.02), np.full(5, -0.01)]
    weeks = np.repeat(np.arange(5), 5)
    lo, hi = E.cluster_boot(x, weeks, reps=500)
    assert lo <= x.mean() <= hi
