"""Tests for scripts/portfolio_overlay.py (AG-56..60): T-bill account math and tail stress."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.portfolio_overlay as PO  # noqa: E402

DAYS = pd.bdate_range("2024-01-01", "2024-12-31")


def test_tbill_only_compounds_act360():
    rate = pd.Series(0.05, index=DAYS)
    eq = PO.account(None, 10000, DAYS, rate, with_spreads=False)
    days = (DAYS[-1] - DAYS[0]).days + 1
    assert eq.iloc[-1] == pytest.approx(10000 * (1 + 0.05 / 360) ** days, rel=2e-3)


def test_collateral_earns_nothing_and_pnl_booked_on_exit():
    rate = pd.Series(0.05, index=DAYS)
    t = pd.DataFrame({"date": [pd.Timestamp("2024-03-01")], "exit": [pd.Timestamp("2024-03-22")],
                      "collateral": [1000.0], "pnl_usd": [30.0]})
    a = PO.account(t, 10000, DAYS, rate)
    b = PO.account(None, 10000, DAYS, rate, with_spreads=False)
    jump = a.loc["2024-03-22"] - a.loc["2024-03-21"]
    assert jump == pytest.approx(30 + (a.loc["2024-03-21"] - 1000) * 0.05 / 360 * 1, rel=0.05) or jump > 29
    # less interest while 1,000 is locked: before the exit the overlay is below T-bills only
    assert a.loc["2024-03-21"] < b.loc["2024-03-21"]
    assert a.iloc[-1] > b.iloc[-1] + 25          # +30 P&L minus ~3 weeks of interest on $1,000


def test_stats_keys_and_drawdown():
    eq = pd.Series(np.r_[np.linspace(100, 110, 100), np.linspace(110, 99, 50), np.linspace(99, 120, 112)],
                   index=DAYS[:262])
    st = PO.stats(eq, 100)
    assert st["maxdd"] == pytest.approx(99 / 110 - 1, rel=1e-6)
    assert set(st) >= {"cagr", "total", "maxdd", "worst_month", "yearly", "monthly"}


def test_stress_no_losses_seen_still_has_a_tail():
    ror = np.full(50, 0.02)
    s = PO.stress_param(ror, 0.10, 4, n_seen=50, years=20000)
    assert s["mean"] < 4 * 0.02 * 0.10                 # full losses drawn from the posterior pull the mean down
    assert s["cvar1"] < -0.05                          # 1% tail includes at least one full loss of 10%
    assert 0 < s["p_full_loss_mean"] < 0.05


def test_stress_more_evidence_shrinks_the_tail():
    ror = np.full(50, 0.02)
    few = PO.stress_param(ror, 0.10, 4, n_seen=20, years=20000)
    many = PO.stress_param(ror, 0.10, 4, n_seen=2000, years=20000)
    assert many["p_full_loss_mean"] < few["p_full_loss_mean"]
    assert many["p_loss_year"] < few["p_loss_year"]


def test_overlay_preregistration_parses():
    spec, sha = PO.load_spec()
    assert len(sha) == 16 and spec["checks"]["n_trials"] == 200 and spec["fill_model"]["skip"] == 0.15
