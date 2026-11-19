"""Tests for the 2026-11-19 overnight fixes: capital rule v2 (AG-28f), legacy XSP quarantine (AG-17q),
and the rebuilt SPY 2024-2026 option file."""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.index_branch as IB  # noqa: E402


# ---------------------------------------------------------------- capital rule v2
@pytest.mark.parametrize("capital,W", [(999, 0), (1000, 1), (2500, 2), (5000, 5), (10000, 10), (25000, 25)])
def test_capital_structure_width(capital, W):
    budget, w, struct = IB.capital_structure(capital)
    assert budget == pytest.approx(capital * 0.10)
    assert w == W
    if W < 1:
        assert struct is None
    else:
        assert struct == (0.10, ("minw", (float(W), 0.03)))


def _chain(d, exp, S, strikes, deltas):
    n = len(strikes)
    mid = np.linspace(0.05, 2.0, n)
    return pd.DataFrame({"date": d, "expiry": exp, "strike": strikes, "delta": deltas, "mid": mid,
                         "bid": mid * 0.95, "ask": mid * 1.05, "S": S, "iv": 0.2})


def _one_trade(W):
    d, exp = pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-21")
    S = 700.0
    strikes = np.arange(600.0, 700.0, 1.0)               # 600..699
    deltas = -0.001 - (strikes - 600) / 100 * 0.30       # 600 -> -0.001, 699 -> -0.298
    g = _chain(d, exp, S, strikes, deltas)
    by = {d: g}
    ix = g.set_index(["date", "expiry", "strike"]).sort_index()
    spot = pd.Series([S, S], index=[d, exp])
    return IB.trade(d, by, ix, spot, (0.10, ("minw", (float(W), 0.03))), "tp")


def test_minw_picks_dollar_leg_when_narrower():
    t = _one_trade(5)
    assert t is not None
    assert t["width"] == pytest.approx(5.0)


def test_minw_picks_delta_leg_when_narrower():
    # short ~ -0.10 delta sits ~33 strikes above 600 (633); the 0.03-delta put sits ~9-10 strikes above 600 (~610)
    t = _one_trade(500)                                   # $500 far wider than the 0.03-delta distance
    assert t is not None
    assert t["width"] < 30                                # the delta leg (~23 wide), not $500


def test_within_budget_filters_every_size():
    t = pd.DataFrame({"width": [5.0, 40.0, 25.0], "credit": [0.2, 0.5, 0.3]})
    kept = IB.within_budget(t, budget=2500)               # $2,500 budget -> max loss <= $2,875
    assert list(kept["width"]) == [5.0, 25.0]
    assert IB.within_budget(pd.DataFrame(), 100).empty


def test_capital_fix_preregistration_exists_and_is_hashed():
    p = ROOT / "strategy" / "preregistered_capital_fix.md"
    text = p.read_text(encoding="utf-8")
    assert "narrower_of_W_and_0.03d" in text and "```json" in text


# ---------------------------------------------------------------- legacy quarantine
def test_index_plan_is_quarantined():
    from src.agent.trade_desk import index_plan
    px = pd.DataFrame({"close": np.linspace(500, 600, 300)}, index=pd.bdate_range("2025-01-01", periods=300))
    with pytest.raises(RuntimeError, match="DEPRECATED"):
        index_plan(px.index[-1], px, 5000, 500)


def test_trade_desk_index_cli_refuses():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "trade_desk.py"), "index", "--capital", "5000",
                        "--risk-usd", "500"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0
    assert "DEPRECATED" in (r.stderr + r.stdout)


# ---------------------------------------------------------------- rebuilt SPY option file
SPY_FILE = ROOT / "data" / "processed" / "options" / "spy_puts_2024_2026.csv"


@pytest.mark.skipif(not SPY_FILE.exists(), reason="SPY option file not present (data/ is not in git)")
def test_rebuilt_spy_file_integrity():
    cols = ["date", "ticker", "volume", "open", "close", "high", "low", "ws", "tx"]
    p = pd.read_csv(SPY_FILE, header=None, names=cols, usecols=["date", "ticker", "close"])
    assert p["date"].max() >= "2026-10-28", "file ends early (the old copy was truncated at 2026-08-17)"
    assert p["date"].min() <= "2024-04-01"
    assert p["ticker"].str.match(r"^O:SPY\d{6}P\d{8}$").all()
    assert p["close"].notna().all()
    d = pd.to_datetime(pd.Series(p["date"].unique())).sort_values()
    gaps = d.diff().dt.days.dropna()
    assert gaps.max() <= 4, "a gap longer than a holiday weekend"
    assert len(d) >= 640
