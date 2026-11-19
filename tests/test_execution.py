"""Tests for the order-placing agent (src/agent/execution.py, AG-37..AG-40). Run: python -m pytest tests/test_execution.py"""
import itertools

import pandas as pd
import pytest

from src.agent import execution as X
from strategy.strategy_logic import should_take_profit

Q = dict(sb=0.62, sa=0.70, lb=0.38, la=0.44)          # mid 0.25, natural (open) 0.18, natural (close) 0.32


def test_fee_matches_backtest():
    import scripts.index_branch as IB
    assert X.FEE == IB.FEE


def test_quotes():
    mid, nat = X.spread_quotes(**Q, side="open")
    assert mid == pytest.approx(0.25) and nat == pytest.approx(0.18)
    mid, nat = X.spread_quotes(**Q, side="close")
    assert mid == pytest.approx(0.25) and nat == pytest.approx(0.32)
    with pytest.raises(ValueError):
        X.spread_quotes(0.5, 0.4, 0.1, 0.2)            # ask below bid


def test_open_ladder_monotonic_bounded_rounded():
    lad = X.ladder(**Q, side="open", steps=4, max_concession=0.5)
    px = [s["limit"] for s in lad["steps"]]
    assert px[0] == pytest.approx(0.25)                 # starts at the mid
    assert all(a > b for a, b in zip(px, px[1:]))       # credit only goes down
    assert min(px) >= 0.25 - 0.5 * 0.07 - 1e-9          # never more than halfway to the natural 0.18
    assert all(abs(p * 100 - round(p * 100)) < 1e-6 for p in px)   # pennies
    assert [s["at_min"] for s in lad["steps"]] == [0, 5, 10, 15][:len(px)]
    assert lad["cancel_at_min"] == lad["steps"][-1]["at_min"] + 5
    assert all(s["net"] == pytest.approx(s["limit"] - 2 * X.FEE) for s in lad["steps"])


def test_close_ladder_monotonic_bounded_and_capped():
    lad = X.ladder(**Q, side="close", steps=4, max_concession=0.5)
    px = [s["limit"] for s in lad["steps"]]
    assert px[0] == pytest.approx(0.25) and all(a < b for a, b in zip(px, px[1:]))
    assert max(px) <= 0.25 + 0.5 * 0.07 + 1e-9
    capped = X.ladder(**Q, side="close", steps=4, max_concession=1.0, max_debit=0.27)
    assert max(s["limit"] for s in capped["steps"]) <= 0.27


@pytest.mark.parametrize("sb,sa,lb,la,mc,steps,tick", [
    (sb, sb + w1, lb, lb + w2, mc, n, t)
    for sb, lb, w1, w2, mc, n, t in itertools.product([0.31, 1.07, 4.4], [0.05, 0.2], [0.01, 0.03, 0.1],
                                                       [0.01, 0.05], [0.0, 0.5, 1.0], [1, 3, 5], [0.01, 0.05])])
def test_ladder_bounds_property(sb, sa, lb, la, mc, steps, tick):
    for side in ("open", "close"):
        lad = X.ladder(sb, sa, lb, la, side=side, steps=steps, max_concession=mc, tick=tick)
        mid, nat = lad["mid"], lad["natural"]
        gap = abs(mid - nat)
        for s in lad["steps"]:
            k = s["limit"] / tick
            assert abs(k - round(k)) < 1e-6
            if side == "open":
                assert s["limit"] >= mid - mc * gap - 1e-9
                assert s["limit"] <= mid + tick + 1e-9
            else:
                assert s["limit"] <= mid + mc * gap + 1e-9
                assert s["limit"] >= max(mid - tick, 0) - 1e-9
        assert len({s["limit"] for s in lad["steps"]}) == len(lad["steps"])


def test_min_credit_skip():
    tiny = X.ladder(0.02, 0.03, 0.01, 0.02, side="open")     # mid 0.01: fees (0.013) eat it
    assert tiny["skip"] and tiny["steps"] == []
    part = X.ladder(**Q, side="open", min_credit=0.22)      # only steps with credit - fees > 0.22 stay
    assert part["steps"] and all(s["net"] > 0.22 for s in part["steps"]) and part["steps"][0]["step"] == 1
    assert "NO ORDER" in X.ticket(tiny, "XSP", "2026-12-11", 655, 650, 1)


@pytest.mark.parametrize("credit", [0.17, 0.25, 0.61, 1.33, 2.0])
@pytest.mark.parametrize("held", [0, 1, 3, 5, 8, 12, 15, 20])
def test_take_profit_price_matches_rule(credit, held):
    opened, expiry = pd.Timestamp("2026-11-20"), pd.Timestamp("2026-12-11")      # 21 days
    tp = X.take_profit_price(credit, opened, expiry, opened + pd.Timedelta(days=held))
    dte = 21
    args = (held, dte, "dynamic", 0.50, 0.20, 0.75)
    assert should_take_profit(credit, tp["limit"], *args)                 # a fill at the limit closes per the rule
    above = round(tp["limit"] + 0.01, 2)
    if above < credit:
        assert not should_take_profit(credit, above, *args) or tp["limit"] == 0   # the highest such penny price
    assert 0.25 - 1e-9 <= 1 - tp["limit"] / credit or tp["limit"] == 0


def test_take_profit_none_on_expiry_day():
    assert X.take_profit_price(0.5, "2026-11-20", "2026-12-11", "2026-12-11") is None


def test_fill_quality_scale():
    assert X.fill_quality(0.25, 0.25, 0.18, "open") == pytest.approx(0.0)
    assert X.fill_quality(0.18, 0.25, 0.18, "open") == pytest.approx(1.0)
    assert X.fill_quality(0.215, 0.25, 0.18, "open") == pytest.approx(0.5)
    assert X.fill_quality(0.26, 0.25, 0.18, "open") < 0                  # better than mid
    assert X.fill_quality(0.32, 0.25, 0.32, "close") == pytest.approx(1.0)
    assert X.fill_quality(0.25, 0.25, 0.32, "close") == pytest.approx(0.0)
    assert X.fill_quality(0.10, 0.10, 0.10, "open") == 0.0               # no spread


def test_log_fill_and_report(tmp_path):
    sens = tmp_path / "sens.csv"
    pd.DataFrame({"capital": [2500] * 4 + [5000] * 4, "fill": [1.0, 0.5, 0.25, 0.0] * 2,
                  "per_year_$": [4, 8, 13, 19, 22, 32, 36, 41],
                  "per_year_%": [0.0016, 0.0032, 0.0052, 0.0076, 0.0044, 0.0064, 0.0072, 0.0082]}).to_csv(sens, index=False)
    fills = tmp_path / "fills.csv"
    X.log_fill("t1", "open", "XSP", "2026-12-11", 655, 650, 1, **Q, limit=0.23, fill=0.23, step=2, path=fills)
    X.log_fill("t1", "close", "XSP", "2026-12-11", 655, 650, 1, **Q, limit=0.25, fill=0.25, step=1, path=fills)
    f = pd.read_csv(fills)
    assert list(f.columns) == X.FILL_COLS and len(f) == 2
    assert f["fill_q"].tolist() == pytest.approx([0.02 / 0.07, 0.0], abs=1e-3)
    r = X.report(5000, path=fills, sens_path=sens)
    q = (0.02 / 0.07) / 2
    assert r["n"] == 2 and r["fill_q"] == pytest.approx(q, abs=1e-3)
    assert r["per_year_$"] == pytest.approx(pd.Series([41, 36, 32, 22]).pipe(
        lambda s: __import__("numpy").interp(q, [0, 0.25, 0.5, 1.0], s)), abs=0.05)
    assert X.report(4000, path=fills, sens_path=sens)["capital_used"] == 5000      # nearest capital
    assert X.report(5000, path=tmp_path / "none.csv", sens_path=sens) == {"n": 0}


def test_value_of_fill_clips():
    import pathlib
    p = pathlib.Path("data/results/index_branch/fill_sensitivity.csv")
    if not p.exists():
        pytest.skip("fill sensitivity results not present")
    lo, hi = X.value_of_fill(-0.3, 2500, p), X.value_of_fill(1.7, 2500, p)
    assert lo["per_year_$"] == pytest.approx(lo["at_mid_$"]) and hi["per_year_$"] == pytest.approx(hi["at_worst_$"])


def test_qty_for_risk():
    assert X.qty_for_risk(5000, 5, 0.20) == 1          # max loss $480 <= $500
    assert X.qty_for_risk(2500, 5, 0.20) == 0          # $480 > $250
    assert X.qty_for_risk(25000, 5, 0.20) == 5
    with pytest.raises(ValueError):
        X.qty_for_risk(5000, 1, 1.2)


def test_ticket_text():
    lad = X.ladder(**Q, side="open")
    t = X.ticket(lad, "XSP", "2026-12-11", 655, 650, 2, "XSP-1")
    assert "SELL 2 x XSP put spread 655/650" in t and "combo" in t and "TEAMMATE MUST APPROVE" in t
    assert sum(line.strip().startswith("step ") for line in t.splitlines()) == len(lad["steps"])
    assert "cancel at" in t
