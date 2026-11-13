import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src.backtest.engine import simulate_event
from src.backtest.grid import event_filters, event_mask, expand_grid, one_at_a_time, portfolio_curve, trade_metrics
from src.backtest.market import ProxyPricer
from src.backtest.walk_forward import quarter_folds, walk_forward


def test_expand_grid_and_one_at_a_time():
    space = {"a": [1, 2, 3], "b": ["x", "y"], "c": [True, False]}
    assert len(expand_grid(space, {}, ["a", "b"])) == 6
    oat = one_at_a_time(space, {"a": 2, "b": "x", "c": True})
    assert [lab for lab, _ in oat] == ["baseline", "a=1", "a=3", "b=y", "c=False"]


def test_event_mask_and_filters():
    ev = pd.DataFrame({"verdict_gpt4o": ["not justified", "justified", "not justified"],
                       "verdict_opus": ["not justified", "justified", np.nan],
                       "category_gpt4o": ["lawsuit", "M&A", "M&A"], "category_opus": ["lawsuit", "M&A", np.nan],
                       "n_headlines": [3, 2, 0], "ratio_opus": [4.0, 1.0, np.nan],
                       "abnormal": [-0.12, -0.06, -0.07], "market_cap": [30e9, 8e9, 6e9]})
    assert event_mask(ev, "nj_both").tolist() == [True, False, False]
    assert event_mask(ev, "j_both").tolist() == [False, True, False]
    assert event_mask(ev, "all", max_shock=-0.10).tolist() == [True, False, False]
    assert event_mask(ev, "all", mcap_floor=20e9).tolist() == [True, False, False]
    assert event_mask(ev, "all", min_ratio=3).tolist() == [True, False, False]
    assert len(event_filters({"event_set": ["all", "nj_both"], "max_shock": [-0.05, -0.1]})) == 4


def _trades(q_ror):
    rows = []
    for q, (cand, ror) in q_ror:
        start = pd.Period(q, freq="Q").start_time
        for i in range(25):
            d = start + pd.Timedelta(days=int(i))
            rows.append({"candidate": cand, "entry_date": d, "exit_date": d + pd.Timedelta(days=10),
                         "ror": ror, "pnl": ror, "days_held": 10, "ric": f"R{i}"})
    return pd.DataFrame(rows)


def test_quarter_folds_rolling_and_expanding():
    q = ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]
    assert quarter_folds(q, train=2) == [(["2024Q1", "2024Q2"], "2024Q3"), (["2024Q2", "2024Q3"], "2024Q4")]
    assert quarter_folds(q, train=2, expanding=True)[1] == (["2024Q1", "2024Q2", "2024Q3"], "2024Q4")


def test_walk_forward_picks_on_train_and_scores_on_test():
    # A wins in training quarters, B wins in the test quarter: the pick must be A, scored on its test result
    t = _trades([("2024Q1", ("A", 0.2)), ("2024Q1", ("B", -0.1)),
                 ("2024Q2", ("A", -0.3)), ("2024Q2", ("B", 0.5))])
    table, oos = walk_forward(t, "candidate", quarter_folds(["2024Q1", "2024Q2"], train=1), baseline="B")
    assert table["chosen"].tolist() == ["A"]
    assert np.isclose(table["test_mean_ror"].iloc[0], -0.3) and np.isclose(table["base_mean_ror"].iloc[0], 0.5)
    assert (oos["candidate"] == "A").all()


def test_walk_forward_purges_trades_open_into_test_quarter():
    t = _trades([("2024Q1", ("A", 0.2)), ("2024Q2", ("A", 0.1))])
    t.loc[t["candidate"] == "A", "exit_date"] = pd.Timestamp("2024-04-15")  # every Q1 trade ends in Q2
    table, _ = walk_forward(t, "candidate", quarter_folds(["2024Q1", "2024Q2"], train=1), min_trades=1)
    assert table.empty


def test_trade_metrics_and_portfolio():
    t = _trades([("2024Q1", ("A", 0.1))])
    m = trade_metrics(t)
    assert m["n"] == 25 and m["win_rate"] == 1.0 and np.isclose(m["mean_ror"], 0.1)
    taken, equity, st = portfolio_curve(t, {f"R{i}": f"S{i}" for i in range(25)}, capital=100_000, risk_pct=0.02)
    _, _, capped = portfolio_curve(t, {}, capital=100_000, risk_pct=0.02)  # one sector: the 30% cap binds
    assert capped["n_blocked"] > 0
    assert st["n_taken"] == 25 and np.isclose(st["total_return"], 25 * 0.1 * 2000 / 100_000)


def _stock(path):
    idx = pd.bdate_range("2024-01-01", periods=len(path))
    close = pd.Series(path, index=idx, dtype=float)
    return pd.DataFrame({"close": close, "low": close * 0.99, "high": close * 1.01,
                         "volume": 1e6, "ret": close.pct_change()})


def test_simulate_event_rally_takes_profit_and_crash_stops():
    rng = np.random.default_rng(0)
    pre = list(100 * np.cumprod(1 + rng.normal(0, 0.02, 80)))
    rates = pd.Series(0.04, index=pd.bdate_range("2023-01-01", "2025-01-01"))
    params = {"min_credit_pct": 0.05, "iv_rv_min": 0.0, "max_volume_mult": None}
    for tail, expect in [(np.linspace(1.0, 1.15, 40), "take profit"), (np.linspace(1.0, 0.6, 40), "stop")]:
        path = pre + [pre[-1] * 0.9] + list(pre[-1] * 0.9 * tail)
        st = _stock(path)
        shock = st.index[80]
        pr = ProxyPricer(rates)
        pr.start("E", st, shock)
        r = simulate_event({"event_id": "E", "ric": "X", "date": shock}, st, pr, [], params)
        assert r["status"] == "traded" and r["exit_reason"].startswith(expect), r
        assert r["short_k"] < st["low"].iloc[80:82].min() and r["long_k"] < r["short_k"]
        assert -1.0 <= r["ror"] <= 1.0


def test_simulate_event_skips_when_earnings_in_window():
    rng = np.random.default_rng(1)
    pre = list(100 * np.cumprod(1 + rng.normal(0, 0.02, 80)))
    st = _stock(pre + [pre[-1] * 0.9] * 41)
    shock = st.index[80]
    pr = ProxyPricer(pd.Series(0.04, index=pd.bdate_range("2023-01-01", "2025-01-01")))
    pr.start("E", st, shock)
    r = simulate_event({"event_id": "E", "ric": "X", "date": shock}, st, pr, [st.index[90]],
                       {"min_credit_pct": 0.0, "iv_rv_min": 0.0, "max_volume_mult": None})
    assert r["status"] == "earnings in window"


def test_simulate_event_closes_spread_when_between_strikes_at_expiry():
    rng = np.random.default_rng(2)
    pre = list(100 * np.cumprod(1 + rng.normal(0, 0.02, 80)))
    rates = pd.Series(0.04, index=pd.bdate_range("2023-01-01", "2025-01-01"))
    params = {"min_credit_pct": 0.0, "iv_rv_min": 0.0, "max_volume_mult": None,
              "tp_mode": "hold", "stop_rules": []}
    flat = pre + [pre[-1] * 0.9] * 60
    st = _stock(flat)
    shock = st.index[80]
    pr = ProxyPricer(rates)
    pr.start("E", st, shock)
    r0 = simulate_event({"event_id": "E", "ric": "X", "date": shock}, st, pr, [], params)
    assert r0["status"] == "traded"
    # same entry, but the stock ends between the strikes on expiry day
    mid_k = (r0["short_k"] + r0["long_k"]) / 2
    path = list(st["close"])
    e = st.index.get_loc(r0["entry_date"])
    x = st.index.get_loc(r0["expiry"])
    path[e + 1:x + 1] = list(np.linspace(path[e], mid_k, x - e))
    st2 = _stock(path)
    for close_it, reason, legs in [(True, "closed at expiry (between strikes)", 4), (False, "expiry", 2)]:
        pr = ProxyPricer(rates)
        pr.start("E", st2, shock)
        r = simulate_event({"event_id": "E", "ric": "X", "date": shock}, st2, pr, [],
                           {**params, "close_between_at_expiry": close_it})
        assert r["entry_date"] == r0["entry_date"] and r["exit_reason"] == reason, r
        intrinsic = r["short_k"] - mid_k
        assert r["exit_cost"] >= intrinsic - 1e-9
        assert np.isclose(r["pnl"], r["credit"] - r["exit_cost"] - 0.65 * legs / 100)
