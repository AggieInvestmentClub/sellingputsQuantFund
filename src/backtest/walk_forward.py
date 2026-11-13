"""
Quarterly tables and walk-forward selection over (trade combination x event filter) candidates.

Walk-forward: for each test quarter, pick the candidate that scored best on the previous
`train` quarters, then record how that pick did in the test quarter. Only the test-quarter
results are out-of-sample; stitched together they are the honest estimate of the strategy.
"""
import numpy as np
import pandas as pd

from src.backtest.grid import trade_metrics


def quarterly_table(trades, by):
    """
    [BT-11] BT-09 metrics per group and entry quarter

    trades: rows with entry_date, ror, pnl, days_held and the `by` columns
    returns DataFrame indexed by (*by, quarter), e.g. by=["label"] gives every parameter variant x quarter
    """
    t = trades.assign(quarter=pd.PeriodIndex(pd.to_datetime(trades["entry_date"]), freq="Q").astype(str))
    return t.groupby([*by, "quarter"]).apply(trade_metrics, include_groups=False)


def quarter_folds(quarters, train=4, expanding=False):
    """
    [BT-12] walk-forward folds: train on `train` quarters (or every earlier quarter if expanding),
    test on the next one

    quarters: sorted list like ["2024Q2", "2024Q3", ...]
    returns list of (train_quarters, test_quarter); the first `train` quarters are never tested
    """
    q = list(quarters)
    return [(q[:i] if expanding else q[i - train:i], q[i]) for i in range(train, len(q))]


def walk_forward(trades, candidate_col, folds, score="mean_ror", min_trades=20, baseline=None):
    """
    [BT-13] pick the best candidate on each training window, score it on the next quarter

    trades: all simulated trades, one row per (candidate, event) with candidate_col, entry_date,
    exit_date, ror, pnl, days_held. A training window only uses trades that EXITED before the test
    quarter starts (purge: a trade still open would leak test-quarter prices into the choice).
    score: a BT-09 column to maximize; candidates with fewer than min_trades in training are skipped.
    baseline: candidate to report next to the pick in each test quarter (None = skip)
    returns (per-fold DataFrame: test quarter, chosen candidate, train score, test metrics, baseline
    test metrics; out-of-sample trades of the chosen candidates, stitched)
    """
    t = trades.dropna(subset=["ror"]).copy()
    t["quarter"] = pd.PeriodIndex(pd.to_datetime(t["entry_date"]), freq="Q").astype(str)
    rows, oos = [], []
    for train_q, test_q in folds:
        test_start = pd.Period(test_q, freq="Q").start_time
        tr = t[t["quarter"].isin(train_q) & (pd.to_datetime(t["exit_date"]) < test_start)]
        if tr.empty:
            continue
        stats = tr.groupby(candidate_col).apply(trade_metrics, include_groups=False)
        stats = stats[stats["n"] >= min_trades]
        if stats.empty or stats[score].isna().all():
            continue
        best = stats[score].idxmax()
        te = t[(t["quarter"] == test_q) & (t[candidate_col] == best)]
        m = trade_metrics(te)
        row = {"test_quarter": test_q, "train": f"{train_q[0]}..{train_q[-1]}", "chosen": best,
               "train_n": stats.at[best, "n"], f"train_{score}": stats.at[best, score],
               "n_candidates": len(stats), **{f"test_{k}": v for k, v in m.items()}}
        if baseline is not None:
            b = trade_metrics(t[(t["quarter"] == test_q) & (t[candidate_col] == baseline)])
            row.update({f"base_{k}": b[k] for k in ["n", "mean_ror", "win_rate", "worst5_ror"]})
        rows.append(row)
        oos.append(te)
    return pd.DataFrame(rows), (pd.concat(oos, ignore_index=True) if oos else pd.DataFrame())
