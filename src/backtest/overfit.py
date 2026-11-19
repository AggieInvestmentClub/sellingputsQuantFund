"""
AG-21 / AG-22: overfitting checks for the grid and walk-forward results.

deflated_sharpe   probability the best Sharpe is real after trying N settings (Bailey & Lopez de Prado 2014)
pbo_cscv          probability of backtest overfitting via combinatorially symmetric CV (Bailey et al. 2015)
Both take plain numbers / arrays so they work on any trades table (one column per setting).
"""
import itertools
import math

import numpy as np
from scipy.stats import norm

EULER = 0.5772156649015329


def deflated_sharpe(sr, n_obs, n_trials, sr_trials_var, skew=0.0, kurt=3.0):
    """
    [AG-21] deflated Sharpe ratio: P(true Sharpe > 0) for the BEST of n_trials settings

    sr: the best setting's Sharpe, per observation (not annualized); n_obs: its number of returns
    n_trials: settings tried; sr_trials_var: variance of the per-observation Sharpes across those settings
    skew / kurt: of the best setting's returns (kurt = 3 for normal)
    returns a probability; < 0.95 means the best result is not distinguishable from luck
    """
    if n_obs < 2 or n_trials < 1:
        return float("nan")
    # expected max Sharpe of n_trials unskilled settings
    sd = math.sqrt(max(sr_trials_var, 0.0))
    if n_trials == 1:
        sr0 = 0.0
    else:
        sr0 = sd * ((1 - EULER) * norm.ppf(1 - 1 / n_trials) + EULER * norm.ppf(1 - 1 / (n_trials * math.e)))
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr0) * math.sqrt(n_obs - 1) / denom))


def deflated_sharpe_from_matrix(returns):
    """
    [AG-21b] deflated Sharpe from a (n_obs x n_settings) matrix of per-period returns
    returns dict: best (column index), sr_best (per period), dsr, n_trials
    """
    R = np.asarray(returns, dtype=float)
    R = R[~np.isnan(R).any(axis=1)]
    mu, sd = R.mean(axis=0), R.std(axis=0, ddof=1)
    sr = np.where(sd > 0, mu / sd, 0.0)
    b = int(np.argmax(sr))
    x = R[:, b]
    z = (x - x.mean()) / (x.std(ddof=1) or 1)
    return {"best": b, "sr_best": float(sr[b]), "n_trials": R.shape[1],
            "dsr": deflated_sharpe(sr[b], len(x), R.shape[1], float(np.var(sr, ddof=1)) if R.shape[1] > 1 else 0.0,
                                   float((z ** 3).mean()), float((z ** 4).mean()))}


def pbo_cscv(returns, n_splits=8, metric=None):
    """
    [AG-22] probability of backtest overfitting (CSCV)

    returns: (n_obs x n_settings) per-period returns in time order; n_splits: even number of blocks
    For every way of choosing half the blocks as in-sample, pick the best setting in-sample and see where it
    ranks out of sample. PBO = share of splits where it lands in the bottom half. > 0.5 = worse than a coin flip.
    returns dict: pbo, n_combos, logits (list)
    """
    R = np.asarray(returns, dtype=float)
    if n_splits % 2 or n_splits < 2:
        raise ValueError("n_splits must be an even number >= 2")
    metric = metric or (lambda m: np.divide(m.mean(0), m.std(0, ddof=1), out=np.zeros(m.shape[1]),
                                            where=m.std(0, ddof=1) > 0))
    blocks = np.array_split(np.arange(len(R)), n_splits)
    logits = []
    for ins in itertools.combinations(range(n_splits), n_splits // 2):
        i_idx = np.concatenate([blocks[k] for k in ins])
        o_idx = np.concatenate([blocks[k] for k in range(n_splits) if k not in ins])
        best = int(np.argmax(metric(R[i_idx])))
        oos = metric(R[o_idx])
        rank = (oos < oos[best]).sum() + 1                     # 1 = worst
        w = rank / (R.shape[1] + 1)
        logits.append(math.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_combos": len(logits), "logits": logits.tolist()}
