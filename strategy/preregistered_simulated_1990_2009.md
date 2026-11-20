# Pre-registered test: the index-branch rule on SIMULATED SPX option prices, 1990-2009 (written 2026-11-19 ~17:40 CST, before any run)

Why: real option quotes before 2010 are not available to us (LSEG expired-option history starts ~2014-2016; WRDS not yet
accessible). This test asks whether the rule would have survived 1990-2009 (1998, 2000-2002, 2008) when option prices
are MODELLED from VIX and a volatility smile calibrated on our real 2010-2023 chains. It is evidence, not proof, and it
can never count as passing the real-data tests. Code: scripts/sim_backtest.py (AG-64..66).

## Data
- S&P 500 daily closes 1988-2010 (LSEG .SPX, unadjusted), VIX daily closes 1990-2026 (Cboe, via the public
  datasets/finance-vix mirror). Calibration chains: Kaggle SPY 2010-2023 (real bid/ask).

## Price model (fitted ONLY on 2010-2023, then frozen)
- ATM put IV (~21 DTE) = VIX/100 x k(regime); regime = VIX < 16 / 16-24 / > 24; k = median ratio in that regime.
- Smile: IV(K) = ATM IV x s(z, regime), z = ln(K/S) / (ATM IV x sqrt(T)); s = median IV ratio per z bin
  (bins of 0.25 from -3.5 to 0), linear interpolation, flat beyond the ends.
- Bid/ask: half-spread = max(m(t) x h(z, regime) x mid, tick(t)); h = median half-spread / mid per z bin and regime;
  m = 2.0 before 2001-04-09 (fractional pricing), 1.5 to 2004-12-31, 1.0 after; tick = 0.0625 SPX points before
  2001-04-09, 0.05 after. Fee $0.65 per contract per leg (0.0065 SPX points).
- Black-Scholes (r = 0, no dividends) for prices and deltas; strikes on a 5-point SPX grid; expiries = monthly third
  Fridays only (weeklies did not exist before 2005), settled at that Friday's close.

## Rule (the walk-forward rule; scale-free delta structure)
Shock: SPX 1-day <= -1.5% or 2-day <= -2.5%. Entry day +1..+3, first close above the post-shock low. Guard: SPX above
its 200-day average (the 9-day vs 30-day IV leg cannot be computed before 2011 and is DROPPED; the same MA-only guard is
used in the calibration check). Expiry 14-35 days nearest 21. Short put nearest -0.10 delta, long nearest -0.03 delta.
Dynamic take-profit on the modelled mid (buy back at mid + half-spreads), else expiry; no stop. One spread at a time;
10% of capital at risk per trade for account statistics.

## Step 1: calibration check (must hold for step 2 to be called usable)
Run the simulator over 2010-2023 with the same MA-only rule and compare with the same rule on the REAL chains
(scripts/index_branch.py trade function, monthly expiries only): |simulated mean ror - real mean ror| <= 1.0 percentage
point AND the per-trade Spearman correlation on common entry dates >= 0.5. If either fails, step 2 is reported as
"model not validated".

## Step 2: 1990-2009 (primary)
A. The 4 checks: mean ror > 0 with 95% bootstrap interval above 0; >= 70% of years with trades positive; worst month
   better than -25%; max drawdown better than -30% (10% risk per trade, one at a time).
B. Permutation test as strategy/preregistered_permutation_test.md: 10,000 runs, same number of entries on random
   tradable days (Null A, primary) and on guard-on days (Null B); p = (1 + k) / (1 + 10,000); seed 20261119.
Simulated result PASSES if step 1 holds, all four checks pass AND Null A p < 0.05. Also reported: number of full
losses, every crash window (1997-10, 1998-08..10, 2000-2002, 2007-08, 2008-09..2009-03), years table.

```json
{"test": "simulated_1990_2009", "calib": ["2010-01-01", "2023-12-31"], "period": ["1990-01-01", "2009-12-31"],
 "regimes": [16, 24], "z_bins": [-3.5, 0, 0.25], "spread_mult": {"2001-04-09": 2.0, "2005-01-01": 1.5, "after": 1.0},
 "tick": {"before_2001-04-09": 0.0625, "after": 0.05}, "strike_grid": 5, "short_delta": 0.10, "long_delta": 0.03,
 "dte": [14, 35, 21], "guard": "ma200_only", "risk": 0.10, "n_perm": 10000, "seed": 20261119, "alpha": 0.05,
 "calib_check": {"max_mean_gap": 0.01, "min_spearman": 0.5}}
```

## Addendum (2026-11-19 ~17:55 CST, after step 1 ran, BEFORE step 2 ran)
Data substitution only, no rule change: if data/raw/market_context/spx_1988_2010.parquet is not available, the SPX
level is taken as 10 x the unadjusted SPY close (LSEG, data/raw/market_context/spy.parquet), which starts 1993-01-29.
The tested period then starts once the 200-day average exists (~1993-11) instead of 1990. Step 1 result, already
known when this was written: MODEL NOT VALIDATED (gap -1.19 pp > 1.0; Spearman 0.42 < 0.5), so step 2 is reported as
"model not validated" whatever it shows.
