# Pre-registered test: agent ablation on post-cutoff shocks (written 2026-11-19 ~02:40 CST, before any run)

Question: do the agent's cached GPT-4.1 verdicts, alone or with the fair-value (valuation) gate, pick single-stock
shock trades that beat the rest OUT OF SAMPLE? Offline: no new LLM calls; uses the cached ag-v1 verdicts.
Code: `scripts/agent_diag.py --ablation` (AG-53). Results: `data/results/agent_diag/ablation.*`.

## Data (fixed)
- Trades: `data/results/agent/signal_trades.csv` (one 0.15-delta put spread per shock, priced from real option closes
  by scripts/score_signals.py; outcome = `ror`, return on the money at risk), joined on event_id to
  `data/results/agent/agent_results.csv` (verdicts: agent_take, confidence, post_cutoff, status).
- Universe: rows with post_cutoff == True (shock after GPT-4.1's June 2024 training cutoff) and status == "ok".
  Earlier shocks are excluded because the model may remember them.
- Split: in-sample = shock date before 2026-06-01; hold-out = shock date on/after 2026-06-01 (same date as
  score_signals.py --holdout). The hold-out is scored once.

## Hypotheses (filter vs complement), thresholds fixed in advance (0.7 comes from the task statement, not tuning)
- H1: among agent TAKEs with fundamentals (fv_has), the valuation gate helps: filter = TAKE & fv_pass,
  complement = TAKE & not fv_pass.
- H2: high-confidence TAKEs beat the rest: filter = TAKE & confidence >= 0.7, complement = every other trade.
- H3: TAKE plus the valuation gate beats the rest: filter = TAKE & fv_pass, complement = every other trade.

## Statistics
- Mean ror per group; 95% percentile bootstrap interval, 10,000 resamples CLUSTERED BY ISO WEEK of the shock date
  (weeks resampled with replacement, all trades of a drawn week kept), seed 0. Same for the difference
  (filter - complement), resampling weeks jointly.
- Deflated Sharpe ratio (src/backtest/overfit.py) of each filter's per-trade ror over the full post-cutoff sample,
  n_trials = 10 (these 3 + the 7 filters already scored on this data: agent TAKE, credit/width top and bottom third,
  no news, IV/RV top third, biggest drops, fair-value gate), sr_trials_var = variance of the 3 filters' per-trade
  Sharpes here, skew/kurtosis of the filter's own returns.

## Pass rule (each hypothesis separately)
PASS only if, on the HOLD-OUT: (a) the filter has at least 10 trades, (b) the lower end of its week-clustered 95%
interval for mean ror is above 0, and (c) its mean ror is above the complement's. Anything else = FAIL.
In-sample results and DSR are reported, not scored. No re-run with changed thresholds or dates.

```json
{"universe": {"post_cutoff": true, "status": "ok"}, "holdout_from": "2026-06-01",
 "hypotheses": {"H1": {"filter": "take & fv_pass", "complement": "take & ~fv_pass", "base": "fv_has"},
                "H2": {"filter": "take & confidence>=0.7", "complement": "rest"},
                "H3": {"filter": "take & fv_pass", "complement": "rest"}},
 "bootstrap": {"n": 10000, "cluster": "iso_week", "seed": 0}, "dsr": {"n_trials": 10},
 "pass": {"min_n_holdout": 10, "ci_lo_gt": 0, "beats_complement": true}}
```
