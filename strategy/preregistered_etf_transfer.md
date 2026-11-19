# Pre-registered test: does the SPY hindsight-best configuration transfer to other ETFs? (written 2026-11-19 ~02:25 CST)

Why: the return branch (strategy/preregistered_return_branch.md) FAILED its walk-forward, but its best full-period
configuration on SPY (found in HINDSIGHT) was T10+guard, short 0.15 / long 0.05 delta, ~21 DTE: CAGR +3.8% at 20% risk,
max DD -5.4%. A hindsight pick can only be tested on data it has not seen. The 28 non-SPY ETFs (2024-04..2026-10) were
never used to choose it. This is that test, run ONCE.

## What was already seen (honesty)
- The ETF branch (strategy/preregistered_etf_branch.md) with the 0.10/0.03-delta, -1.5% rule on these same 28 ETFs:
  FAIL, with losses concentrated in QQQ, XLE and ARKK. So this sample is not fully fresh for the ETFs; it is fresh for
  THIS configuration (threshold -1.0%, short 0.15, long 0.05).
- SPY is excluded from this test (the configuration was chosen on SPY).

## Rule (everything else identical to the ETF branch spec and code, scripts/etf_branch.py)
- Shock: 1-day <= -1.0% x s or 2-day <= -2.0% x s, s = sigma_ETF(t) / sigma_SPY(t) (252-day, adjusted closes).
- Guard: adjusted close above its 200-day average and ~9-day ATM put IV not above ~30-day.
- Entry day +1..+3, first close above the post-shock low. 14-35 DTE nearest 21.
- Short put nearest -0.15 delta; long = NARROWER of (strike nearest short - W) and the put nearest -0.05 delta,
  W = floor(capital x 10% / 100).
- Exit: dynamic take-profit, else expiry; no stop. Fills at bid/ask with the ETF branch's conservative spreads.
- Portfolio: one spread per ETF, 10% risk per trade, total open risk <= 30% (NOT the 20% / 40% of the SPY pick: the
  smaller risk is the conservative choice and makes the result comparable to the ETF branch).
- Capitals $2,500, $5,000, $10,000; primary $5,000 (the smallest size where the index branch is clearly positive).

## Pass / fail (the ETF branch's 4 checks, at the primary capital)
1. Mean return on risk > 0 with the entry-week-clustered 95% bootstrap interval above 0.
2. Account positive over the period. 3. Max drawdown better than -30%. 4. Worst month better than -25%.
Label: evidence, not proof (~2.5 years). A FAIL means the SPY hindsight pick is not adopted anywhere.

```json
{
  "branch": "etf_transfer",
  "written": "2026-11-19 02:25 CST",
  "roots": ["QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "SMH", "XBI",
            "KRE", "GLD", "SLV", "TLT", "HYG", "EEM", "EFA", "FXI", "EWZ", "GDX", "XOP", "USO", "ARKK", "KWEB"],
  "period": ["2024-04-01", "2026-10-28"],
  "shock": {"one_day": -0.010, "two_day": -0.020, "scale": "sigma_etf(t) / sigma_spy(t)", "spy_scale": 1.0},
  "guard": {"ma_days": 200},
  "short_delta": 0.15,
  "long_delta": 0.05,
  "dte": {"min": 14, "max": 35, "target": 21},
  "costs": {"tight_roots": ["SPY", "QQQ", "IWM"], "tight": {"min": 0.01, "pct": 0.05}, "other": {"min": 0.02, "pct": 0.10}},
  "risk_per_trade": 0.10,
  "open_risk_cap": 0.30,
  "capitals": [2500, 5000, 10000],
  "primary_capital": 5000
}
```
