# Pre-registered test: ETF branch (frozen 2026-11-19 01:15 CST, before any non-SPY ETF was run)

Question: does the index branch's tested rule still make money when it runs on many liquid ETFs at once,
so the fund gets more trades per year than SPY alone (~3)?

Nothing below is chosen from results. Entry, strikes, expiry and exit are the rule the walk-forward picked from
2018 on (strategy/preregistered_index_branch.md); sizing is the capital rule
(strategy/preregistered_index_branch_small.md, "Capital rule"). Only three things are new, and they are written
down here before any non-SPY ETF result exists:

1. **Shock size scales with each ETF's volatility.** A -1.5% day is routine for KWEB and huge for HYG.
   Threshold(ETF, day t) = -1.5% (1-day) or -2.5% (2-day) x sigma_ETF(t) / sigma_SPY(t), where sigma = standard
   deviation of daily close-to-close returns on ADJUSTED closes over the 252 trading days ending the day before t
   (at least 200 days, else no shock). For SPY the ratio is 1, so SPY keeps exactly the tested rule.
2. **Wider assumed bid/ask outside the big three.** OPRA day files have closes only. SPY, QQQ, IWM: half-spread
   max($0.01, 5% of the close) (as in every earlier test). Every other ETF: max($0.02, 10% of the close).
3. **Portfolio limits.** One open spread per ETF; several ETFs may be open at once. Each trade risks at most 10% of
   capital: contracts = floor(10% of capital / max loss per contract), or 1 contract if the max loss is at most
   1.15 x that budget, else the trade is skipped. Total open max loss may not pass 30% of capital; a trade that would
   pass it is skipped. Entries are walked by date, then in the ETF order below. Capital is fixed (no compounding).

Trend guard and "no new low" use adjusted closes (so ex-dividend drops are not shocks); strikes, the short-strike
spot and settlement use unadjusted closes. Long leg: the NARROWER of (the strike nearest short - W) and the
~0.03-delta put, W = floor(capital x 10% / 100) dollars. A trade must expire on or before the last date with both
option data and a settlement price.

Checks (all four must pass, for the primary capital $2,500; $5,000 and $10,000 reported the same way):
1. mean return on risk > 0 with a 95% bootstrap interval, resampled by ENTRY WEEK (same-week trades move together),
   above 0;
2. the account is positive over the period;
3. max drawdown better than -30%;
4. worst month better than -25%.

Data: OPRA day closes 2024-04-01 .. 2026-10-28 (about 2.5 years), so a PASS here is **evidence, not proof**.
Already seen before writing: SPY's own trades in this period (index branch, capital test). Not seen: any result
for the other 28 ETFs. Run: `py -3.12 -u scripts/etf_branch.py --run`.

```json
{
  "branch": "etf",
  "written": "2026-11-19 01:15 CST",
  "roots": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "SMH", "XBI",
            "KRE", "GLD", "SLV", "TLT", "HYG", "EEM", "EFA", "FXI", "EWZ", "GDX", "XOP", "USO", "ARKK", "KWEB"],
  "period": ["2024-04-01", "2026-10-28"],
  "entry": "shock+guard, day +1..+3, first close above the post-shock low (adjusted closes)",
  "shock": {"one_day": -0.015, "two_day": -0.025, "scale": "sigma_etf(t) / sigma_spy(t)",
            "sigma": "std of daily returns on adjusted closes, 252 trading days ending t-1, min 200", "spy_scale": 1.0},
  "guard": {"ma_days": 200, "trend": "adjusted close > 200-day average",
            "iv_term": "ATM put IV at ~9 days not above ~30 days (missing IV does not block)"},
  "short_delta": 0.10,
  "long_delta": 0.03,
  "long_leg": "narrower of: strike nearest (short - W), and the ~0.03-delta put; W = floor(capital x risk_per_trade / 100)",
  "dte": {"min": 14, "max": 35, "target": 21},
  "exit": "dynamic take-profit (>= 50% of credit kept and 20 pts ahead of time, or >= 75%), else expiry; no stop",
  "fills": "sell at bid, buy at ask (FILL = 1); fee $0.65 per contract per leg, each way",
  "costs": {"tight_roots": ["SPY", "QQQ", "IWM"], "tight": {"min": 0.01, "pct": 0.05}, "other": {"min": 0.02, "pct": 0.10}},
  "risk_per_trade": 0.10,
  "open_risk_cap": 0.30,
  "sizing": "contracts = floor(budget / max loss), or 1 if max loss <= 1.15 x budget, else skip",
  "portfolio": "one open spread per ETF; walk by date then roots order; skip if open risk would pass the cap; no compounding",
  "capitals": [2500, 5000, 10000],
  "primary_capital": 2500,
  "checks": ["mean ror > 0, entry-week-clustered 95% bootstrap interval above 0", "account positive over the period",
             "max drawdown better than -30%", "worst month better than -25%"],
  "label": "evidence, not proof (about 2.5 years of option data)"
}
```
