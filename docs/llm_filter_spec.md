# LLM news filter: model and setup (spec 2.3 / 2.4)

**Status (2026-11-05):** model picked, inputs pulled (`scripts/download_event_news.py`: Reuters headlines + story texts), prompt and
tests specified below. Code not written yet (STR-2.3a, STR-2.4a, STR-2.4b are TODO in `strategy_logic.py`).

## 1. Model

| Setting | Choice | Why |
|---|---|---|
| Provider | DeepSeek API (`DEEPSEEK_API_KEY`, `https://api.deepseek.com`) | already chosen by the team (`.env.example`) |
| Model | **`deepseek-v4-pro`** (version DeepSeek-V4-Pro-0813) | stronger reasoning for $-impact estimates than `deepseek-flash`; cost is still tiny (section 6) |
| Thinking mode | **off** (`"thinking": {"type": "disabled"}`) | temperature is ignored in thinking mode; we need repeatable answers |
| Temperature | **0** | repeatable answers |
| Output | `response_format = {"type": "json_object"}` + schema in the prompt | parsed straight into the event table |
| Version pinning | save the `model` field of every response; re-run the recognition test if it changes | API names point to a version that DeepSeek can swap (`deepseek-flash` moved to V4.1-Flash-1017) |

Sources: [Models & pricing](https://api-docs.deepseek.com/quick_start/pricing/), [Chat completions](https://api-docs.deepseek.com/api/create-chat-completion/), [V4 model card](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf) (checked 2026-11-05).

## 2. The look-ahead problem (most important)

V4 was released 2026-04-24. DeepSeek does not publish a knowledge cutoff; an unofficial source puts it
around **May 2025**. Our option-period events run **Apr 2024 – Oct 2026**, so the model may already know
how the earlier ones turned out (lawsuit settled small, stock recovered, company went bankrupt...).
That would make the filter look better in the backtest than it can be live.

Controls, in this order:
1. **Cutoff probe** (before anything else): ask the model about ~24 well-known market events, one per
   month from 2024-01 to 2026-10, and record the last month it answers correctly. That month = our cutoff.
2. **Clean test set = events after the cutoff.** If the cutoff is mid-2025, that is about Jul 2025 – Oct 2026
   (~1,000 events). Results here are the ones we trust.
3. **Events before the cutoff only with anonymization that passes the recognition test** (section 4).
4. **No future information in the prompt:** only headlines published before the shock-day close
   (16:00 New York time, DAT-09; 20:00 UTC in summer, 21:00 UTC in winter), no year, no dates, no stock moves after the shock.

## 3. Inputs per event

| Field | Source | Notes |
|---|---|---|
| Headlines before the decision | `data/raw/news/reuters_events/<RIC>_<YYYYMMDD>.parquet` | Reuters only; first version of each story (DAT-08), published before the shock-day close in New York time (DAT-09); max 30 |
| Up to 2 Reuters story texts | `data/raw/news/reuters_stories/<RIC>_<YYYYMMDD>.parquet` | first stories that broke between the previous close and the shock-day close (DAT-10), else the latest before the close; truncated to 2,000 characters each |
| Sector, industry | `data/raw/reference/gics.parquet` | GICS sector + industry |
| Market cap before the shock | `data/raw/market_cap/` | rounded to the nearest $5B in the prompt |
| Shock size | `events.csv` `abnormal` | beta-adjusted 1-day return |
| Market cap lost | abnormal × market cap | rounded to 2 significant figures |

**Anonymization** (before the prompt is built): replace the company's names, short names, ticker and
RIC with "the Company"; replace dates with "Day 0" / "Day −1"; keep dollar amounts (needed for the impact
estimate). Product and executive names can still identify a company: the recognition test measures that.

## 4. Recognition test (spec 2.4)

Run on 100 random anonymized events from before the cutoff:
"Which company is this, and what did its stock do over the following month?"
- Identified = correct company name or ticker.
- **Pass: ≤ 10% identified.** If it fails, anonymize harder (product / people names) or use only
  post-cutoff events for any result we report.

## 5. Prompt and output

System message:
```text
You are an equity analyst. You judge whether a stock's one-day drop is justified by the news.
You only know what is in the message. Answer in JSON only, with exactly the keys requested.
```

User message (template):
```text
Company: a {sector} company ({industry}), market value about ${cap_bn}B before the news.
On Day 0 the stock fell {abn_pct}% more than the market, about ${lost_bn}B of market value.

News published up to the close of Day 0 (oldest first):
{headlines}

Story excerpts:
{stories}

Return JSON:
{"category": one of ["analyst downgrade", "peer/sector contagion", "lawsuit", "regulatory probe",
  "short seller report", "product recall", "dilution/offering", "guidance cut", "management change",
  "M&A", "fraud/accounting", "macro/other", "unclear"],
 "est_impact_usd": {"low": number, "mid": number, "high": number},
 "impact_basis": "one sentence: what the dollar estimate is based on",
 "verdict": "justified" | "not justified" | "unsure",
 "reason": "at most two sentences",
 "confidence": number between 0 and 1}
```

`est_impact_usd` is the estimated permanent hit to the company's value in dollars (e.g. likely
settlement, lost revenue × multiple, dilution amount). The categories extend spec 2.3 with
"guidance cut", "management change", "M&A", "fraud/accounting" (binary events that spec 1.2 excludes).

Code side (to write):
- **STR-2.3a** `news_category`: the LLM's `category`, with spec 1.2 exclusions mapped to "do not trade".
- **STR-2.4a** `overreaction_ratio` = market cap lost ÷ `est_impact_usd.mid` (also low and high).
- **STR-2.4b** `is_overreaction`: ratio ≥ 2x / 3x* / 5x **and** verdict = "not justified".
- Proposed new prefix **LLM-nn** in `src/llm/` for prompt building, anonymization, API calls with a
  cache (hash of the prompt → saved raw response) and response parsing.

## 6. Cost and time

About 1,500 events × ~4,000 input tokens + ~400 output tokens ≈ 5.8M in / 0.6M out.
At `deepseek-v4-pro` prices ($0.66–1.32 per 1M input, $1.98–3.96 per 1M output; lower off-peak):
**about $5–10 for the whole set**, plus under $1 for the cutoff probe and recognition test.
Run off-peak (outside 01:00–04:00 and 06:00–10:00 UTC on weekdays).

## 7. How we judge the filter

Compare events the LLM calls "not justified" with the rest, on the clean (post-cutoff) set:
`car_e1_10`, worst close in the hold (`low_e1_10`, EVT-07) and backtest P&L, overall and per category,
at ratio thresholds 2x / 3x / 5x. The filter is useful only if "not justified" events have a thinner
tail than all events.

## 8. Open questions for the team

- Cutoff month: set by the probe; agree before anyone looks at backtest results.
- One model, or an "LLM council" (README idea: several models vote)? Start with one; add a second
  (e.g. `deepseek-flash`) only if results are borderline.
- Re-check on new news while holding (STR-3.7a `llm_recheck`): same prompt with the new headlines.
