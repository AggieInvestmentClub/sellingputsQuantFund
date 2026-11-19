"""
[AG-54] Index/ETF macro shock classifier (PROTOTYPE, descriptive only: not pre-registered, not a trading rule).

For an SPY/XSP or sector-ETF drop it asks: is this a SYSTEMIC / LIQUIDITY BREAK (stay out: the crash guard's job,
done more broadly) or a TRADEABLE VRP DISLOCATION (a normal sell-off in a calm regime, where selling put spreads has
been paid)? Deterministic features from closes already on disk (data/raw/prices_bulk, LSEG):

  SPY: 1- and 2-day return, drawdown from its 63-day high, distance to its 200-day average, 20-day realized vol
  term structure: ~9-day vs ~30-day ATM put IV when given (index_branch features)
  cross-asset, same day / 5 days: TLT (bonds), GLD (gold), HYG (high-yield credit)

Break flags (each one a reason): SPY below its 200-day average; SPY 10%+ below its 63-day high; 20-day realized vol
>= 30%; 9-day IV above 30-day IV (inverted term structure); HYG down 2%+ over 5 days (credit stress); "dash for cash"
(SPY down 1.5%+ with bonds AND gold also down the same day). Label: >= 2 flags -> "systemic/liquidity break";
0 flags with SPY above its 200-day average -> "tradeable VRP dislocation"; otherwise "unclear".

Optional LLM step (classify_headlines): macro headlines through llm.safe_call_llm; it can only make the label MORE
cautious (vrp -> unclear, unclear -> break), never less; offline it is skipped and says so.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("systemic/liquidity break", "tradeable VRP dislocation", "unclear")
ROOT = Path("data/raw/prices_bulk")
TICKERS = ("SPY", "TLT", "GLD", "HYG")


def load_closes(root=ROOT, tickers=TICKERS):
    """adjusted closes from data/raw/prices_bulk/adj via etf_rics.csv (ticker -> RIC); DataFrame, one column each"""
    root = Path(root)
    m = pd.read_csv(root / "etf_rics.csv")
    rics = dict(zip(m["ticker"], m["ric"]))
    out = {}
    for t in tickers:
        if t not in rics:
            continue
        p = root / "adj" / f"{rics[t].replace('.', '_')}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            s = d.set_index(pd.to_datetime(d["date"]).dt.normalize())["TRDPRC_1"].astype(float)
            out[t] = s[~s.index.duplicated(keep="last")].sort_index()
    return pd.DataFrame(out)


def macro_features(px, date, iv9=None, iv30=None):
    """point-in-time features at the close of `date` (px: load_closes output); None if SPY lacks history"""
    date = pd.Timestamp(date).normalize()
    h = px.loc[:date]
    if "SPY" not in h or len(h["SPY"].dropna()) < 210 or h.index[-1] != date:
        return None
    spy = h["SPY"].dropna()
    r = spy.pct_change()

    def ret(t, n):
        if t not in h:
            return np.nan
        s = h[t].dropna()
        return float(s.iloc[-1] / s.iloc[-1 - n] - 1) if len(s) > n else np.nan
    return {"date": date, "spy_ret1": float(r.iloc[-1]), "spy_ret2": ret("SPY", 2), "spy_ret5": ret("SPY", 5),
            "spy_dd63": float(spy.iloc[-1] / spy.iloc[-63:].max() - 1),
            "spy_vs_200": float(spy.iloc[-1] / spy.iloc[-200:].mean() - 1),
            "rv20": float(r.iloc[-20:].std() * np.sqrt(252)),
            "tlt_ret1": ret("TLT", 1), "gld_ret1": ret("GLD", 1), "hyg_ret1": ret("HYG", 1), "hyg_ret5": ret("HYG", 5),
            "iv9": iv9 if iv9 is not None and np.isfinite(iv9) else np.nan,
            "iv30": iv30 if iv30 is not None and np.isfinite(iv30) else np.nan}


def classify(f, dd_break=-0.10, rv_break=0.30, hyg_break=-0.02, dash=-0.015):
    """[AG-54] label + reasons from macro_features (thresholds are round numbers chosen a priori, untuned)"""
    if f is None:
        return {"label": "unclear", "flags": [], "reasons": ["not enough SPY history"]}
    flags = []
    if f["spy_vs_200"] < 0:
        flags.append(f"SPY {f['spy_vs_200']:+.1%} vs its 200-day average (downtrend)")
    if f["spy_dd63"] <= dd_break:
        flags.append(f"SPY {f['spy_dd63']:+.1%} from its 63-day high")
    if f["rv20"] >= rv_break:
        flags.append(f"20-day realized vol {f['rv20']:.0%}")
    if np.isfinite(f["iv9"]) and np.isfinite(f["iv30"]) and f["iv9"] > f["iv30"]:
        flags.append(f"inverted term structure (9-day IV {f['iv9']:.0%} > 30-day {f['iv30']:.0%})")
    if np.isfinite(f["hyg_ret5"]) and f["hyg_ret5"] <= hyg_break:
        flags.append(f"credit stress: HYG {f['hyg_ret5']:+.1%} over 5 days")
    if (f["spy_ret1"] <= dash and np.isfinite(f["tlt_ret1"]) and np.isfinite(f["gld_ret1"])
            and f["tlt_ret1"] < 0 and f["gld_ret1"] < 0):
        flags.append(f"dash for cash: SPY {f['spy_ret1']:+.1%}, bonds {f['tlt_ret1']:+.1%} and gold {f['gld_ret1']:+.1%} all down")
    if len(flags) >= 2:
        label = LABELS[0]
    elif not flags:
        label = LABELS[1]
    else:
        label = LABELS[2]
    reasons = flags or [f"SPY {f['spy_vs_200']:+.1%} above its 200-day average, vol {f['rv20']:.0%}, no credit or "
                        "term-structure stress"]
    return {"label": label, "flags": flags, "reasons": reasons}


MACRO_PROMPT = """Headlines from the day the S&P 500 fell (names and dates removed):
{headlines}

Is this sell-off driven by a SYSTEMIC break (funding or liquidity stress, credit event, banking or sovereign crisis,
forced deleveraging, war or pandemic shock) or by a NORMAL repricing (data surprise, earnings, rates expectations,
positioning)? Return JSON: {{"systemic": true | false, "reason": "one sentence", "confidence": number 0-1}}"""


def classify_headlines(headlines, llm=None):
    """
    optional LLM view of the day's macro headlines. llm: callable(messages) -> content string or None
    (e.g. a lambda around llm.safe_call_llm). None / unavailable -> {"systemic": None, "source": "offline"}.
    """
    if not headlines or llm is None:
        return {"systemic": None, "source": "offline", "reason": "no LLM step (offline or not configured)"}
    from src.agent.llm import extract_json
    msgs = [{"role": "system", "content": "You are a macro strategist. Answer in JSON only."},
            {"role": "user", "content": MACRO_PROMPT.format(headlines="\n".join(f"- {h}" for h in headlines[:30]))}]
    try:
        content = llm(msgs)
    except Exception as e:                                     # noqa: BLE001
        return {"systemic": None, "source": "offline", "reason": f"LLM error {type(e).__name__}"}
    d = extract_json(content) if content else None
    if not d or not isinstance(d.get("systemic"), bool):
        return {"systemic": None, "source": "offline" if content is None else "unparsed", "reason": "no usable answer"}
    return {"systemic": d["systemic"], "source": "llm", "reason": str(d.get("reason", "")),
            "confidence": d.get("confidence")}


def combine(det, llm_view):
    """the LLM can only make the deterministic label more cautious"""
    out = dict(det)
    if llm_view.get("systemic") is True:
        out["label"] = LABELS[2] if det["label"] == LABELS[1] else LABELS[0]
        out["reasons"] = det["reasons"] + [f"LLM: systemic ({llm_view.get('reason', '')})"]
    out["llm"] = llm_view
    return out


def shock_days(px, start=None, end=None, d1=-0.015, d2=-0.025):
    """SPY shock days (the index-branch definition: 1-day <= -1.5% or 2-day <= -2.5%)"""
    s = px["SPY"].dropna()
    r1, r2 = s.pct_change(), s.pct_change(2)
    d = s.index[(r1 <= d1) | (r2 <= d2)]
    if start:
        d = d[d >= pd.Timestamp(start)]
    if end:
        d = d[d <= pd.Timestamp(end)]
    return d


def evaluate(px, days, iv=None, horizon=21):
    """descriptive table: label per day + SPY forward return and worst close over the next `horizon` days"""
    spy = px["SPY"].dropna()
    rows = []
    for d in days:
        i9 = i30 = None
        if iv is not None and d in iv.index:
            i9, i30 = iv.loc[d, "iv9"], iv.loc[d, "iv30"]
        f = macro_features(px, d, i9, i30)
        c = classify(f)
        k = spy.index.get_loc(d)
        fwd = spy.iloc[k:k + horizon + 1]
        rows.append({"date": d, "label": c["label"], "n_flags": len(c["flags"]), "flags": "; ".join(c["flags"]),
                     "fwd_ret": float(fwd.iloc[-1] / fwd.iloc[0] - 1) if len(fwd) > horizon else np.nan,
                     "fwd_worst": float(fwd.min() / fwd.iloc[0] - 1) if len(fwd) > 1 else np.nan})
    return pd.DataFrame(rows)


def to_json(c):
    return json.dumps({k: v for k, v in c.items() if k != "llm"}, default=str)
