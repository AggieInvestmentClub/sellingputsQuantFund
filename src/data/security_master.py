"""
Lookups on the security master (built by src/ingest/security_master.py, ING-18).
Pure functions: no downloads, work on the saved tables.
"""
import pandas as pd


def _base(ric):
    # "SIVB.OQ^C23" -> "SIVB", "BRKb.N" -> "BRKb"
    return ric.split("^")[0].split(".")[0]


def lseg_option_root(ric):
    """[DAT-01] LSEG option root for a stock RIC: "META.OQ" -> "META", "BRKb.N" -> "BRKB", "SIVB.OQ^C23" -> "SIVB" """
    return _base(ric).upper()


def massive_ticker_from_ric(ric):
    """[DAT-02] fallback Massive ticker from a RIC: "META.OQ" -> "META", "BRKb.N" -> "BRK.B" (class letter after a dot)"""
    base = _base(ric)
    if len(base) > 1 and base[-1].islower():
        return base[:-1] + "." + base[-1].upper()
    return base


def massive_ticker_on(massive_tickers, ric, date):
    """
    [DAT-03] Massive ticker for a RIC on a date (Massive keys option contracts by the ticker at the time)

    massive_tickers: table from ING-18 (ric, massive_ticker, valid_from, source)
    ex: META.OQ on 2022-05-13 -> "FB", on 2022-06-15 -> "META"
    before the first recorded change, returns the earliest known ticker; None if the RIC is unknown
    """
    rows = massive_tickers[massive_tickers["ric"] == ric]
    if rows.empty:
        return None
    d = pd.Timestamp(date)
    known = rows.dropna(subset=["valid_from"]).sort_values("valid_from")
    if known.empty:
        return rows["massive_ticker"].iloc[0]
    valid = known[known["valid_from"] <= d]
    return (valid if not valid.empty else known).iloc[-1 if not valid.empty else 0]["massive_ticker"]


def companies(rics, master):
    """
    [DAT-04] group RICs by company so share classes count once: {company_id: [rics]}

    rics: iterable of RICs (e.g. members on a date), master: security master from ING-18
    ex: ["GOOG.OQ", "GOOGL.OQ", "AAPL.OQ"] -> {"5030853586": ["GOOG.OQ", "GOOGL.OQ"], "4295905573": ["AAPL.OQ"]}
    RICs missing from the master become their own group ("RIC:<ric>")
    """
    ids = dict(zip(master["ric"], master["company_id"]))
    out = {}
    for r in rics:
        out.setdefault(ids.get(r, f"RIC:{r}"), []).append(r)
    return {k: sorted(v) for k, v in out.items()}
