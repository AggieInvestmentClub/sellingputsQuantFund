"""
AG-30: I/B/E/S consensus history from LSEG, point-in-time, for the fair-value gate (AG-14).

One file per RIC: data/raw/fundamentals/<RIC>.parquet, long format, one row per (field, date):
    field ("eps_ntm", "eps_fy1", "target", "rec", "n_est"), date (estimate date, tz-naive), value
Every value carries the date LSEG says the consensus had that value, so the gate can take the last one
dated before a shock close, the same rule as the news (DAT-09).
Needs LSEG Workspace open (lseg.data session, src/ingest/lseg_client.py).
"""
import time

import pandas as pd

# (our name, LSEG field, period parameter or None). Change here if a field name differs in your entitlement:
# check with scripts/download_fundamentals.py --limit 3 (it prints which fields came back empty).
FIELDS = [
    ("eps_ntm", "TR.EPSMean", "NTM"),
    ("eps_fy1", "TR.EPSMean", "FY1"),
    ("n_est", "TR.EPSNumIncEstimates", "FY1"),
    ("target", "TR.PriceTargetMean", None),
    ("rec", "TR.RecMean", None),
]


def _one(ld, ric, name, field, period, start, end, frq):
    params = {"SDate": start, "EDate": end, "Frq": frq}
    if period:
        params["Period"] = period
    df = ld.get_data(universe=[ric], fields=[f"{field}.date", field], parameters=params)
    if df is None or df.empty or df.shape[1] < 3:
        return pd.DataFrame(columns=["field", "date", "value"])
    df = df.iloc[:, [1, 2]].copy()                      # columns: Instrument, Date, value
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().drop_duplicates("date", keep="last")
    df.insert(0, "field", name)
    return df


def fetch_fundamentals(ric, start="2021-01-01", end=None, frq="W", fields=FIELDS, retries=3, pause=1.0):
    """
    [AG-30] I/B/E/S history for one RIC (weekly by default: enough for P/E percentiles and 30-day revisions)
    returns long DataFrame (field, date, value); empty fields are simply absent
    """
    import lseg.data as ld
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    out = []
    for name, field, period in fields:
        for k in range(retries):
            try:
                out.append(_one(ld, ric, name, field, period, start, end, frq))
                break
            except Exception:
                if k == retries - 1:
                    raise
                time.sleep(pause * (k + 1))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["field", "date", "value"])
