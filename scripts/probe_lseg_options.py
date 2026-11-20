"""
[AG-62] How far back does LSEG keep daily prices for EXPIRED SPX / SPY put options? (Workspace must be open.)

LSEG names an expired US option with a reconstructed RIC (OPRA rule, as in LSEG's "Finding Expired Options and
Backtesting a Short Iron Condor Strategy" article):
    ROOT + month code (calls A-L, puts M-X) + day(2) + year(2) + strike code + ".U" + "^" + expiry month (A-L) + year(2)
    strike code: <1000 -> 3-digit integer + 2 decimals ("45000" = 450.00); 1000-9999 -> 4 digits + "0" ("40000" = 4000);
                 10000-19999 -> "A" + last 4 digits.
For one monthly expiry per test year it tries a few strikes ~5-15% below the index and reports which RICs return
BID / ASK / close history in the 40 days before expiry. Nothing is saved except the report.

    py -3.12 -u scripts/probe_lseg_options.py
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import lseg.data as ld  # noqa: E402

from src.ingest.lseg_client import lseg_session  # noqa: E402

YEARS = [1998, 2002, 2005, 2008, 2010, 2013, 2016, 2019, 2022, 2024]
MONTH = "ABCDEFGHIJKL"
OUT = Path("data/results/agent_diag/lseg_option_probe.csv")


def third_friday(y, m):
    d = pd.Timestamp(y, m, 15)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


def strike_code(k):
    k = float(k)
    i = int(round(k))
    if k < 1000:
        return f"{int(k):03d}{int(round((k - int(k)) * 100)):02d}"
    if k < 10000:
        return f"{i:04d}0"
    return "A" + f"{i:05d}"[-4:]


def put_ric(root, exp, k):
    code = chr(ord(MONTH[exp.month - 1]) + 12)          # puts: M-X
    return f"{root}{code}{exp.day:02d}{str(exp.year)[-2:]}{strike_code(k)}.U^{MONTH[exp.month - 1]}{str(exp.year)[-2:]}"


def level(ric, day):
    try:
        h = ld.get_history(ric, fields=["TRDPRC_1"], start=(day - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                           end=day.strftime("%Y-%m-%d"))
        return float(h.iloc[-1, 0])
    except Exception:  # noqa: BLE001
        return None


def main():
    rows = []
    with lseg_session():
        for y in YEARS:
            exp = third_friday(y, 3)
            for root, under, step in (("SPX", ".SPX", 25), ("SPY", "SPY", 1)):
                S = level(under, exp - pd.Timedelta(days=30))
                if not S:
                    rows.append({"year": y, "root": root, "ric": None, "rows": 0, "note": f"no {under} level"})
                    continue
                for pct in (0.85, 0.90, 0.95):
                    k = round(S * pct / step) * step
                    ric = put_ric(root, exp, k)
                    try:
                        h = ld.get_history(ric, fields=["BID", "ASK", "TRDPRC_1"],
                                           start=(exp - pd.Timedelta(days=40)).strftime("%Y-%m-%d"),
                                           end=exp.strftime("%Y-%m-%d"))
                        n = 0 if h is None else int(h.notna().any(axis=1).sum())
                        has_ba = bool(h is not None and {"BID", "ASK"} <= set(h.columns) and h[["BID", "ASK"]].notna().any().all())
                        rows.append({"year": y, "root": root, "ric": ric, "rows": n, "bid_ask": has_ba, "note": ""})
                    except Exception as e:  # noqa: BLE001
                        rows.append({"year": y, "root": root, "ric": ric, "rows": 0, "bid_ask": False,
                                     "note": str(e)[:80]})
                    print(f"  {y} {root:3s} {rows[-1]['ric']}: {rows[-1]['rows']} days"
                          f"{' with bid/ask' if rows[-1].get('bid_ask') else ''} {rows[-1]['note']}", flush=True)
    r = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT, index=False)
    ok = r[r["rows"] > 0]
    print(f"\nRICs with data: {len(ok)}/{len(r)}; earliest year with data: "
          f"{ok['year'].min() if len(ok) else 'none'} | saved {OUT}")
    print("Paste this output to Claude. If the early years (before 2010) have data with bid/ask, LSEG can extend the test.")


if __name__ == "__main__":
    main()
