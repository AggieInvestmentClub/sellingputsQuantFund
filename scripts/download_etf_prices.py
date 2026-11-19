"""
[AG-32] Daily prices for the ETFs in scripts/extract_etf_options.py, from LSEG (Workspace must be open).

Writes data/raw/prices_bulk/adj/<RIC>.parquet and .../unadj/<RIC>.parquet (src/ingest/lseg_bulk.py) and a map
data/raw/prices_bulk/etf_rics.csv (ticker -> RIC that worked). Tries the plain ticker first, then exchange suffixes.

    py -3.12 -u scripts/download_etf_prices.py
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=DeprecationWarning)

from scripts.extract_etf_options import ROOTS  # noqa: E402
from src.ingest.lseg_bulk import ROOT, KINDS, fetch_series, safe_name  # noqa: E402
from src.ingest.lseg_client import lseg_session  # noqa: E402

START, END = "2009-01-01", pd.Timestamp.today().strftime("%Y-%m-%d")
SUFFIXES = ["", ".P", ".O", ".OQ", ".K", ".N"]


def main():
    for kind, _, _ in KINDS:
        (ROOT / kind).mkdir(parents=True, exist_ok=True)
    found = []
    with lseg_session():
        for t in ROOTS:
            ok = None
            for sfx in SUFFIXES:
                ric = t + sfx
                try:
                    test = fetch_series(ric, "2024-01-02", "2024-01-31", ["TRDPRC_1"], True)
                except Exception:
                    test = pd.DataFrame()
                if not test.empty and test["TRDPRC_1"].notna().any():
                    ok = ric
                    break
            if ok is None:
                print(f"  {t:5s} no RIC found", flush=True)
                continue
            for kind, fields, adjusted in KINDS:
                df = fetch_series(ok, START, END, fields, adjusted)
                df.to_parquet(ROOT / kind / f"{safe_name(ok)}.parquet", index=False)
            print(f"  {t:5s} -> {ok:7s} {len(df):5d} days, last {df['date'].max().date()}", flush=True)
            found.append({"ticker": t, "ric": ok})
    pd.DataFrame(found).to_csv(ROOT / "etf_rics.csv", index=False)
    print(f"\n{len(found)}/{len(ROOTS)} ETFs saved; map in {ROOT / 'etf_rics.csv'}")


if __name__ == "__main__":
    main()
