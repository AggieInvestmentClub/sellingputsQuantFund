"""
[AG-63] Daily S&P 500 index closes 1988-2010 from LSEG (Workspace open), for the simulated 1990-2009 test.
Writes data/raw/market_context/spx_1988_2010.parquet (date, close).

    py -3.12 -u scripts/download_spx_history.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from src.ingest.lseg_bulk import fetch_series  # noqa: E402
from src.ingest.lseg_client import lseg_session  # noqa: E402

OUT = Path("data/raw/market_context/spx_1988_2010.parquet")


def main():
    with lseg_session():
        for ric in (".SPX", ".INX", "SPY"):
            try:
                df = fetch_series(ric, "1988-01-01", "2010-12-31", ["TRDPRC_1"], False)
            except Exception as e:  # noqa: BLE001
                print(f"  {ric}: {e}")
                continue
            if len(df) > 4000:
                df = df.rename(columns={"TRDPRC_1": "close"})[["date", "close", "ric"]]
                OUT.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(OUT, index=False)
                print(f"{ric}: {len(df)} days, {df['date'].min().date()} .. {df['date'].max().date()} -> {OUT}")
                return
            print(f"  {ric}: only {len(df)} days")
    sys.exit("no long S&P 500 history found")


if __name__ == "__main__":
    main()
