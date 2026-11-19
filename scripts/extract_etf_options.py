"""
[AG-31] Pull the put rows for a list of ETFs out of the OPRA day files we already have (no new download).

Reads data/raw/options/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz (every US option, one file per day, from Massive)
and writes one file per month with only the puts of the ETFs below:
    data/processed/options/etf/puts_YYYY-MM.csv.gz      columns: date,ticker,volume,open,close,high,low,ws,tx
Resumable: a finished month is skipped. Plain Python, no pandas, so it runs anywhere.

    py -3.12 -u scripts/extract_etf_options.py
"""
import gzip
import re
import sys
from pathlib import Path

ROOTS = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "SMH", "XBI",
         "KRE", "GLD", "SLV", "TLT", "HYG", "EEM", "EFA", "FXI", "EWZ", "GDX", "XOP", "USO", "ARKK", "KWEB"]
SRC = Path("data/raw/options/day_aggs_v1")
OUT = Path("data/processed/options/etf")
PAT = re.compile(r"^O:(" + "|".join(ROOTS) + r")\d{6}P\d{8},")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    months = sorted({p.parent for p in SRC.glob("*/*/*.csv.gz")})
    print(f"{len(months)} months of day files", flush=True)
    for m in months:
        tag = f"{m.parent.name}-{m.name}"
        dst = OUT / f"puts_{tag}.csv.gz"
        if dst.exists():
            continue
        tmp = dst.with_suffix(".tmp")
        n = 0
        with gzip.open(tmp, "wt") as out:
            for f in sorted(m.glob("*.csv.gz")):
                day = f.name[:10]
                try:
                    with gzip.open(f, "rt") as fh:
                        for line in fh:
                            if PAT.match(line):
                                out.write(f"{day},{line}")
                                n += 1
                except (OSError, EOFError) as e:
                    print(f"  skip {f.name}: {e}", flush=True)
        tmp.replace(dst)
        print(f"  {tag}: {n:,} put rows", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
