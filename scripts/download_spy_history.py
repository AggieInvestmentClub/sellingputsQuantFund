"""
Download SPY put history (AG-25) from Massive (ex-Polygon) options flat files for the out-of-sample test.
Keeps only SPY put rows from each daily file, so ~10 years is a few hundred MB, not ~7 GB.

    py -3.12 -u scripts/download_spy_history.py --start 2014-06-02 --end 2024-03-28
    py -3.12 -u scripts/download_spy_history.py --check        # one file: does our plan reach that far back?

Needs MASSIVE_S3_* in .env (same keys as the existing option downloads) and pip install boto3.
Massive plans limit history (Starter 2 years, Developer 4, Advanced all, records from 2014-06-02):
if --check returns 403/AccessDenied, the plan does not cover the date.
Writes data/processed/options/spy_puts_hist.csv in the format scripts/spy_benchmark.py reads. Re-runnable:
days already fetched are skipped (data/processed/options/spy_hist_parts/).
"""
import argparse
import gzip
import io
import os
import sys
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)      # .env wins over variables already set in the shell
except ImportError:
    pass

PARTS = Path("data/processed/options/spy_hist_parts")
OUT = Path("data/processed/options/spy_puts_hist.csv")


def client():
    import boto3
    return boto3.client("s3", aws_access_key_id=os.getenv("MASSIVE_S3_ACCESS_KEY_ID"),
                        aws_secret_access_key=os.getenv("MASSIVE_S3_SECRET_ACCESS_KEY"),
                        endpoint_url=os.getenv("MASSIVE_S3_ENDPOINT", "https://files.massive.com"))


def fetch(c, day):
    key = f"us_options_opra/day_aggs_v1/{day:%Y}/{day:%m}/{day:%Y-%m-%d}.csv.gz"
    body = c.get_object(Bucket=os.getenv("MASSIVE_S3_BUCKET", "flatfiles"), Key=key)["Body"].read()
    with gzip.open(io.BytesIO(body), "rt") as fh:
        next(fh)
        return [f"{day:%Y-%m-%d},{line}" for line in fh if line.startswith("O:SPY") and line[11] == "P"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-06-02")
    ap.add_argument("--end", default="2024-03-28")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    c = client()
    days = pd.bdate_range(a.start, a.end)
    if a.check:
        for d in (days[0], days[len(days) // 2], days[-1]):
            try:
                print(f"{d.date()}: {len(fetch(c, d))} SPY put rows")
            except Exception as e:
                print(f"{d.date()}: {type(e).__name__}: {str(e)[:120]}")
        return
    PARTS.mkdir(parents=True, exist_ok=True)
    miss = denied = 0
    for i, d in enumerate(days, 1):
        f = PARTS / f"{d:%Y-%m-%d}.csv"
        if f.exists():
            continue
        try:
            rows = fetch(c, d)
        except Exception as e:
            msg = str(e)
            if "NoSuchKey" in msg or "404" in msg:      # market holiday
                miss += 1
                continue
            if "AccessDenied" in msg or "403" in msg:
                denied += 1
                if denied >= 5:
                    sys.exit(f"access denied from {d.date()}: the Massive plan does not cover this history (see --check)")
                continue
            print(f"  {d.date()}: {type(e).__name__} {msg[:100]}")
            continue
        f.write_text("".join(rows))
        if i % 100 == 0:
            print(f"  {i}/{len(days)} days", flush=True)
    parts = sorted(PARTS.glob("*.csv"))
    with open(OUT, "w") as out:
        for p in parts:
            out.write(p.read_text())
    print(f"wrote {OUT}: {len(parts)} days ({miss} holidays skipped)")


if __name__ == "__main__":
    main()
