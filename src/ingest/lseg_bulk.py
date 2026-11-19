"""Bulk daily price download for a list of RICs. Resumable: finished tickers are skipped."""
import time
from pathlib import Path

import pandas as pd
import lseg.data as ld

ROOT = Path("data/raw/prices_bulk")
ADJ_FIELDS = ["TRDPRC_1", "HIGH_1", "LOW_1", "OPEN_PRC", "ACVOL_UNS"]
UNADJ_FIELDS = ["TRDPRC_1", "ACVOL_UNS"]
KINDS = (("adj", ADJ_FIELDS, True), ("unadj", UNADJ_FIELDS, False))


def safe_name(ric):
    return ric.replace(".", "_").replace("^", "-")


def _with_retries(fn, retries=4):
    for attempt in range(retries):
        try:
            return fn()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * 2 ** attempt)


def fetch_series(ric, start, end, fields, adjusted):
    """Daily bars with a plain (timezone-naive) date column.
    Asks for one extra day at the start, because the API drops the first day."""
    s = (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    kwargs = dict(interval="daily", start=s, end=end, fields=fields)
    if not adjusted:
        kwargs["adjustments"] = "unadjusted"
    df = _with_retries(lambda: ld.get_history(ric, **kwargs))
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df["ric"] = ric
    return df.reset_index()


def download_universe(rics, start, end, limit=None, pause=0.2):
    for kind, _, _ in KINDS:
        (ROOT / kind).mkdir(parents=True, exist_ok=True)

    def done(r):
        return all((ROOT / k / f"{safe_name(r)}.parquet").exists() for k, _, _ in KINDS)

    rics = list(dict.fromkeys(r for r in rics if isinstance(r, str) and r))
    todo = [r for r in rics if not done(r)]
    print(f"{len(rics)} tickers total | {len(rics) - len(todo)} already done | {len(todo)} to go")
    if limit:
        todo = todo[:limit]

    failures, t0, streak = [], time.time(), 0
    for i, ric in enumerate(todo, 1):
        n_before = len(failures)
        for kind, fields, adjusted in KINDS:
            path = ROOT / kind / f"{safe_name(ric)}.parquet"
            if path.exists():
                continue
            try:
                df = fetch_series(ric, start, end, fields, adjusted)
                if df.empty:
                    failures.append((ric, kind, "empty"))
                else:
                    df.to_parquet(path, index=False)
            except Exception as e:
                failures.append((ric, kind, type(e).__name__))
            time.sleep(pause)
        streak = streak + 1 if len(failures) > n_before else 0
        if streak >= 10:
            print("10 tickers in a row failed, stopping. Last reasons:", failures[-3:])
            break
        if i % 25 == 0 or i == len(todo):
            rate = (time.time() - t0) / i
            print(f"{i}/{len(todo)} | {rate:.1f}s per ticker | ~{rate * (len(todo) - i) / 60:.0f} min left")

    if failures:
        pd.DataFrame(failures, columns=["ric", "kind", "reason"]).to_csv(
            ROOT / "failures.csv", index=False)
        print(f"{len(failures)} failures, saved to {ROOT / 'failures.csv'}")
    else:
        print("no failures")
    return failures

