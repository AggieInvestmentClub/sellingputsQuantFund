"""
Build the S&P 500 membership history (if missing) and the security master, then check the
Massive ticker mapping against real option contract listings.

Run from the repo root (LSEG Workspace open):  python scripts/build_security_master.py
"""
import os
import random
import sys
import time

sys.path.insert(0, ".")
import pandas as pd
import requests
from dotenv import load_dotenv

from src.data.security_master import companies, massive_ticker_on
from src.ingest.lseg_client import lseg_session
from src.ingest.lseg_universe import RAW, build_membership, fetch_members, save_membership
from src.ingest.security_master import build_security_master

load_dotenv()
MEMBERSHIP = RAW / "sp500_membership.parquet"
MIN_SIZE = 450


def massive_has_contracts(ticker, as_of, tries=4):
    # a 429 (rate limit) is not "no contracts": wait and retry, and fail loudly if it persists
    headers = {"Authorization": f"Bearer {os.getenv('POLYGON_API_KEY')}"}
    for i in range(tries):
        r = requests.get("https://api.massive.com/v3/reference/options/contracts", headers=headers, timeout=60,
                         params={"underlying_ticker": ticker, "as_of": as_of, "limit": 1})
        if r.status_code != 429:
            break
        time.sleep(int(r.headers.get("Retry-After", 0)) or 15 * 2 ** i)
    r.raise_for_status()
    return len(r.json().get("results", [])) > 0


with lseg_session():
    # 1) membership (month-end snapshots from April 2015, when LSEG history starts)
    if MEMBERSHIP.exists():
        snaps = pd.read_parquet(MEMBERSHIP)
        print("membership: loaded", MEMBERSHIP)
    else:
        print("membership: downloading month-end snapshots from 2015-04-30 ...")
        snaps = build_membership(start="2015-04-30")
    sizes = snaps.groupby("date").size()
    for d in sizes[sizes < MIN_SIZE].index:
        for attempt in range(3):
            try:
                rics = fetch_members(d.strftime("%Y-%m-%d"))
                if len(rics) >= MIN_SIZE:
                    snaps = pd.concat([snaps[snaps["date"] != d], pd.DataFrame({"date": d, "ric": rics})])
                    print("  repaired", d.date(), len(rics))
                    break
            except Exception as e:
                print("  retry", d.date(), type(e).__name__)
            time.sleep(5)
    snaps = snaps.sort_values(["date", "ric"]).reset_index(drop=True)
    save_membership(snaps)
    sizes = snaps.groupby("date").size()
    print(f"membership: {sizes.size} month-ends {sizes.index.min().date()}..{sizes.index.max().date()}, "
          f"size {sizes.min()}-{sizes.max()}, {snaps['ric'].nunique()} unique RICs")

    # 2) security master
    print("\nsecurity master: fetching company IDs (LSEG) and ticker history (Massive) ...")
    master, tickers = build_security_master(snaps)

# 3) report
print(f"\nRICs: {len(master)} | companies: {master['company_id'].nunique()} | delisted RICs: {master['delisted'].sum()}")
print("unresolved by LSEG (own placeholder ID):", master.loc[master["company_id"].str.startswith("RIC:"), "ric"].tolist())
groups = {k: v for k, v in companies(master["ric"], master).items() if len(v) > 1}
print("companies with more than one RIC (share classes):")
for k, v in groups.items():
    print(f"  {master.loc[master['ric'] == v[0], 'name'].iloc[0]} ({k}): {v}")
print("ticker history source:", tickers.drop_duplicates("ric")["source"].value_counts().to_dict())
changed = tickers.groupby("ric").filter(lambda g: g["massive_ticker"].nunique() > 1)
print(f"RICs with a Massive ticker change since listing: {changed['ric'].nunique()}")
print(changed[changed["valid_from"] >= "2015-01-01"].to_string(index=False))

# 4) validate: does the mapped Massive ticker actually have option contracts on the date?
print("\nvalidating Massive tickers against option listings ...")
checks = []
for ric, g in changed.groupby("ric"):
    for start in g["valid_from"].dropna():
        if start >= pd.Timestamp("2015-06-01"):
            checks += [(ric, start - pd.Timedelta(days=10)), (ric, start + pd.Timedelta(days=10))]
random.seed(0)
member_dates = snaps.groupby("ric")["date"].apply(list).to_dict()
for ric in random.sample(sorted(member_dates), 60):
    checks.append((ric, random.choice(member_dates[ric])))
fails = []
for ric, d in checks:
    t = massive_ticker_on(tickers, ric, d)
    try:
        if not massive_has_contracts(t, d.strftime("%Y-%m-%d")):
            fails.append((ric, d.date(), t))
    except requests.RequestException as e:
        fails.append((ric, d.date(), t, f"request failed: {e}"))
print(f"checked {len(checks)} (RIC, date) pairs: {len(checks) - len(fails)} OK, {len(fails)} with no Massive contracts")
for f in fails:
    print("  NO CONTRACTS:", f)
