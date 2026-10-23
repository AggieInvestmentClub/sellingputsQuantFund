import sys
import time
sys.path.insert(0, ".")
import pandas as pd
from src.ingest.lseg_client import lseg_session
from src.ingest.lseg_universe import fetch_members
from src.ingest.lseg_prices import fetch_prices

SRC = "data/raw/universe/sp500_membership.parquet"
OUT = "data/raw/universe/sp500_membership_clean.parquet"
MIN_SIZE = 450

snaps = pd.read_parquet(SRC)
sizes = snaps.groupby("date").size()

print("lone RIC in the earliest snapshot:")
print(snaps[snaps["date"] == snaps["date"].min()].to_string(index=False))

first_valid = sizes[sizes >= MIN_SIZE].index.min()
print("\nfirst month with >=", MIN_SIZE, "members:", first_valid.date())

expected = pd.date_range(first_valid, snaps["date"].max(), freq="ME")
bad = [d for d in expected if sizes.get(d, 0) < MIN_SIZE]
print("months to repair:", [d.date() for d in bad])

with lseg_session():
    for d in bad:
        for attempt in range(5):
            try:
                rics = fetch_members(d.strftime("%Y-%m-%d"))
                if len(rics) >= MIN_SIZE:
                    snaps = snaps[snaps["date"] != d]
                    snaps = pd.concat([snaps, pd.DataFrame({"date": d, "ric": rics})])
                    print("repaired", d.date(), len(rics))
                    break
            except Exception as e:
                print("retry", d.date(), type(e).__name__)
            time.sleep(5)
        else:
            print("STILL BAD:", d.date())

    for r in [".SPX", "SPY", "SPY.P"]:
        try:
            print(r, "bars:", len(fetch_prices(r, "2026-09-01", "2026-09-15")))
        except Exception as e:
            print(r, "FAILED:", type(e).__name__)

clean = snaps[snaps["date"] >= first_valid]
sizes2 = clean.groupby("date").size()
clean = clean[clean["date"].isin(sizes2[sizes2 >= MIN_SIZE].index)]
clean.sort_values(["date", "ric"]).to_parquet(OUT, index=False)

per_month = clean.groupby("date").size()
delisted = clean.loc[clean["ric"].str.contains("^", regex=False), "ric"].nunique()
print("\nclean snapshots:", clean["date"].nunique(),
      "| unique RICs ever:", clean["ric"].nunique(),
      "| unique delisted (^) RICs:", delisted,
      "| size min/max:", per_month.min(), per_month.max())