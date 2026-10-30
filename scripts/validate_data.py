"""
Check the downloaded data before using it: membership (DAT-06), every price file (DAT-05)
and splits from adjusted vs unadjusted prices (DAT-07). No LSEG needed.

Run from the repo root after scripts/overnight.py:  python scripts/validate_data.py
Writes data/results/checks/*.csv and prints a short report.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import pandas as pd

from src.data.validate import membership_issues, price_issues, split_factors

RAW = Path("data/raw")
UNIVERSE, PRICES = RAW / "universe", RAW / "prices" / "daily"
OUT = Path("data/results/checks")
OUT.mkdir(parents=True, exist_ok=True)


def fname(ric):
    return ric.replace(".", "_").replace("^", "~")      # same as scripts/overnight.py


# ---------------- membership ----------------
membership = pd.read_parquet(UNIVERSE / "sp500_membership.parquet")
changes_path = UNIVERSE / "sp500_changes.parquet"
changes = pd.read_parquet(changes_path) if changes_path.exists() else None
sizes = membership.groupby("date").size()
print(f"membership: {sizes.size} snapshots {sizes.index.min().date()}..{sizes.index.max().date()}, "
      f"{len(membership)} rows, {membership['ric'].nunique()} RICs")
if sizes.index.min() < pd.Timestamp("2015-04-30"):
    print("  WARNING: snapshots before 2015-04-30. LSEG history starts there; earlier months have 1 member."
          " Drop them before build_security_master.py / overnight.py.")
mi = membership_issues(membership, changes)
mi.to_csv(OUT / "membership_issues.csv", index=False)
print("  issues by kind:", mi["kind"].value_counts().to_dict() if len(mi) else "none")
if changes is None:
    print("  (no sp500_changes.parquet, so joins / leaves were not cross-checked)")

# ---------------- prices ----------------
master_path = UNIVERSE / "security_master.parquet"
if master_path.exists():
    rics = sorted(pd.read_parquet(master_path)["ric"])
else:
    rics = sorted(membership["ric"].dropna().unique())
    print("\n(no security_master.parquet yet: checking every RIC in the membership table)")

rows, splits, missing = [], [], []
for ric in rics:
    adj_path, raw_path = PRICES / f"{fname(ric)}_adj.parquet", PRICES / f"{fname(ric)}_unadj.parquet"
    if not adj_path.exists():
        missing.append((ric, "not downloaded"))
        continue
    adj = pd.read_parquet(adj_path)
    r = price_issues(adj)
    if r["rows"] == 0:
        missing.append((ric, "empty file"))
    rows.append({"ric": ric, **r})
    if raw_path.exists() and not adj.empty:
        raw = pd.read_parquet(raw_path)
        if not raw.empty and "TRDPRC_1" in raw and "TRDPRC_1" in adj:
            s = split_factors(adj["TRDPRC_1"], raw["TRDPRC_1"])
            if len(s):
                splits.append(s.assign(ric=ric))

pi = pd.DataFrame(rows)
pi.to_csv(OUT / "price_issues.csv", index=False)
pd.DataFrame(missing, columns=["ric", "reason"]).to_csv(OUT / "missing_prices.csv", index=False)
sp = pd.concat(splits, ignore_index=True) if splits else pd.DataFrame(columns=["ric", "date", "split"])
sp.to_csv(OUT / "splits.csv", index=False)

print(f"\nprices: {len(rics)} RICs | files checked: {len(pi)} | not downloaded or empty: {len(missing)}")
if len(pi):
    flagged = pi[(pi["problems"] != "") & (pi["problems"] != "empty")]
    print(f"  files with problems: {len(flagged)}")
    if len(flagged):
        print(flagged[["ric", "rows", "first", "last", "problems"]].head(25).to_string(index=False))
print(f"  price adjustments found (splits, spin-offs; adjusted vs unadjusted): {len(sp)}")
print(f"\nfull lists in {OUT}/ : membership_issues.csv, price_issues.csv, missing_prices.csv, splits.csv")
print("Re-download a missing / empty RIC by deleting its file and re-running scripts/overnight.py.")
