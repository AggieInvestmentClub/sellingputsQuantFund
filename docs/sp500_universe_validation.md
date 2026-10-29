# Historical S&P 500 Universe Validation

## Objective
Build a point-in-time S&P 500 constituent universe for historical backtesting, reducing survivorship bias in the put credit spread strategy.

## Dataset
- Period: April 2015 – September 2026
- Membership records: 69,579
- Monthly snapshots: 138
- Unique historical RICs: 736
- RICs marked as delisted: 132
- Constituents per snapshot: 500–506

## Implementation
Used the repository's existing LSEG membership ingestion pipeline to retrieve historical S&P 500 constituents.

## Validation
Ran scripts/check_membership.py and confirmed 138 clean monthly snapshots.

The validation run also logged an LSEG error for .SPX; this should be investigated separately.

## Data Storage
The generated dataset is stored locally at:
data/raw/universe/sp500_membership.parquet

Market data is excluded from Git tracking.
