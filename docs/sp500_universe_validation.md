# Historical S&P 500 Universe Validation

## Objective
Build a point-in-time S&P 500 constituent universe for historical backtesting, reducing survivorship bias in the put credit spread strategy.

## Dataset
- Built on 2026-10-29 from April 2015 onward because LSEG returns empty results before April 2015.
- Period covered: April 2015 – September 2026
- Membership records: 69,579
- Monthly snapshots: 138
- Unique historical RICs: 736
- RICs marked as delisted: 132
- Constituents per snapshot: 500–506
- Found all S&P 500 companies represented across the 2015-2026 historical window.
- [View Historical S&P 500 Membership CSV](../data/sp500_membership_formatted.csv)

## Implementation
Used the repository's existing LSEG membership ingestion pipeline to retrieve historical S&P 500 constituents. The dataset was built starting at April 2015 because earlier months come back empty from LSEG and do not contain usable membership data.

## Validation
Ran scripts/check_membership.py and confirmed 138 clean monthly snapshots.

The validation run also logged an LSEG error for .SPX; this should be investigated separately.

## Data Storage
The generated raw dataset is stored locally at:
data/raw/universe/sp500_membership.parquet

The exported historical snapshot CSV is stored at:
data/sp500_membership_formatted.csv

Market data is excluded from Git tracking.
