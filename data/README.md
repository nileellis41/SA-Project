# Data (not included)

Raw Bloomberg exports are licensed and not distributed. Place them in `data/raw/`:
`rates_1.csv`, `930.csv`, `credit.csv`, `Features_Costs_1.csv`, `pnt.csv` (layouts documented in
`src/amp_sa/data/load_bbg.py`). Then run `python -m amp_sa.data.load_bbg` to build `data/processed/`.

Public inputs used by the diagnostics only:
- FRED `DFEDTAR`, `DFEDTARU`, `DFEDTARL` -> `data/raw/fred/<SERIES>.csv` (https://fred.stlouisfed.org)
- TreasuryDirect auction results -> `data/raw/treasury/` (https://www.treasurydirect.gov)
- Yahoo Finance grain futures (appendix only) are fetched and cached by `src/amp_sa/data/load_yahoo.py`.
