# Data Cleaning Report

## Raw data summary

Shape: 5,078,345 rows x 11 columns
Memory usage: 733.5 MB

Dtypes:
  timestamp: str
  from_bank: int64
  from_account: str
  to_bank: int64
  to_account: str
  amount_received: float64
  receiving_currency: str
  amount_paid: float64
  payment_currency: str
  payment_format: str
  is_laundering: int64

Missing values: none

Exact duplicate rows: 9 (0.00%)

## Account id check

4 of 496,995 account numbers (0.00%) appear under more than one bank. This confirms account numbers are NOT globally unique, so combining bank + account into from_id/to_id was necessary.

Sampled to 5% of accounts, keeping every transaction that touches one of them (5,078,345 -> 467,083 rows, sample_mode=accounts). This is done instead of taking the first N% of rows because this dataset is extremely front-loaded in time (99.98% of transactions happen in the first 10 of 18 days -- see the EDA notebook), so a row-count sample would only cover a few minutes and give every account a near-empty history. Sampling by account instead means every included account keeps its FULL history across the whole time range, which the time-window features in Phase 3 need.

## Exact duplicates

Dropped 0 exact duplicate rows (467,083 -> 467,083). These are data entry repeats, not genuine repeated transfers.

## Invalid amounts

Dropped 0 rows with amount_paid or amount_received <= 0.0 (467,083 -> 467,083). A transaction of zero or negative value can't be a real transfer, so these are treated as broken rows rather than genuine data.

## Self-transfers

29,881 rows (6.40%) are self-transfers (same account on both sides). Kept (not dropped) and flagged with a new is_self_transfer column, since this is a real account behaviour pattern, not a data error.

Self-transfer rate by payment_format:
  Reinvestment: 100.0% self-transfers
  Bitcoin: 10.8% self-transfers
  ACH: 6.2% self-transfers
  Wire: 0.5% self-transfers
  Credit Card: 0.2% self-transfers
  Cheque: 0.2% self-transfers
  Cash: 0.1% self-transfers

## Extreme outliers

Outlier amounts are NOT removed. A very large or unusual transaction is exactly the kind of signal we're trying to detect, so deleting it could throw away real laundering cases. Skew is handled later with a log-amount feature (Phase 3) instead of deleting rows here.

amount_paid (USD-labelled rows only, for a quick look): min=0.01, median=927.17, max=16620608535.74

## Inferred exchange rates (to USD)

  Australian Dollar: rate=0.707814, based on 63 rows, std=0.000253
  Bitcoin: rate=11881.298992, based on 171 rows, std=795.281886
  Brazil Real: rate=0.177101, based on 23 rows, std=0.000012 (LOW CONFIDENCE: fewer than 30 supporting rows)
  Canadian Dollar: rate=0.757979, based on 106 rows, std=0.023508
  Euro: rate=1.171783, based on 1,330 rows, std=0.000541
  Mexican Peso: rate=0.047297, based on 90 rows, std=0.000156
  Ruble: rate=0.012853, based on 67 rows, std=0.001271
  Rupee: rate=0.013616, based on 163 rows, std=0.000031
  Saudi Riyal: rate=0.266589, based on 40 rows, std=0.000001
  Shekel: rate=0.296121, based on 151 rows, std=0.003028
  Swiss Franc: rate=1.092896, based on 171 rows, std=0.000251
  UK Pound: rate=1.291656, based on 176 rows, std=0.000018
  Yen: rate=0.009488, based on 215 rows, std=0.000157
  Yuan: rate=0.149307, based on 381 rows, std=0.000018

Sanity check: median relative difference between amount_paid_usd and amount_received_usd is 0.00% (small values here mean the inferred rates are consistent).

## Dtype optimisation

Converted from_bank, to_bank, receiving_currency, payment_currency, payment_format to category dtype, and is_laundering to int8. Memory usage: 92.8 MB -> 63.8 MB.

## Final shape

467,083 rows x 17 columns