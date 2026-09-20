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

Sampled to 5% of rows (253,917 rows), taken from the start (sample_mode=head).

## Account id check

0 of 198,833 account numbers (0.00%) appear under more than one bank. No collisions were found in this sample, but we keep from_id/to_id anyway since the dataset documentation says collisions are possible.

## Exact duplicates

Dropped 0 exact duplicate rows (253,917 -> 253,917). These are data entry repeats, not genuine repeated transfers.

## Invalid amounts

Dropped 0 rows with amount_paid or amount_received <= 0.0 (253,917 -> 253,917). A transaction of zero or negative value can't be a real transfer, so these are treated as broken rows rather than genuine data.

## Self-transfers

185,102 rows (72.90%) are self-transfers (same account on both sides). Kept (not dropped) and flagged with a new is_self_transfer column, since this is a real account behaviour pattern, not a data error.

Self-transfer rate by payment_format:
  Reinvestment: 100.0% self-transfers
  Bitcoin: 74.0% self-transfers
  ACH: 8.1% self-transfers
  Cash: 0.5% self-transfers
  Wire: 0.5% self-transfers
  Cheque: 0.3% self-transfers
  Credit Card: 0.2% self-transfers

## Extreme outliers

Outlier amounts are NOT removed. A very large or unusual transaction is exactly the kind of signal we're trying to detect, so deleting it could throw away real laundering cases. Skew is handled later with a log-amount feature (Phase 3) instead of deleting rows here.

amount_paid (USD-labelled rows only, for a quick look): min=0.01, median=2049.58, max=5351188958.74

## Inferred exchange rates (to USD)

  Australian Dollar: rate=0.707814, based on 33 rows, std=0.036175
  Bitcoin: rate=11881.606765, based on 19 rows, std=2650.769780 (LOW CONFIDENCE: fewer than 30 supporting rows)
  Brazil Real: rate=0.177101, based on 14 rows, std=0.000008 (LOW CONFIDENCE: fewer than 30 supporting rows)
  Canadian Dollar: rate=0.757978, based on 37 rows, std=0.000028
  Euro: rate=1.171783, based on 320 rows, std=0.000685
  Mexican Peso: rate=0.047297, based on 16 rows, std=0.002896 (LOW CONFIDENCE: fewer than 30 supporting rows)
  Ruble: rate=0.012853, based on 14 rows, std=0.000000 (LOW CONFIDENCE: fewer than 30 supporting rows)
  Rupee: rate=0.013616, based on 50 rows, std=0.000044
  Saudi Riyal: rate=0.266588, based on 9 rows, std=0.000000 (LOW CONFIDENCE: fewer than 30 supporting rows)
  Shekel: rate=0.296121, based on 26 rows, std=0.000000 (LOW CONFIDENCE: fewer than 30 supporting rows)
  Swiss Franc: rate=1.092896, based on 44 rows, std=0.000015
  UK Pound: rate=1.291656, based on 41 rows, std=0.000220
  Yen: rate=0.009488, based on 55 rows, std=0.000931
  Yuan: rate=0.149307, based on 108 rows, std=0.004898

Sanity check: median relative difference between amount_paid_usd and amount_received_usd is 0.00% (small values here mean the inferred rates are consistent).

## Dtype optimisation

Converted from_bank, to_bank, receiving_currency, payment_currency, payment_format to category dtype, and is_laundering to int8. Memory usage: 51.5 MB -> 34.9 MB.

## Final shape

253,917 rows x 17 columns